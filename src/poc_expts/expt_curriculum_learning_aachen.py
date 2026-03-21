""" Python file to train the OmniJet-alpha model with continuous tokens - Aachen anomaly detection variant.

PROJECT STRUCTURE:
AnomalyDetection/
├── src/
│   ├── train/              # Training scripts (e.g., train.py, train_custom_aachen.py)
│   ├── eval/               # Evaluation scripts (evaluate.py, evaluate_true_roc.py, etc.)
│   ├── data/               # Data processing (split_h5_dataset.py)
│   ├── viz/                # Visualization (unsupervised_learning.py, visualize_clustering.py)
│   └── poc_expts/          # Proof-of-concept experiments (this directory)
├── scripts/                # Job launchers (.sh and .sub files)
├── gabbro/                 # Core library (models, data utilities, metrics)
└── [output directories: plots/, results/, logs/, checkpoints/, dijet_expts/, aachen_head_expts/]

COMPONENTS USED:
├── Data Loading
│   ├── load_multiple_h5_files
│   ├── ak_select_and_preprocess
│   ├── ak_pad
│   └── JetDataset
│
├── Model Architecture
│   ├── BackboneAachenClassificationLightning
│   │   ├── BackboneTransformer (use_continuous_input=True)
│   │   └── AachenClassificationHead (class_head_type="aachen")
│   │
│   └── Loss: CrossEntropyLoss
│
└── Training
    ├── PyTorch Lightning Trainer
    ├── AdamW + ConstantLR Scheduler
    ├── AUC & ARGOS metric callbacks
    └── Weights & Biases logging (optional)

USAGE:
Run from project root with:
  python -m src.poc_expts.train_custom_aachen [args]

Or via shell launcher:
  ./scripts/train_aachen.sh

DESCRIPTION:
Trains a custom anomaly detection model on LHCO datasets using weak supervision.
- Training: 200k clean background + 100k polluted background + 2k-25k signal jets
- Testing: 200k clean background + 50k signal jets
- Merge strategies: concat, average, weighted_sum, attention
- Aachen head: Two-jet model for anomaly detection
"""
# imports
import os
import json
import torch
import argparse
import numpy as np
import awkward as ak
import lightning as L
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, roc_curve
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import wandb

# gabbro imports
from gabbro.utils.arrays import ak_pad
from gabbro.data.data_utils import create_custom_lhco_h5_dataloaders
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

load_dotenv()  # Load environment variables from .env file (for W&B API key, etc.)


class CurriculumSplitter:
    """
    Manages curriculum learning label reassignments across 3 phases.
    
    Phase 1 (Epoch 0-9): Contaminated-only split
      - Label 0 = Half of contaminated (Label 1)
      - Label 1 = Other half of contaminated
    
    Phase 2 (Epoch 10-100): Gradual restoration
      - Pure background gradually returns to Label 0
      - Contaminated from new Label 0 gradually returns to Label 1
    
    Phase 3 (Epoch 100+): Back to original
      - Label 0 = pure background
      - Label 1 = all contaminated
    """
    
    def __init__(self, n_real_signal, n_supp_bg, n_pure_bg, seed=42):
        """
        Parameters
        ----------
        n_real_signal : int
            Number of real signal jets in Label 1
        n_supp_bg : int
            Number of suppressed (fake signal) background jets in Label 1
        n_pure_bg : int
            Number of pure background jets in original Label 0
        seed : int
            Random seed for reproducible split
        """
        np.random.seed(seed)
        
        self.n_real_signal = n_real_signal
        self.n_supp_bg = n_supp_bg
        self.n_pure_bg = n_pure_bg
        self.total_contaminated = n_real_signal + n_supp_bg
        
        # Create index pools
        contaminated_indices = np.arange(self.total_contaminated)
        np.random.shuffle(contaminated_indices)
        
        # Split contaminated in half for Phase 1
        split_point = self.total_contaminated // 2
        self.contaminated_half_a = contaminated_indices[:split_point]
        self.contaminated_half_b = contaminated_indices[split_point:]
        
        self.pure_bg_indices = np.arange(self.total_contaminated, 
                                        self.total_contaminated + n_pure_bg)
        
        self.current_epoch = 0
        self.epoch_log = []
        
        print(f"\n{'='*80}")
        print(f"CURRICULUM LEARNING SETUP")
        print(f"{'='*80}")
        print(f"Contaminated pool (signal + fake): {self.total_contaminated} jets")
        print(f"  - Phase 1 split A: {len(self.contaminated_half_a)} jets")
        print(f"  - Phase 1 split B: {len(self.contaminated_half_b)} jets")
        print(f"Pure background pool: {n_pure_bg} jets")
        print(f"\nPhase 1 (Epoch 0-9): Contaminated-only split")
        print(f"Phase 2 (Epoch 10-100): Gradual restoration")
        print(f"Phase 3 (Epoch 100+): Back to original labels")
        print(f"{'='*80}\n")
    
    def get_labels_for_epoch(self, epoch):
        """
        Returns label assignments for all samples at a given epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
        
        Returns
        -------
        dict with keys:
          - 'label_0': Array of indices assigned to Label 0
          - 'label_1': Array of indices assigned to Label 1
          - 'phase': Current phase (1, 2, or 3)
          - 'progress': Progress through Phase 2 (0 to 1)
        """
        self.current_epoch = epoch
        
        if epoch < 10:
            # Phase 1: Contaminated-only split
            label_0 = self.contaminated_half_a.copy()
            label_1 = self.contaminated_half_b.copy()
            phase = 1
            progress = 0.0
        
        elif epoch <= 100:
            # Phase 2: Gradual restoration
            progress = (epoch - 10) / 90.0  # Linear: 0->1
            
            # Pure background gradually returns to Label 0
            n_pure_to_label0 = int(self.n_pure_bg * progress)
            pure_to_label0 = self.pure_bg_indices[:n_pure_to_label0]
            
            # Remaining pure background (if any) stays in Label 1
            pure_remaining = self.pure_bg_indices[n_pure_to_label0:]
            
            # Contaminated half A: gradually transitions from Label 0 to Label 1
            n_half_a_to_label1 = int(len(self.contaminated_half_a) * progress)
            half_a_to_label1 = self.contaminated_half_a[:n_half_a_to_label1]
            half_a_to_label0 = self.contaminated_half_a[n_half_a_to_label1:]
            
            label_0 = np.concatenate([half_a_to_label0, pure_to_label0])
            label_1 = np.concatenate([half_a_to_label1, self.contaminated_half_b, pure_remaining])
            
            phase = 2
        
        else:
            # Phase 3: Back to original
            label_0 = np.concatenate([self.pure_bg_indices])
            label_1 = np.concatenate([self.contaminated_half_a, self.contaminated_half_b])
            phase = 3
            progress = 1.0
        
        result = {
            'label_0': label_0,
            'label_1': label_1,
            'phase': phase,
            'progress': progress,
            'n_label_0': len(label_0),
            'n_label_1': len(label_1),
        }
        
        # Log for analysis
        self.epoch_log.append({
            'epoch': epoch,
            'phase': phase,
            'progress': progress,
            'n_label_0': len(label_0),
            'n_label_1': len(label_1),
        })
        
        return result
    
    def print_status(self, epoch):
        """Print curriculum status at current epoch."""
        result = self.get_labels_for_epoch(epoch)
        print(f"Epoch {epoch:3d} | Phase {result['phase']} | Progress: {result['progress']:.2%} | "
              f"Label 0: {result['n_label_0']:5d} | Label 1: {result['n_label_1']:5d}")
    
    def save_log(self, filepath):
        """Save epoch log to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.epoch_log, f, indent=2)


class CurriculumBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that dynamically assigns labels based on curriculum epoch.
    """
    
    def __init__(self, splitter, dataset_size, batch_size, shuffle=True):
        """
        Parameters
        ----------
        splitter : CurriculumSplitter
            Curriculum scheduler
        dataset_size : int
            Total number of samples in dataset
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle within epoch
        """
        self.splitter = splitter
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update epoch for curriculum progression."""
        self.current_epoch = epoch
    
    def __iter__(self):
        """Generate indices for current epoch with curriculum labels."""
        # Get curriculum label assignment for this epoch
        curriculum = self.splitter.get_labels_for_epoch(self.current_epoch)
        indices = np.concatenate([curriculum['label_0'], curriculum['label_1']])
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.dataset_size


class CurriculumCallback(Callback):
    """
    PyTorch Lightning callback to manage curriculum progression.
    """
    
    def __init__(self, splitter, verbose=True):
        super().__init__()
        self.splitter = splitter
        self.verbose = verbose
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update sampler and print status at epoch start."""
        epoch = trainer.current_epoch
        
        # Update sampler
        if hasattr(trainer.train_dataloader.sampler, 'set_epoch'):
            trainer.train_dataloader.sampler.set_epoch(epoch)
        
        if self.verbose and epoch % 5 == 0:
            self.splitter.print_status(epoch)


def main():
    parser = argparse.ArgumentParser(description="OmniJet-alpha Anomaly Detection Training Script")
    parser.add_argument("--dataset_path", default=str(os.getenv("DATASET_PATH")), type=str, help="Path to the LHCO dataset")
    parser.add_argument("--gpu_id", type=int, default=int(os.getenv("GPU_ID")), help="GPU ID to use for computation")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED")), help="Random seed for reproducibility")
    parser.add_argument("--jet_name", type=str, default=str(os.getenv("JET_NAME")), choices=["jet1", "jet2", "both"], help="Name of the jet to use from the dataset")
    parser.add_argument("--merge_strategy", type=str, default=str(os.getenv("MERGE_STRATEGY")), choices=["concat", "average", "weighted_sum", "attention"], help="Merge strategy for dijet model")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE")), help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("MAX_STEPS")), help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE")), help="Learning rate")
    parser.add_argument("--train_val_split", type=float, default=float(os.getenv("TRAIN_VAL_SPLIT")), help="Train/validation split ratio")
    parser.add_argument("--n_jets_train", type=list, default=list(map(int,os.getenv("N_JETS_TRAIN_CUSTOM_AACHEN").strip('[]').split(','))), help="Number of jets per class for training [signal, supp, background]")
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM")), help="Embedding dimension")
    parser.add_argument("--naming_identifier", type=str, default="", help="Optional identifier to add to the run name for easier tracking")
    parser.add_argument("--log_dir", type=str, default=str(os.getenv("LOG_DIR_AACHEN")), help="Directory for experiment logs")
    parser.add_argument("--pretrained_ckpt", type=str, help="Path to pre-trained checkpoint")
    parser.add_argument("--load_pretrained", action="store_true", help="Load pre-trained backbone weights from checkpoint")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == 'true', default=True, help="Use automatic class weighting for imbalanced data (default: True)")

    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="anomaly-detection-lhco", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional, auto-generated if not provided)")
    
    args = parser.parse_args()

    # ============================================================
    # 2. Load Data
    # ============================================================

    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }

    signal_path = os.path.join(args.dataset_path, "sn_25k_SR_train.h5")
    supp_background_path = os.path.join(args.dataset_path, "bg_100k_SR_supp.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    
    h5_files_all = [signal_path, supp_background_path, background_path]
    print("n_jets_train:", args.n_jets_train)
    print("Using Jet:", args.jet_name)
    
    train_loader, val_loader = create_custom_lhco_h5_dataloaders(
        h5_files_train=h5_files_all,
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=args.batch_size,
        n_jets_train=args.n_jets_train,  # [signal, background]
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        train_val_split=args.train_val_split,
        shuffle_train=True,
        num_workers=1,
    )
    
    # ============================================================
    # 3. Curriculum Learning Setup
    # ============================================================
    # n_jets_train = [signal_real, supp_bg_labeled_as_signal, background_real]
    n_real_signal = args.n_jets_train[0]
    n_supp_bg = args.n_jets_train[1]
    n_pure_bg = args.n_jets_train[2]
    
    # Initialize curriculum scheduler
    curriculum_splitter = CurriculumSplitter(
        n_real_signal=n_real_signal,
        n_supp_bg=n_supp_bg,
        n_pure_bg=n_pure_bg,
        seed=args.seed
    )
    
    # Add curriculum progress callback
    curriculum_callback = CurriculumCallback(curriculum_splitter, verbose=True)


if __name__ == "__main__":
    main()
