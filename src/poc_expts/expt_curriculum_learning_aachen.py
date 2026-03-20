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


def analyze_label1_composition(signal_path, supp_bg_path, n_signal, n_supp_bg, n_trials=100):
    """
    Analyze the composition of Label 1 (real signal + suppressed background) 
    across random splits.
    
    When we randomly divide Label 1 into two groups, what fraction of REAL 
    signal ends up in each group? This tells us about the statistical variability
    in signal distribution when we randomly split the weak labels.
    
    Parameters
    ----------
    signal_path : str
        Path to signal H5 file
    supp_bg_path : str
        Path to suppressed background H5 file
    n_signal : int
        Number of real signal jets
    n_supp_bg : int
        Number of suppressed background jets
    n_trials : int
        Number of random splits to test (default: 100)
        
    Returns
    -------
    dict
        Statistics on signal distribution across random splits
    """
    from gabbro.data.loading import load_lhco_jets_from_h5
    
    print(f"\n{'='*80}")
    print(f"ANALYZING LABEL 1 COMPOSITION")
    print(f"{'='*80}\n")
    
    # Create index arrays representing Label 1
    signal_indices = np.arange(n_signal)
    supp_bg_indices = np.arange(n_signal, n_signal + n_supp_bg)
    
    # Combine into single Label 1 pool
    label1_indices = np.concatenate([signal_indices, supp_bg_indices])
    total_label1 = len(label1_indices)
    
    print(f"Label 1 Composition:")
    print(f"  Real signal jets: {n_signal}")
    print(f"  Suppressed background (fake signal): {n_supp_bg}")
    print(f"  Total Label 1 jets: {total_label1}")
    print(f"\nPerforming {n_trials} random splits...\n")
    
    # Track signal distribution across random splits
    signal_in_group1 = []  # Fraction of real signals in first group
    signal_in_group2 = []  # Fraction of real signals in second group
    
    split_size = total_label1 // 2
    
    for trial in range(n_trials):
        # Randomly shuffle Label 1 pool
        shuffled_indices = np.random.permutation(label1_indices)
        
        # Split into two groups
        group1 = shuffled_indices[:split_size]
        group2 = shuffled_indices[split_size:]
        
        # Count real signals in each group
        real_signal_in_g1 = np.sum(group1 < n_signal)
        real_signal_in_g2 = np.sum(group2 < n_signal)
        
        # Calculate fractions
        frac_g1 = real_signal_in_g1 / split_size
        frac_g2 = real_signal_in_g2 / split_size
        
        signal_in_group1.append(frac_g1)
        signal_in_group2.append(frac_g2)
    
    # Compute statistics
    signal_in_group1 = np.array(signal_in_group1)
    signal_in_group2 = np.array(signal_in_group2)
    
    results = {
        "total_trials": n_trials,
        "group_size": split_size,
        "total_label1": total_label1,
        "n_real_signal": n_signal,
        "n_fake_signal": n_supp_bg,
        
        "group1_signal_fraction": {
            "mean": float(np.mean(signal_in_group1)),
            "std": float(np.std(signal_in_group1)),
            "min": float(np.min(signal_in_group1)),
            "max": float(np.max(signal_in_group1)),
            "percentile_5": float(np.percentile(signal_in_group1, 5)),
            "percentile_25": float(np.percentile(signal_in_group1, 25)),
            "percentile_50": float(np.percentile(signal_in_group1, 50)),
            "percentile_75": float(np.percentile(signal_in_group1, 75)),
            "percentile_95": float(np.percentile(signal_in_group1, 95)),
        },
        
        "group2_signal_fraction": {
            "mean": float(np.mean(signal_in_group2)),
            "std": float(np.std(signal_in_group2)),
            "min": float(np.min(signal_in_group2)),
            "max": float(np.max(signal_in_group2)),
            "percentile_5": float(np.percentile(signal_in_group2, 5)),
            "percentile_25": float(np.percentile(signal_in_group2, 25)),
            "percentile_50": float(np.percentile(signal_in_group2, 50)),
            "percentile_75": float(np.percentile(signal_in_group2, 75)),
            "percentile_95": float(np.percentile(signal_in_group2, 95)),
        },
    }
    
    # Print results
    print(f"{'='*80}")
    print(f"RESULTS: Random Split Analysis of Real Signals in Label 1")
    print(f"{'='*80}\n")
    
    print(f"Each random split divides {total_label1} Label-1 jets into two groups of {split_size} each")
    
    print(f"\n┌─ GROUP 1 Real Signal Fraction ──────────────────────────┐")
    print(f"│ Mean:  {results['group1_signal_fraction']['mean']:.4f} (≈ {results['group1_signal_fraction']['mean']*split_size:.1f} real signals)")
    print(f"│ Std:   {results['group1_signal_fraction']['std']:.4f}")
    print(f"│ Range: {results['group1_signal_fraction']['min']:.4f} - {results['group1_signal_fraction']['max']:.4f}")
    print(f"│ ")
    print(f"│ Percentiles:")
    print(f"│   5th: {results['group1_signal_fraction']['percentile_5']:.4f}")
    print(f"│  25th: {results['group1_signal_fraction']['percentile_25']:.4f}")
    print(f"│  50th: {results['group1_signal_fraction']['percentile_50']:.4f}")
    print(f"│  75th: {results['group1_signal_fraction']['percentile_75']:.4f}")
    print(f"│  95th: {results['group1_signal_fraction']['percentile_95']:.4f}")
    print(f"└─────────────────────────────────────────────────────────┘\n")
    
    print(f"┌─ GROUP 2 Real Signal Fraction ──────────────────────────┐")
    print(f"│ Mean:  {results['group2_signal_fraction']['mean']:.4f} (≈ {results['group2_signal_fraction']['mean']*split_size:.1f} real signals)")
    print(f"│ Std:   {results['group2_signal_fraction']['std']:.4f}")
    print(f"│ Range: {results['group2_signal_fraction']['min']:.4f} - {results['group2_signal_fraction']['max']:.4f}")
    print(f"│ ")
    print(f"│ Percentiles:")
    print(f"│   5th: {results['group2_signal_fraction']['percentile_5']:.4f}")
    print(f"│  25th: {results['group2_signal_fraction']['percentile_25']:.4f}")
    print(f"│  50th: {results['group2_signal_fraction']['percentile_50']:.4f}")
    print(f"│  75th: {results['group2_signal_fraction']['percentile_75']:.4f}")
    print(f"│  95th: {results['group2_signal_fraction']['percentile_95']:.4f}")
    print(f"└─────────────────────────────────────────────────────────┘\n")
    
    print(f"INTERPRETATION:")
    print(f"  • If distributions are symmetric: random split distributes signals evenly")
    print(f"  • Expected mean: {n_signal / total_label1:.4f} (uniform distribution)")
    print(f"  • Actual mean difference from expected: {abs(results['group1_signal_fraction']['mean'] - n_signal/total_label1):.4f}")
    print(f"\n{'='*80}\n")
    
    return results


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
    
    # n_jets_train = [signal_real, supp_bg_labeled_as_signal, background_real]
    # Actual label distribution after loading:
    #   - Label 1: signal_real + supp_bg_labeled_as_signal  
    #   - Label 0: background_real
    n_label_1 = args.n_jets_train[0] + args.n_jets_train[1]  # signal + supp background
    n_label_0 = args.n_jets_train[2]  # clean background
    total = n_label_1 + n_label_0
    
    # Analyze the composition of Label 1 across random splits
    print(f"\n{'='*80}")
    print(f"DATA LOADING COMPLETE - ANALYZING LABEL 1 COMPOSITION")
    print(f"{'='*80}")
    label1_analysis = analyze_label1_composition(
        signal_path=signal_path,
        supp_bg_path=supp_background_path,
        n_signal=args.n_jets_train[0],
        n_supp_bg=args.n_jets_train[1],
        n_trials=100
    )


if __name__ == "__main__":
    main()
