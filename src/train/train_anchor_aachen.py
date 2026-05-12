# Anchor Signal Injection
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
from gabbro.utils.arrays import ak_pad, ak_to_np_stack
from gabbro.data.data_utils import create_custom_lhco_h5_dataloaders
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

load_dotenv()  # Load environment variables from .env file (for W&B API key, etc.)


# ============================================================================
# CUSTOM BATCHING WITH GUARANTEED SIGNAL INJECTION
# ============================================================================

class GuaranteedSignalBatchSampler:
    """
    Custom batch sampler that guarantees a fixed number of true signal jets
    in every batch (CWoLa diagnostic ablation study).
    
    Batch structure (batch_size=64):
    - Label 0 (Pure QCD): 32 jets from clean background pool
    - Label 1 (Mixed/Signal): (32-k) random QCD from supp pool + k true signal jets
    where k is the number of guaranteed signal jets per batch.
    """
    
    def __init__(self, signal_indices, supp_bg_indices, clean_bg_indices, 
                 batch_size=64, guaranteed_signal_per_batch=1, shuffle=True):
        """
        Parameters
        ----------
        signal_indices : np.ndarray
            Indices of true signal jets
        supp_bg_indices : np.ndarray
            Indices of suppressed background (labeled as signal in CWoLa)
        clean_bg_indices : np.ndarray
            Indices of clean background (labeled as background)
        batch_size : int
            Total batch size (default: 64)
        guaranteed_signal_per_batch : int
            Number of true signal jets guaranteed per batch (default: 1)
        shuffle : bool
            Whether to shuffle indices each epoch
        """
        self.signal_indices = signal_indices
        self.supp_bg_indices = supp_bg_indices
        self.clean_bg_indices = clean_bg_indices
        self.batch_size = batch_size
        self.k = guaranteed_signal_per_batch
        self.shuffle = shuffle
        
        # Sanity checks
        if self.k > self.batch_size // 2:
            raise ValueError(f"k ({self.k}) cannot exceed batch_size//2 ({self.batch_size//2})")
        if self.batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even (got {self.batch_size})")
        
        self.half_batch = self.batch_size // 2  # 32
        self.supp_bg_per_batch = self.half_batch - self.k  # 32-k
        
        # Compute number of batches we can create
        min_clean_batches = len(self.clean_bg_indices) // self.half_batch
        min_supp_batches = len(self.supp_bg_indices) // self.supp_bg_per_batch
        min_signal_batches = len(self.signal_indices) // self.k
        
        self.num_batches = min(min_clean_batches, min_supp_batches, min_signal_batches)
        
        print(f"\n{'='*80}")
        print(f"GuaranteedSignalBatchSampler Configuration:")
        print(f"{'='*80}")
        print(f"Batch size: {self.batch_size}")
        print(f"Guaranteed signal per batch (k): {self.k}")
        print(f"Suppressed background per batch: {self.supp_bg_per_batch}")
        print(f"Clean background per batch: {self.half_batch}")
        print(f"\nPool sizes:")
        print(f"  - Signal jets: {len(self.signal_indices)}")
        print(f"  - Suppressed background: {len(self.supp_bg_indices)}")
        print(f"  - Clean background: {len(self.clean_bg_indices)}")
        print(f"\nEstimated batches per epoch: {self.num_batches}")
        print(f"Total jets per epoch: {self.num_batches * self.batch_size}")
        print(f"{'='*80}\n")
    
    def __iter__(self):
        """Yield batches of indices."""
        # Copy indices for shuffling
        signal_idx = self.signal_indices.copy()
        supp_bg_idx = self.supp_bg_indices.copy()
        clean_bg_idx = self.clean_bg_indices.copy()
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(signal_idx)
            np.random.shuffle(supp_bg_idx)
            np.random.shuffle(clean_bg_idx)
        
        # Generate batches
        for batch_num in range(self.num_batches):
            start_signal = batch_num * self.k
            end_signal = start_signal + self.k
            
            start_supp = batch_num * self.supp_bg_per_batch
            end_supp = start_supp + self.supp_bg_per_batch
            
            start_clean = batch_num * self.half_batch
            end_clean = start_clean + self.half_batch
            
            # Construct batch: [clean_bg (label 0), signal + supp_bg (label 1)]
            batch = np.concatenate([
                clean_bg_idx[start_clean:end_clean],        # label 0
                signal_idx[start_signal:end_signal],        # label 1 (true signal)
                supp_bg_idx[start_supp:end_supp],           # label 1 (fake signal)
            ])
            
            yield batch
    
    def __len__(self):
        return self.num_batches


class GuaranteedSignalDataset(torch.utils.data.Dataset):
    """
    Dataset that uses GuaranteedSignalBatchSampler to create batches with
    guaranteed signal injection for diagnostic ablation studies.
    """
    
    def __init__(self, signal_data, supp_bg_data, clean_bg_data, batch_size=64, 
                 guaranteed_signal_per_batch=1, shuffle=True):
        """
        Parameters
        ----------
        signal_data : dict
            Dictionary with tensors: part_features, part_mask, jet_type_labels
        supp_bg_data : dict
            Dictionary with tensors: part_features, part_mask, jet_type_labels
        clean_bg_data : dict
            Dictionary with tensors: part_features, part_mask, jet_type_labels
        batch_size : int
            Total batch size
        guaranteed_signal_per_batch : int
            Number of guaranteed signal jets per batch
        shuffle : bool
            Whether to shuffle indices
        """
        self.signal_data = signal_data
        self.supp_bg_data = supp_bg_data
        self.clean_bg_data = clean_bg_data
        self.batch_size = batch_size
        self.k = guaranteed_signal_per_batch
        
        # Create index arrays
        self.signal_indices = np.arange(len(signal_data["part_features"]))
        self.supp_bg_indices = np.arange(len(supp_bg_data["part_features"]))
        self.clean_bg_indices = np.arange(len(clean_bg_data["part_features"]))
        
        # Create sampler
        self.sampler = GuaranteedSignalBatchSampler(
            self.signal_indices, self.supp_bg_indices, self.clean_bg_indices,
            batch_size=batch_size, guaranteed_signal_per_batch=guaranteed_signal_per_batch,
            shuffle=shuffle
        )
        
        # Pre-compute all batches to avoid re-computation during training
        self.batches = []
        for batch_indices in self.sampler:
            self.batches.append(batch_indices)
        
        print(f"Pre-computed {len(self.batches)} batches for training")
        
        # Now set number of batches for epoch computation
        self.num_batches = len(self.batches)
    
    def __len__(self):
        """Return number of batches (not individual samples)."""
        return self.num_batches
    
    def __getitem__(self, batch_idx):
        """
        Returns a full batch as a dictionary, NOT a single sample.
        This is meant to be used with batch_size=1 in DataLoader.
        """
        indices = self.batches[batch_idx]
        
        # Reconstruct the batch from the three pools
        batch_dict = {
            "part_features": [],
            "part_mask": [],
            "jet_type_labels": [],
        }
        
        for idx in indices:
            # Determine which pool this index belongs to
            if idx in self.signal_indices:
                pool_idx = idx
                data = self.signal_data
                label = 1  # True signal
            elif idx in self.supp_bg_indices:
                pool_idx = idx
                data = self.supp_bg_data
                label = 1  # Fake signal (suppressed background)
            else:
                pool_idx = idx
                data = self.clean_bg_data
                label = 0  # Clean background
            
            # Actually, we need to handle this differently
            # The indices are LOCAL to each pool, not global
            # Let me fix this approach
            pass
        
        # This approach is getting complicated. Let me use a simpler collate function instead.
        raise NotImplementedError("Use collate_fn with DataLoader instead")


def create_guaranteed_signal_dataloaders(
        signal_path, supp_bg_path, clean_bg_path,
        feature_dict, batch_size=64, guaranteed_signal_per_batch=1,
        n_signal=None, n_supp_bg=None, n_clean_bg=None,
        max_sequence_len=128, mom4_format="epxpypz",
        jet_name="both", train_val_split=0.8, shuffle_train=True,
        num_workers=1, injection_probability=1.0,
):
    """
    Create data loaders with guaranteed signal injection for CWoLa diagnostic study.
    
    Parameters
    ----------
    signal_path : str
        Path to signal H5 file
    supp_bg_path : str
        Path to suppressed background H5 file
    clean_bg_path : str
        Path to clean background H5 file
    feature_dict : dict
        Feature preprocessing dictionary
    batch_size : int
        Physical batch size (default 64)
    guaranteed_signal_per_batch : int
        Number of guaranteed true signal per batch (k, default 1)
    n_signal, n_supp_bg, n_clean_bg : int, optional
        Number of jets to load from each file (None = all)
    max_sequence_len : int
        Maximum sequence length for padding
    mom4_format : str
        4-momentum format
    jet_name : str
        "jet1", "jet2", or "both"
    train_val_split : float
        Fraction for training (remaining for validation)
    shuffle_train : bool
        Whether to shuffle training batches
    num_workers : int
        Number of workers for data loading
        
    Returns
    -------
    train_loader, val_loader : tuple of DataLoaders
    """
    print(f"\n{'='*80}")
    print(f"Creating Guaranteed Signal Injection Data Loaders")
    print(f"{'='*80}\n")
    
    # Load all three datasets separately with their original labels
    feature_names = list(feature_dict.keys())
    
    def load_and_prepare_data(filepath, n_jets, dataset_type="signal"):
        """Load data from H5 and prepare tensors."""
        print(f"Loading {dataset_type} from {filepath}...")
        
        if jet_name == "both":
            features_jet1, features_jet2, labels = load_lhco_jets_from_h5(
                h5_filename=filepath,
                feature_dict=feature_dict,
                n_jets=n_jets,
                jet_name=jet_name,
                mom4_format=mom4_format,
            )
            
            # Pad
            feat_jet1_padded, mask_jet1 = ak_pad(
                features_jet1, maxlen=max_sequence_len, return_mask=True
            )
            feat_jet2_padded, mask_jet2 = ak_pad(
                features_jet2, maxlen=max_sequence_len, return_mask=True
            )
            
            # Stack
            feat_jet1_stacked = ak.concatenate(
                [feat_jet1_padded[feat][..., np.newaxis] for feat in feature_names],
                axis=-1
            )
            feat_jet2_stacked = ak.concatenate(
                [feat_jet2_padded[feat][..., np.newaxis] for feat in feature_names],
                axis=-1
            )
            
            # Convert to torch
            jet1_tensor = torch.from_numpy(ak.to_numpy(feat_jet1_stacked)).float()
            mask1_tensor = torch.from_numpy(ak.to_numpy(mask_jet1)).float()
            jet2_tensor = torch.from_numpy(ak.to_numpy(feat_jet2_stacked)).float()
            mask2_tensor = torch.from_numpy(ak.to_numpy(mask_jet2)).float()
            labels_tensor = torch.from_numpy(labels).long()
            
            return {
                "part_features": jet1_tensor,
                "part_mask": mask1_tensor,
                "part_features_jet2": jet2_tensor,
                "part_mask_jet2": mask2_tensor,
                "jet_type_labels": labels_tensor,
            }
        
        else:  # jet_name in ["jet1", "jet2"]
            features, labels = load_lhco_jets_from_h5(
                h5_filename=filepath,
                feature_dict=feature_dict,
                n_jets=n_jets,
                jet_name=jet_name,
                mom4_format=mom4_format,
            )
            
            # Pad
            feat_padded, mask = ak_pad(
                features, maxlen=max_sequence_len, return_mask=True
            )
            
            # Stack
            feat_stacked = ak.concatenate(
                [feat_padded[feat][..., np.newaxis] for feat in feature_names],
                axis=-1
            )
            
            # Convert to torch
            feat_tensor = torch.from_numpy(ak.to_numpy(feat_stacked)).float()
            mask_tensor = torch.from_numpy(ak.to_numpy(mask)).float()
            labels_tensor = torch.from_numpy(labels).long()
            
            return {
                "part_features": feat_tensor,
                "part_mask": mask_tensor,
                "jet_type_labels": labels_tensor,
            }
    
    # Load all three datasets
    signal_data = load_and_prepare_data(signal_path, n_signal, "signal")
    supp_bg_data = load_and_prepare_data(supp_bg_path, n_supp_bg, "suppressed background")
    clean_bg_data = load_and_prepare_data(clean_bg_path, n_clean_bg, "clean background")
    
    print(f"\nDataset sizes:")
    print(f"  Signal: {len(signal_data['part_features'])} jets")
    print(f"  Suppressed BG: {len(supp_bg_data['part_features'])} jets")
    print(f"  Clean BG: {len(clean_bg_data['part_features'])} jets")
    
    # Create train/val split for each dataset separately
    def train_val_split_data(data, split_ratio):
        """Split data into train and validation."""
        n = len(data["part_features"])
        n_train = int(n * split_ratio)
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_dict = {}
        val_dict = {}
        for key, value in data.items():
            train_dict[key] = value[train_idx]
            val_dict[key] = value[val_idx]
        
        return train_dict, val_dict
    
    signal_train, signal_val = train_val_split_data(signal_data, train_val_split)
    supp_bg_train, supp_bg_val = train_val_split_data(supp_bg_data, train_val_split)
    clean_bg_train, clean_bg_val = train_val_split_data(clean_bg_data, train_val_split)
    
    print(f"\nTrain/Val split ({train_val_split:.0%}/{1-train_val_split:.0%}):")
    print(f"  Signal: {len(signal_train['part_features'])}/{len(signal_val['part_features'])}")
    print(f"  Supp BG: {len(supp_bg_train['part_features'])}/{len(supp_bg_val['part_features'])}")
    print(f"  Clean BG: {len(clean_bg_train['part_features'])}/{len(clean_bg_val['part_features'])}")
    
    # Create custom batch samplers
    def create_guaranteed_signal_loader(signal_dict, supp_bg_dict, clean_bg_dict, 
                                       batch_size, guaranteed_signal, shuffle,
                                       injection_probability=1.0):
        """Create a dataloader with guaranteed signal batching."""
        
        class BatchCollator:
            """Collate function that constructs batches with guaranteed signal."""
            def __init__(self, signal_dict, supp_bg_dict, clean_bg_dict, 
                        batch_size, guaranteed_signal, do_shuffle, injection_probability=1.0):
                self.signal_dict = signal_dict
                self.supp_bg_dict = supp_bg_dict
                self.clean_bg_dict = clean_bg_dict
                self.batch_size = batch_size
                self.k = guaranteed_signal
                self.half_batch = batch_size // 2
                self.supp_per_batch = self.half_batch - self.k
                self.do_shuffle = do_shuffle
                self.injection_probability = injection_probability
                
                # Compute max batches per epoch
                # Signal is sampled with replacement so it does not limit epoch length
                n_supp = len(supp_bg_dict["part_features"])
                n_clean = len(clean_bg_dict["part_features"])
                
                self.max_batches = min(
                    n_supp // self.supp_per_batch,
                    n_clean // self.half_batch,
                )
                self.batch_count = 0
                self.epoch_count = 0
                
            def __call__(self, batch_indices):
                """Called by DataLoader.collate_fn."""
                # Check if we need to start a new epoch
                if self.batch_count >= self.max_batches:
                    # Start new epoch: shuffle if needed and reset counters
                    self.epoch_count += 1
                    self.batch_count = 0
                    
                    if self.do_shuffle:
                        # Shuffle each pool independently for this epoch
                        perm_signal = np.random.permutation(len(self.signal_dict["part_features"]))
                        perm_supp = np.random.permutation(len(self.supp_bg_dict["part_features"]))
                        perm_clean = np.random.permutation(len(self.clean_bg_dict["part_features"]))
                        
                        # Apply permutation to all keys in each dictionary
                        for key in self.signal_dict:
                            self.signal_dict[key] = self.signal_dict[key][perm_signal]
                        for key in self.supp_bg_dict:
                            self.supp_bg_dict[key] = self.supp_bg_dict[key][perm_supp]
                        for key in self.clean_bg_dict:
                            self.clean_bg_dict[key] = self.clean_bg_dict[key][perm_clean]
                
                # Extract slices for this batch
                b = self.batch_count
                signal_start, signal_end = b * self.k, (b + 1) * self.k
                supp_start, supp_end = b * self.supp_per_batch, (b + 1) * self.supp_per_batch
                clean_start, clean_end = b * self.half_batch, (b + 1) * self.half_batch
                
                # Decide whether to inject signal this batch
                inject_signal = np.random.random() < self.injection_probability
                
                # Sample k fill jets (with replacement): true signal or extra supp_bg
                if inject_signal:
                    fill_idx = np.random.choice(len(self.signal_dict["part_features"]), size=self.k, replace=True)
                    fill_dict = self.signal_dict
                else:
                    fill_idx = np.random.choice(len(self.supp_bg_dict["part_features"]), size=self.k, replace=True)
                    fill_dict = self.supp_bg_dict
                
                # Construct batch: [clean_bg (label 0), label-1 half]
                batch_data = {}
                for key in self.clean_bg_dict:
                    clean_part = self.clean_bg_dict[key][clean_start:clean_end]
                    supp_part = self.supp_bg_dict[key][supp_start:supp_end]
                    fill_part = fill_dict[key][fill_idx]
                    label1_part = torch.cat([fill_part, supp_part], dim=0)
                    batch_data[key] = torch.cat([clean_part, label1_part], dim=0)
                
                # Create labels: [0, 0, ..., 0 (half_batch), 1, 1, ..., 1 (half_batch)]
                batch_data["jet_type_labels"] = torch.cat([
                    torch.zeros(self.half_batch, dtype=torch.long),
                    torch.ones(self.half_batch, dtype=torch.long),
                ])
                
                # Increment counter for next batch
                self.batch_count += 1
                
                return batch_data
        
        collator = BatchCollator(signal_dict, supp_bg_dict, clean_bg_dict, 
                                batch_size, guaranteed_signal, shuffle,
                                injection_probability=injection_probability)
        
        # Create a dummy dataset that returns indices
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return idx
        
        dummy_dataset = DummyDataset(collator.max_batches)
        
        loader = DataLoader(
            dummy_dataset,
            batch_size=1,  # Process one batch at a time
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
        )
        
        return loader, collator.max_batches
    
    # Create train and validation loaders
    train_loader, n_train_batches = create_guaranteed_signal_loader(
        signal_train, supp_bg_train, clean_bg_train,
        batch_size, guaranteed_signal_per_batch, shuffle_train,
        injection_probability=injection_probability,
    )
    
    val_loader, n_val_batches = create_guaranteed_signal_loader(
        signal_val, supp_bg_val, clean_bg_val,
        batch_size, guaranteed_signal_per_batch, shuffle=False,
        injection_probability=injection_probability,
    )
    
    print(f"\nDataLoader created:")
    print(f"  Train batches per epoch: {n_train_batches}")
    print(f"  Val batches per epoch: {n_val_batches}")
    print(f"  Each batch: {batch_size} jets ({batch_size//2} label 0 + {batch_size//2} label 1)")
    print(f"  Label 1 composition: {guaranteed_signal_per_batch} true signal + {batch_size//2 - guaranteed_signal_per_batch} fake signal")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader

class ExperimentLogger:
    """Handles logging of experiment configuration and results."""
    
    def __init__(self, log_dir="logs", naming_identifier=""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = str(naming_identifier).strip()
        if identifier:
            self.run_name = f"run_{identifier}_{self.timestamp}"
        else:
            self.run_name = f"run_{self.timestamp}"

        # Create run-specific directory
        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config dictionary
        self.config = {}
        self.results = {}
        
    def log_config(self, config_dict):
        """Log experiment configuration."""
        self.config.update(config_dict)
        
        # Save config to JSON
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
        
    def log_results(self, results_dict):
        """Log experiment results."""
        self.results.update(results_dict)
        
        # Save results to JSON
        results_path = self.run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def log_final_results(self, trainer, checkpoint_callback):
        """Log final training results and metrics."""
        final_results = {
            "best_model_path": checkpoint_callback.best_model_path,
            "best_model_score": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "training_completed": True,
            "timestamp_end": datetime.now().isoformat(),
        }
        
        # Add callback metrics if available
        if hasattr(trainer, 'callback_metrics'):
            metrics = {k: float(v) if torch.is_tensor(v) else v 
                      for k, v in trainer.callback_metrics.items()}
            final_results["final_metrics"] = metrics
        
        self.log_results(final_results)
        
        # Create summary log file
        summary_path = self.run_dir / "summary.log"
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.run_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write("CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("RESULTS:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.results.items():
                f.write(f"{key}: {value}\n")
    
    def get_checkpoint_dir(self):
        """Get checkpoint directory for this run."""
        return str(self.run_dir / "checkpoints")


def create_model_config(pp_dict, args):
    """Create model configuration for BackboneTransformer.
    
    Parameters
    ----------
    pp_dict : dict
        Preprocessing dictionary
        
    Returns
    -------
    dict
        Model configuration
    """
    model_kwargs = {
        # Feature specification
        "particle_features_dict": pp_dict,
        
        # Architecture
        "embedding_dim": args.embedding_dim,
        "max_sequence_len": 128,
        "n_out_nodes": 2,  # Binary classification (signal vs background)
        
        "embed_cfg": OmegaConf.create({
            "type": "continuous_project_add",
            "intermediate_dim": None,
        }),
        
        # Transformer configuration (matching pre-trained checkpoint)
        "transformer_cfg": OmegaConf.create({
            "dim": args.embedding_dim,  # Must match embedding_dim
            "n_blocks": 8,
            "norm_after_blocks": True,
            "residual_cfg": {
                "gate_type": "local",
                "init_value": 1,
            },
            "attn_cfg": {
                "num_heads": 8,
                "dropout_rate": 0.1,
                "norm_before": True,
                "norm_after": False,
            },
            "mlp_cfg": {
                "dropout_rate": 0.0,
                "norm_before": True,
                "expansion_factor": 4,
                "activation": "GELU",
            },
        }),
        
        # # Classification head settings (for class_attention type)
        # "class_head_hidden_dim": 128,
        # "class_head_num_heads": 8,
        # "class_head_num_CA_blocks": 2,
        # "class_head_num_SA_blocks": 0,
        # "class_head_dropout_rate": 0.1,

        # Anomaly detection head settings (for Aachen method)
        "class_head_hidden_dim": 128,
        "class_head_num_heads": 2,
        "class_head_num_CA_blocks": 2,
        "class_head_num_SA_blocks": 0,
        "class_head_dropout_rate": 0.1,
        
        # Jet-level features
        "jet_features_input_dim": 0,
        
        # Other settings
        "apply_causal_mask": False,
        "zero_padded_start_particle": False,
    }
    
    return model_kwargs


class AUCCallback(Callback):
    """Compute AUC on the validation set at the end of each validation epoch
    and log it to the LightningModule so it becomes part of callback_metrics.
    Note: this will run the model over the validation loader again, so it
    duplicates work performed by the Lightning validation loop (but it ensures
    we have a reliable ROC AUC metric in trainer.callback_metrics and therefore
    in the experiment results.json).
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        # Try to get first validation dataloader
        try:
            val_loaders = trainer.val_dataloaders
        except Exception:
            val_loaders = None
        if not val_loaders:
            return

        # Handle both single DataLoader and list of DataLoaders
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders
            
        device = pl_module.device if hasattr(pl_module, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        all_preds = []
        all_labels = []

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                # expect dict-style batches as used in data_utils
                labels = batch["jet_type_labels"].to(device)
                
                # Check if model is dijet or single-jet
                if isinstance(pl_module, (BackboneDijetClassificationLightning, BackboneAachenClassificationLightning)):
                    # Dijet model: needs both jets
                    X1 = batch["part_features"].to(device)
                    X2 = batch["part_features_jet2"].to(device)
                    mask1 = batch["part_mask"].to(device)
                    mask2 = batch["part_mask_jet2"].to(device)
                    logits = pl_module(X1, mask1, X2, mask2)
                else:
                    # Single-jet model: needs only one jet
                    X = batch["part_features"].to(device)
                    mask = batch["part_mask"].to(device)
                    logits = pl_module(X, mask)
                
                # Handle different logit shapes
                if logits.dim() == 1:
                    # Binary classification with single logit (BCEWithLogitsLoss)
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    # Multi-class with softmax
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # If only one class present, roc_auc_score will fail - handle gracefully
        try:
            auc_val = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc_val = float('nan')

        # Log the metric so Lightning records it in callback_metrics
        pl_module.log("val_auc", auc_val, prog_bar=True, logger=True)

class ARGOSCallback(Callback):
    """Compute ARGOS metric on the validation set at the end of each validation epoch.
    ARGOS is defined as: max(tpr/sqrt(fpr) - sqrt(tpr)) for fpr > 0.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation dataloader
        try:
            val_loaders = trainer.val_dataloaders
        except Exception:
            val_loaders = None
        if not val_loaders:
            return

        # Handle both single DataLoader and list of DataLoaders
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders
            
        device = pl_module.device if hasattr(pl_module, 'device') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        all_preds = []
        all_labels = []

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["jet_type_labels"].to(device)
                
                # Check if model is dijet or single-jet
                if isinstance(pl_module, BackboneAachenClassificationLightning):
                    X1 = batch["part_features"].to(device)
                    X2 = batch["part_features_jet2"].to(device)
                    mask1 = batch["part_mask"].to(device)
                    mask2 = batch["part_mask_jet2"].to(device)
                    logits = pl_module(X1, mask1, X2, mask2)
                else:
                    X = batch["part_features"].to(device)
                    mask = batch["part_mask"].to(device)
                    logits = pl_module(X, mask)

                # Handle different logit shapes
                if logits.dim() == 1:
                    # Binary classification with single logit (BCEWithLogitsLoss)
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    # Multi-class with softmax
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # Compute ARGOS metric
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            inds = np.nonzero(fpr)
            tpr = tpr[inds]
            fpr = fpr[inds]
            argos = float(np.max(tpr/np.sqrt(fpr) - np.sqrt(tpr)))
        except Exception:
            argos = float('nan')

        # Log the metric
        pl_module.log("val_argos", argos, prog_bar=True, logger=True)

def load_pretrained_backbone(model, ckpt_path, strict=False):
    """Load pre-trained backbone weights from a checkpoint.
    
    This function loads backbone weights from a pre-trained checkpoint with
    flexible handling of dimension mismatches. Layers with compatible dimensions
    are loaded, while incompatible layers (e.g., input projection due to different
    feature counts) are initialized randomly.
    
    Parameters
    ----------
    model : BackboneClassificationLightning
        The model to load weights into
    ckpt_path : str
        Path to the checkpoint file
    strict : bool, optional
        Whether to strictly enforce that the keys in state_dict match (default: False)
        When False, allows partial loading with dimension mismatches
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load checkpoint
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    # Filter to only backbone weights
    backbone_state_dict = {}
    for key, value in state_dict.items():
        # Keep only backbone-related keys, skip head/classifier keys
        if key.startswith("backbone."):
            # Remove "backbone." prefix for loading into model.backbone
            new_key = key.replace("backbone.", "")
            backbone_state_dict[new_key] = value
        elif key.startswith("module."):
            # Handle case where weights might be saved with "module." prefix
            new_key = key.replace("module.", "")
            if not new_key.startswith("head"):  # Skip head weights
                backbone_state_dict[new_key] = value
    
    # Remove tril keys for backwards compatibility
    backbone_state_dict = {k: v for k, v in backbone_state_dict.items() if ".tril" not in k}
    
    # Get current model state dict
    current_state_dict = model.backbone.state_dict()
    
    # Filter out keys with dimension mismatches
    compatible_state_dict = {}
    incompatible_keys = []
    
    for key, value in backbone_state_dict.items():
        if key in current_state_dict:
            if current_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                incompatible_keys.append(
                    f"{key}: checkpoint shape {value.shape} vs model shape {current_state_dict[key].shape}"
                )
        else:
            # Key exists in checkpoint but not in current model
            incompatible_keys.append(f"{key}: not found in current model")
    
    print(f"\nLoading {len(compatible_state_dict)}/{len(backbone_state_dict)} compatible backbone parameters")
    print(f"Sample compatible keys: {list(compatible_state_dict.keys())[:5]}")
    
    if incompatible_keys:
        print(f"\n⚠️  Found {len(incompatible_keys)} incompatible parameters (will be randomly initialized):")
        for key in incompatible_keys[:10]:
            print(f"  - {key}")
        if len(incompatible_keys) > 10:
            print(f"  ... and {len(incompatible_keys) - 10} more")
    
    # Load the compatible weights
    missing_keys, unexpected_keys = model.backbone.load_state_dict(compatible_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n📝 Missing keys ({len(missing_keys)}) - these will remain randomly initialized:")
        for key in missing_keys[:10]:
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    print("\n✓ Backbone weights loaded successfully!")
    print("  - Transformer blocks: loaded from checkpoint")
    print("  - Input/output projections: may be randomly initialized due to feature dimension differences")
    print("  - Classification head: randomly initialized (2 classes for LHCO vs 10 classes in checkpoint)")

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
    parser.add_argument("--signal_file_name", type=str, default=str(os.getenv("SIGNAL_FILE_NAME")), help="Name of the signal H5 file in the dataset directory")
    parser.add_argument("--naming_identifier", type=str, default="", help="Optional identifier to add to the run name for easier tracking")
    parser.add_argument("--log_dir", type=str, default=str(os.getenv("LOG_DIR_AACHEN")), help="Directory for experiment logs")
    parser.add_argument("--pretrained_ckpt", type=str, help="Path to pre-trained checkpoint")
    parser.add_argument("--load_pretrained", action="store_true", help="Load pre-trained backbone weights from checkpoint")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == 'true', default=True, help="Use automatic class weighting for imbalanced data (default: True)")
    
    # Diagnostic ablation study: guaranteed signal injection
    parser.add_argument("--guaranteed_signal_per_batch", type=int, default=0, 
                       help="Number of guaranteed true signal jets per batch (k). Set to 0 to use standard weak supervision. Set to 1-4 for ablation studies.")
    parser.add_argument("--injection_probability", type=float, default=1.0,
                       help="Probability of injecting signal jets into a batch (default: 1.0). If < 1.0, some batches will have the signal slots replaced with extra suppressed background jets.")

    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="anchor-injection-lhco", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional, auto-generated if not provided)")
    
    args = parser.parse_args()

    # ============================================================
    # 0. Initialize Experiment Logger
    # ============================================================
    exp_logger = ExperimentLogger(log_dir=args.log_dir, naming_identifier=args.naming_identifier)
    print(f"Experiment: {exp_logger.run_name}")
    print(f"Log directory: {exp_logger.run_dir}")
    if args.guaranteed_signal_per_batch == 0:
        print("WARNING: Running with k=0 (no guaranteed signal injection).")
    
    # ============================================================
    # 1. Configuration
    # ============================================================
    # Set random seed for reproducibility
    L.seed_everything(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================
    # 2. Load Data
    # ============================================================

    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }

    signal_path = os.path.join(args.dataset_path, args.signal_file_name)
    supp_background_path = os.path.join(args.dataset_path, "bg_100k_SR_supp.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    
    h5_files_all = [signal_path, supp_background_path, background_path]
    print("Signal File Used:", signal_path)
    print("n_jets_train:", args.n_jets_train)
    print("Using Jet:", args.jet_name)

    # Log data configuration (dependent on k value)
    if args.guaranteed_signal_per_batch > 0:
        # For k>0: All three files used with guaranteed signal injection
        data_config = {
            "experiment_type": "guaranteed_signal_injection",
            "dataset_path": args.dataset_path,
            "signal_file": signal_path,
            "supp_background_file": supp_background_path,
            "background_file": background_path,
            "n_jets_train": args.n_jets_train,
            "batch_size": args.batch_size,
            "max_sequence_len": 128,
            "mom4_format": "epxpypz",
            "train_val_split": args.train_val_split,
            "features": list(input_features_dict.keys()),
            "feature_preprocessing": input_features_dict,
            "shuffle_train": True,
            "jet_name": args.jet_name,
            "guaranteed_signal_per_batch": args.guaranteed_signal_per_batch,
            "note": "All three files used. Signal injection is guaranteed in every batch.",
        }
    else:
        # For k=0: Standard weak supervision baseline with all three files
        # Label 0: Clean QCD (200k jets)
        # Label 1: Mixed (25k real signal + 100k mislabeled suppressed QCD = 125k total)
        # Signal fraction in Label 1: 25k / 125k ≈ 20% (not 1% due to file sizes)
        data_config = {
            "experiment_type": "weak_supervision_baseline",
            "dataset_path": args.dataset_path,
            "signal_file": signal_path,
            "supp_background_file": supp_background_path,
            "background_file": background_path,
            "n_jets_train": args.n_jets_train,
            "batch_size": args.batch_size,
            "max_sequence_len": 128,
            "mom4_format": "epxpypz",
            "train_val_split": args.train_val_split,
            "features": list(input_features_dict.keys()),
            "feature_preprocessing": input_features_dict,
            "shuffle_train": True,
            "jet_name": args.jet_name,
            "guaranteed_signal_per_batch": args.guaranteed_signal_per_batch,
            "note": "Standard weak supervision (all three files). Baseline for k=1-4 ablation.",
        }
    
    # Load data using appropriate loader
    if args.guaranteed_signal_per_batch > 0:
        # Use the diagnostic ablation study loader with guaranteed signal injection
        print(f"Guaranteed true signal per batch: {args.guaranteed_signal_per_batch}")
        print(f"Expected label 1 composition per batch:")
        print(f"  - True signal: {args.guaranteed_signal_per_batch} jets")
        print(f"  - Fake signal (supp BG): {args.batch_size // 2 - args.guaranteed_signal_per_batch} jets")
        print(f"This is designed to diagnose gradient flow issues at low signal injection levels.")
        print(f"{'='*80}\n")
        
        train_loader, val_loader = create_guaranteed_signal_dataloaders(
            signal_path=signal_path,
            supp_bg_path=supp_background_path,
            clean_bg_path=background_path,
            feature_dict=input_features_dict,
            batch_size=args.batch_size,
            guaranteed_signal_per_batch=args.guaranteed_signal_per_batch,
            n_signal=args.n_jets_train[0],
            n_supp_bg=args.n_jets_train[1],
            n_clean_bg=args.n_jets_train[2],
            max_sequence_len=128,
            mom4_format="epxpypz",
            jet_name=args.jet_name,
            train_val_split=args.train_val_split,
            shuffle_train=True,
            num_workers=1,
            injection_probability=args.injection_probability,
        )
    else:
        # k=0: Standard weak supervision baseline (all three files)
        # Label 0: Clean QCD background (bg_200k_SR_train.h5)
        # Label 1: Mixed weak labels (signal + suppressed QCD)
        # This is the baseline for comparison with k=1,2,3,4 guaranteed signal experiments
        print(f"\n{'='*80}")
        print(f"WEAK SUPERVISION BASELINE (k=0)")
        print(f"{'='*80}")
        print(f"Label 0: Clean QCD background ({args.n_jets_train[2]} jets)")
        print(f"Label 1: Mixed weak labels (")
        print(f"          {args.n_jets_train[0]} real signal jets +")
        print(f"          {args.n_jets_train[1]} mislabeled suppressed QCD)")
        print(f"Total Label 1: {args.n_jets_train[0] + args.n_jets_train[1]} jets")
        print(f"Signal fraction in Label 1: {args.n_jets_train[0] / (args.n_jets_train[0] + args.n_jets_train[1]):.1%}")
        print(f"This is the baseline for k=1,2,3,4 ablation studies.")
        print(f"{'='*80}\n")
        
        train_loader, val_loader = create_custom_lhco_h5_dataloaders(
            h5_files_train=h5_files_all,  # All three files: signal, supp_bg, clean_bg
            h5_files_val=None,
            feature_dict=input_features_dict,
            batch_size=args.batch_size,
            n_jets_train=args.n_jets_train,  # [signal, supp_bg, clean_bg]
            max_sequence_len=128,
            mom4_format="epxpypz",
            jet_name=args.jet_name,
            train_val_split=args.train_val_split,
            shuffle_train=True,
            num_workers=1,
        )

    # ============================================================
    # 3. Create Model
    # ============================================================

    # Calculate class weights for imbalanced dataset
    model_kwargs = create_model_config(input_features_dict, args)
    
    if args.use_class_weights:
        if args.guaranteed_signal_per_batch > 0:
            # For diagnostic ablation: class weights based on GUARANTEED composition
            # Each batch has exactly batch_size//2 label 0 and batch_size//2 label 1
            # This is perfectly balanced!
            print(f"\n=== Guaranteed Signal Injection - Class Weights ===")
            print(f"With guaranteed signal, each batch is perfectly balanced:")
            print(f"  Label 0: {args.batch_size // 2} jets (clean background)")
            print(f"  Label 1: {args.batch_size // 2} jets (true signal + fake signal)")
            print(f"  Ratio: 1:1 (perfectly balanced)")
            print(f"  Using equal class weights for balanced classification.\n")
            
            # For balanced data, equal weights work well
            class_weights = [1.0, 1.0]
            model_kwargs["class_weights"] = class_weights
        else:
            # For weak supervision: Calculate weights based on ACTUAL label distribution
            # n_jets_train = [signal_real, supp_bg_labeled_as_signal, background_real]
            # Actual label distribution after loading:
            #   - Label 1: signal_real + supp_bg_labeled_as_signal  
            #   - Label 0: background_real
            n_label_1 = args.n_jets_train[0] + args.n_jets_train[1]  # signal + supp background
            n_label_0 = args.n_jets_train[2]  # clean background
            total = n_label_1 + n_label_0
            
            # Weight = total / (n_classes * n_samples_per_class)
            # Higher weight for minority class
            weight_label_0 = total / (2.0 * n_label_0)  # Weight for class 0 (clean background)
            weight_label_1 = total / (2.0 * n_label_1)  # Weight for class 1 (signal + polluted)
            # PyTorch CrossEntropyLoss expects weights in class order: [weight_for_class_0, weight_for_class_1]
            class_weights = [weight_label_0, weight_label_1]  # CORRECT ORDER!
            
            print(f"\n=== Weak Supervision Label Distribution ===")
            print(f"Label 0 (clean background): {n_label_0} jets → weight={weight_label_0:.4f}")
            print(f"Label 1 (signal + polluted bg): {n_label_1} jets → weight={weight_label_1:.4f}")
            print(f"  - True signal: {args.n_jets_train[0]}")
            print(f"  - Polluted background: {args.n_jets_train[1]}")
            print(f"Weight ratio (Label_1/Label_0): {weight_label_1/weight_label_0:.4f}")
            print(f"Class weights array: {class_weights}\n")
            model_kwargs["class_weights"] = class_weights
    else:
        print("Class weighting disabled - using standard CrossEntropyLoss")
        model_kwargs["class_weights"] = None

    # For constant learning rate, use ConstantLR
    scheduler_with_params = torch.optim.lr_scheduler.ConstantLR
    
    # For cosine annealing, uncomment the following:
    # scheduler_with_params = partial(
    #     torch.optim.lr_scheduler.CosineAnnealingLR,
    #     T_max=args.max_steps,
    #     eta_min=1e-6, # minimum learning rate
    # )

    # -------------------------------------------------------------------------
    # ---------------------- Single Jet Data Model ----------------------------
    # -------------------------------------------------------------------------

    # Initialize the Backbone + Classification Head
    # model = BackboneClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": args.learning_rate,
    #         "weight_decay": 1e-2,
    #     },
    #     scheduler=scheduler_with_params,
    #     class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_argos",
    #         "mode": "max",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    # -------------------------------------------------------------------------
    # ------------------------- DiJet Data Model ------------------------------
    # -------------------------------------------------------------------------

    # model = BackboneDijetClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": args.learning_rate,
    #         "weight_decay": 1e-2,
    #     },
    #     scheduler=scheduler_with_params,
    #     merge_strategy=args.merge_strategy,  # other options: "average", "weighted_sum", "attention"
    #     class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_argos",
    #         "mode": "max",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    # -------------------------------------------------------------------------
    # ------------------ Aachen Anomaly Detection Model -----------------------
    # -------------------------------------------------------------------------

    model = BackboneAachenClassificationLightning(
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": args.learning_rate,
            "weight_decay": 1e-2,
        },
        scheduler=scheduler_with_params,
        merge_strategy=args.merge_strategy,  # options: "concat", "average", "weighted_sum", "attention"
        model_kwargs=model_kwargs,
        use_continuous_input=True,
        scheduler_lightning_kwargs={
            "monitor": "val_argos",
            "mode": "max",
            "interval": "step",
            "frequency": 1,
        },
    )

    
    # Load pre-trained backbone weights if requested
    if args.load_pretrained and args.pretrained_ckpt:
        print(f"Loading pre-trained backbone weights from: {args.pretrained_ckpt}")
        load_pretrained_backbone(model, args.pretrained_ckpt)
        print("Successfully loaded pre-trained backbone weights!")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Log model configuration
    model_config = {
        "architecture": "BackboneAachenClassificationLightning",
        "class_head_type": "aachen", #"class_attention",
        "merge_strategy": args.merge_strategy,
        "use_continuous_input": True,
        "num_parameters": num_params,
        "embedding_dim": args.embedding_dim,
        "n_transformer_blocks": 8,  # Updated to match pre-trained model
        "num_attention_heads": 8,
        "max_sequence_len": 128,
        "n_output_classes": 2,
        "pretrained_checkpoint": args.pretrained_ckpt if args.load_pretrained else None,
        "load_pretrained": args.load_pretrained,
        "model_kwargs": {k: v for k, v in model_kwargs.items() if k != "particle_features_dict"},
    }
    
    # Log training configuration
    training_config = {
        "optimizer": "AdamW",
        "optimizer_params": {
            "lr": args.learning_rate,
            "weight_decay": 1e-2,
        },
        "scheduler": "ConstantLR",
        "max_steps": args.max_steps,
        "gradient_clip_val": 1.0,
        "precision": "32",
        "early_stopping_patience": 15,
        "early_stopping_monitor": "val_argos",
        "checkpoint_monitor": "val_argos",
        "checkpoint_mode": "max",
        "load_pretrained": args.load_pretrained,
        "pretrained_ckpt": args.pretrained_ckpt,
        "use_class_weights": args.use_class_weights,
        "class_weights": model_kwargs.get("class_weights", None),
        "diagnostic_study": {
            "enabled": args.guaranteed_signal_per_batch > 0,
            "guaranteed_signal_per_batch": args.guaranteed_signal_per_batch,
            "purpose": "Validate model's ability to detect signal when guaranteed exposure to true signal",
        } if args.guaranteed_signal_per_batch > 0 else None,
    }
    
    # Log system configuration
    system_config = {
        "device": str(device),
        "gpu_id": args.gpu_id,
        "random_seed": args.seed,
        "timestamp_start": datetime.now().isoformat(),
    }
    
    # Combine all configs and log
    full_config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "system": system_config,
    }
    exp_logger.log_config(full_config)
    print(f"Configuration saved to: {exp_logger.run_dir / 'config.json'}")
    
    # # Setup callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=exp_logger.get_checkpoint_dir(),
    #     filename="epoch_{epoch:02d}_{val_loss:.4f}",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=3,
    #     save_last=False,
    # )

    # Alternatively, monitor AUC instead of loss
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=exp_logger.get_checkpoint_dir(),
    #     filename="epoch_{epoch:02d}_{val_loss:.4f}",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=3,
    #     save_last=False,
    # )

    # Or using ARGOS metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_logger.get_checkpoint_dir(),
        filename="{epoch:02d}_{val_argos:.4f}",
        monitor="val_argos",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    
    # Early stopping disabled
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=30,
    #     mode="min",
    # )

    # AUC callback: computes ROC AUC on validation set each epoch and logs it
    auc_callback = AUCCallback()

    # ARGOS callback: computes ARGOS metric on validation set each epoch and logs it
    argos_callback = ARGOSCallback()

    # Setup W&B logger
    loggers = []
    if args.use_wandb:
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_logger.run_name
        
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            save_dir=str(exp_logger.run_dir),
            config=full_config,
            log_model=True,
        )
        loggers.append(wandb_logger)
        
        print(f"\n{'=' * 80}")
        print(f"W&B logging enabled!")
        print(f"  Project: {args.wandb_project}")
        if args.wandb_entity:
            print(f"  Entity: {args.wandb_entity}")
        print(f"  Run name: {wandb_run_name}")
        print(f"  Run URL: {wandb_logger.experiment.url}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\nW&B logging disabled. Use --use_wandb to enable.\n")
        wandb_logger = None

    # Create trainer
    print("Starting training...")
    # Ensure Lightning uses the GPU requested via --gpu_id.
    # Use explicit accelerator and devices so PL selects the correct device
    # (accelerator="auto", devices=1 will pick the first visible GPU i.e. GPU 0).
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=[args.gpu_id],
        logger=loggers if loggers else False,
        callbacks=[checkpoint_callback, auc_callback, argos_callback], #early_stop_callback],
        log_every_n_steps=20,
        gradient_clip_val=1,
        precision="32",
        num_nodes=1,
    )

    # ============================================================
    # 4. Training Loop
    # ============================================================
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        # Log final results
        exp_logger.log_final_results(trainer, checkpoint_callback)
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        print(f"Results saved to: {exp_logger.run_dir}")
        if args.use_wandb:
            print(f"W&B run: {wandb.run.url}")
            wandb.finish()
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user!")
        print(f"Partial results saved to: {exp_logger.run_dir}")
        print("=" * 80 + "\n")
        
        if args.use_wandb:
            wandb.finish()
        
    except Exception as e:
        # Log error if training fails
        error_info = {
            "training_completed": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp_error": datetime.now().isoformat(),
        }
        exp_logger.log_results(error_info)
        print(f"\n{'=' * 80}")
        print(f"Training failed! Error logged to: {exp_logger.run_dir}")
        print(f"Error: {e}")
        print("=" * 80 + "\n")
        
        if args.use_wandb:
            wandb.finish(exit_code=1)
        
        raise

if __name__ == "__main__":
    main()
