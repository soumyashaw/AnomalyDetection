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
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")

# gabbro imports
from gabbro.utils.arrays import ak_pad
from gabbro.data.data_utils import create_custom_lhco_h5_dataloaders
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

load_dotenv()  # Load environment variables from .env file (for W&B API key, etc.)


# ============================================================================
# LATENT SPACE TRACKING FOR CATASTROPHIC FORGETTING DETECTION
# ============================================================================

def load_golden_batch(signal_path, supp_bg_path, feature_dict, batch_size=64, 
                      n_signal=32, n_background=32, max_sequence_len=128, 
                      mom4_format="epxpypz", jet_name="both"):
    """
    Load a "Golden Batch" with guaranteed 50/50 signal-background composition.
    
    Injected at step 100 to study catastrophic forgetting recovery.
    Data comes from TRAINING sets (not test), but separate from train/val split.
    """
    print(f"\n{'='*80}")
    print(f"Loading Golden Batch (50% signal / 50% background)")
    print(f"{'='*80}\n")
    
    feature_names = list(feature_dict.keys())
    
    # Load signal jets
    print(f"Loading {n_signal} signal jets from {signal_path}...")
    if jet_name == "both":
        signal_jet1, signal_jet2, _ = load_lhco_jets_from_h5(
            signal_path, feature_dict, n_jets=n_signal, jet_name="both", mom4_format=mom4_format
        )
        signal_jet1_padded, signal_mask1 = ak_pad(signal_jet1, maxlen=max_sequence_len, return_mask=True)
        signal_jet2_padded, signal_mask2 = ak_pad(signal_jet2, maxlen=max_sequence_len, return_mask=True)
        
        signal_jet1_stacked = ak.concatenate(
            [signal_jet1_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_jet2_stacked = ak.concatenate(
            [signal_jet2_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_data = {
            "part_features": torch.from_numpy(ak.to_numpy(signal_jet1_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(signal_mask1)).float(),
            "part_features_jet2": torch.from_numpy(ak.to_numpy(signal_jet2_stacked)).float(),
            "part_mask_jet2": torch.from_numpy(ak.to_numpy(signal_mask2)).float(),
        }
    else:
        signal_jets, _ = load_lhco_jets_from_h5(
            signal_path, feature_dict, n_jets=n_signal, jet_name=jet_name, mom4_format=mom4_format
        )
        signal_padded, signal_mask = ak_pad(signal_jets, maxlen=max_sequence_len, return_mask=True)
        signal_stacked = ak.concatenate(
            [signal_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_data = {
            "part_features": torch.from_numpy(ak.to_numpy(signal_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(signal_mask)).float(),
        }
    
    # Load background jets
    print(f"Loading {n_background} background jets from {supp_bg_path}...")
    if jet_name == "both":
        bg_jet1, bg_jet2, _ = load_lhco_jets_from_h5(
            supp_bg_path, feature_dict, n_jets=n_background, jet_name="both", mom4_format=mom4_format
        )
        bg_jet1_padded, bg_mask1 = ak_pad(bg_jet1, maxlen=max_sequence_len, return_mask=True)
        bg_jet2_padded, bg_mask2 = ak_pad(bg_jet2, maxlen=max_sequence_len, return_mask=True)
        
        bg_jet1_stacked = ak.concatenate(
            [bg_jet1_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_jet2_stacked = ak.concatenate(
            [bg_jet2_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_data = {
            "part_features": torch.from_numpy(ak.to_numpy(bg_jet1_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(bg_mask1)).float(),
            "part_features_jet2": torch.from_numpy(ak.to_numpy(bg_jet2_stacked)).float(),
            "part_mask_jet2": torch.from_numpy(ak.to_numpy(bg_mask2)).float(),
        }
    else:
        bg_jets, _ = load_lhco_jets_from_h5(
            supp_bg_path, feature_dict, n_jets=n_background, jet_name=jet_name, mom4_format=mom4_format
        )
        bg_padded, bg_mask = ak_pad(bg_jets, maxlen=max_sequence_len, return_mask=True)
        bg_stacked = ak.concatenate(
            [bg_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_data = {
            "part_features": torch.from_numpy(ak.to_numpy(bg_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(bg_mask)).float(),
        }
    
    # Combine into single batch
    golden_batch = {}
    for key in signal_data:
        golden_batch[key] = torch.cat([signal_data[key], bg_data[key]], dim=0)
    
    # Labels: [1, 1, ..., 1 (n_signal), 0, 0, ..., 0 (n_background)]
    golden_batch["jet_type_labels"] = torch.cat([
        torch.ones(n_signal, dtype=torch.long),
        torch.zeros(n_background, dtype=torch.long),
    ])
    
    print(f"Golden batch created: {n_signal} signal + {n_background} background = {batch_size} total\n")
    return golden_batch


def load_tracking_set(signal_test_path, bg_test_path, feature_dict, 
                      n_signal=100, n_background=100, max_sequence_len=128,
                      mom4_format="epxpypz", jet_name="both"):
    """
    Load tracking set from TEST data (independent of train/val/golden_batch).
    
    Used to monitor latent space and catastrophic forgetting recovery.
    """
    print(f"\n{'='*80}")
    print(f"Loading Tracking Set from TEST data (independent from training)")
    print(f"{'='*80}\n")
    
    feature_names = list(feature_dict.keys())
    
    # Load signal jets from TEST
    print(f"Loading {n_signal} signal jets from {signal_test_path}...")
    if jet_name == "both":
        signal_jet1, signal_jet2, _ = load_lhco_jets_from_h5(
            signal_test_path, feature_dict, n_jets=n_signal, jet_name="both", mom4_format=mom4_format
        )
        signal_jet1_padded, signal_mask1 = ak_pad(signal_jet1, maxlen=max_sequence_len, return_mask=True)
        signal_jet2_padded, signal_mask2 = ak_pad(signal_jet2, maxlen=max_sequence_len, return_mask=True)
        
        signal_jet1_stacked = ak.concatenate(
            [signal_jet1_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_jet2_stacked = ak.concatenate(
            [signal_jet2_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_data = {
            "part_features": torch.from_numpy(ak.to_numpy(signal_jet1_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(signal_mask1)).float(),
            "part_features_jet2": torch.from_numpy(ak.to_numpy(signal_jet2_stacked)).float(),
            "part_mask_jet2": torch.from_numpy(ak.to_numpy(signal_mask2)).float(),
        }
    else:
        signal_jets, _ = load_lhco_jets_from_h5(
            signal_test_path, feature_dict, n_jets=n_signal, jet_name=jet_name, mom4_format=mom4_format
        )
        signal_padded, signal_mask = ak_pad(signal_jets, maxlen=max_sequence_len, return_mask=True)
        signal_stacked = ak.concatenate(
            [signal_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        signal_data = {
            "part_features": torch.from_numpy(ak.to_numpy(signal_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(signal_mask)).float(),
        }
    
    # Load background jets from TEST
    print(f"Loading {n_background} background jets from {bg_test_path}...")
    if jet_name == "both":
        bg_jet1, bg_jet2, _ = load_lhco_jets_from_h5(
            bg_test_path, feature_dict, n_jets=n_background, jet_name="both", mom4_format=mom4_format
        )
        bg_jet1_padded, bg_mask1 = ak_pad(bg_jet1, maxlen=max_sequence_len, return_mask=True)
        bg_jet2_padded, bg_mask2 = ak_pad(bg_jet2, maxlen=max_sequence_len, return_mask=True)
        
        bg_jet1_stacked = ak.concatenate(
            [bg_jet1_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_jet2_stacked = ak.concatenate(
            [bg_jet2_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_data = {
            "part_features": torch.from_numpy(ak.to_numpy(bg_jet1_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(bg_mask1)).float(),
            "part_features_jet2": torch.from_numpy(ak.to_numpy(bg_jet2_stacked)).float(),
            "part_mask_jet2": torch.from_numpy(ak.to_numpy(bg_mask2)).float(),
        }
    else:
        bg_jets, _ = load_lhco_jets_from_h5(
            bg_test_path, feature_dict, n_jets=n_background, jet_name=jet_name, mom4_format=mom4_format
        )
        bg_padded, bg_mask = ak_pad(bg_jets, maxlen=max_sequence_len, return_mask=True)
        bg_stacked = ak.concatenate(
            [bg_padded[feat][..., np.newaxis] for feat in feature_names], axis=-1
        )
        bg_data = {
            "part_features": torch.from_numpy(ak.to_numpy(bg_stacked)).float(),
            "part_mask": torch.from_numpy(ak.to_numpy(bg_mask)).float(),
        }
    
    # Combine into tracking set
    tracking_set = {}
    for key in signal_data:
        tracking_set[key] = torch.cat([signal_data[key], bg_data[key]], dim=0)
    
    # True labels (ground truth)
    tracking_set["true_labels"] = torch.cat([
        torch.ones(n_signal, dtype=torch.long),
        torch.zeros(n_background, dtype=torch.long),
    ])
    
    print(f"Tracking set created: {n_signal} signal + {n_background} background = {n_signal + n_background} total")
    print(f"⚠️  Data from TEST set (completely independent)\n")
    
    return tracking_set


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


class LatentSpaceCallback(Callback):
    """Track latent space representations and visualize catastrophic forgetting."""
    
    def __init__(self, tracking_set, device, jet_name="both", log_dir="logs", 
                 tracking_steps=None, embedding_dim=128):
        """
        Parameters
        ----------
        tracking_set : dict
            Fixed set of signal/background jets for latent tracking
        device : torch.device
            Device to run on
        jet_name : str
            "jet1", "jet2", or "both"
        log_dir : str
            Directory to save visualizations
        tracking_steps : list
            Steps at which to track (e.g., [99, 100, 101, 150, 199])
        embedding_dim : int
            Backbone embedding dimension
        """
        super().__init__()
        self.tracking_set = tracking_set
        self.device = device
        self.jet_name = jet_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.track_steps = tracking_steps if tracking_steps else [1, 99, 100, 101, 150, 199]
        self.true_labels = tracking_set["true_labels"].numpy()
        
        # Store embeddings and metrics
        self.embeddings_history = {}
        self.metrics_history = {"silhouette": {}, "davies_bouldin": {}}
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after each training batch."""
        current_step = trainer.global_step
        
        if current_step not in self.track_steps:
            return
        
        print(f"\n{'='*80}")
        print(f"LATENT SPACE TRACKING at Step {current_step}")
        print(f"{'='*80}")
        
        embeddings = self._extract_embeddings(pl_module)
        if embeddings is None:
            print("⚠️ Failed to extract embeddings")
            return
        
        self.embeddings_history[current_step] = embeddings
        
        # Compute clustering metrics
        sil_score = silhouette_score(embeddings, self.true_labels)
        db_score = davies_bouldin_score(embeddings, self.true_labels)
        
        self.metrics_history["silhouette"][current_step] = sil_score
        self.metrics_history["davies_bouldin"][current_step] = db_score
        
        print(f"Signal-Background Separation Metrics:")
        print(f"  Silhouette Score: {sil_score:.4f} (↑ better)")
        print(f"  Davies-Bouldin Index: {db_score:.4f} (↓ better)")
        print(f"{'='*80}\n")
        
        # Visualize
        self._visualize_embeddings(current_step, embeddings)
    
    def _extract_embeddings(self, model):
        """Extract latent embeddings from backbone."""
        model.eval()
        embeddings = []
        
        def hook_fn(module, input, output):
            embeddings.append(output.detach().cpu().numpy())
        
        if hasattr(model, 'backbone'):
            hook = model.backbone.register_forward_hook(hook_fn)
        else:
            return None
        
        try:
            with torch.no_grad():
                batch_size = 64
                for i in range(0, len(self.tracking_set["part_features"]), batch_size):
                    batch_end = min(i + batch_size, len(self.tracking_set["part_features"]))
                    batch = {
                        k: v[i:batch_end].to(self.device) 
                        for k, v in self.tracking_set.items() 
                        if torch.is_tensor(v) and k != "true_labels"
                    }
                    
                    if self.jet_name == "both":
                        X1 = batch["part_features"]
                        mask1 = batch["part_mask"]
                        X2 = batch["part_features_jet2"]
                        mask2 = batch["part_mask_jet2"]
                        _ = model(X1, mask1, X2, mask2)
                    else:
                        X = batch["part_features"]
                        mask = batch["part_mask"]
                        _ = model(X, mask)
        finally:
            hook.remove()
        
        if not embeddings:
            return None
        
        return np.concatenate(embeddings, axis=0)
    
    def _visualize_embeddings(self, step, embeddings):
        """Create UMAP and t-SNE visualizations."""
        output_dir = self.log_dir / "catastrophic_forgetting_tracking"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reduce dimensionality if needed
        if embeddings.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # UMAP
        if HAS_UMAP:
            print(f"Computing UMAP...")
            umap_proj = umap.UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(embeddings_reduced)
            self._save_visualization(umap_proj, step, "umap", output_dir)
        
        # t-SNE
        print(f"Computing t-SNE...")
        tsne_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings_reduced)
        self._save_visualization(tsne_proj, step, "tsne", output_dir)
    
    def _save_visualization(self, proj_2d, step, method, output_dir):
        """Save 2D visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_signal = len(self.true_labels) // 2
        bg_indices = np.arange(n_signal, len(self.true_labels))
        signal_indices = np.arange(n_signal)
        
        ax.scatter(proj_2d[bg_indices, 0], proj_2d[bg_indices, 1], 
                  c='blue', label='Background', alpha=0.6, s=50)
        ax.scatter(proj_2d[signal_indices, 0], proj_2d[signal_indices, 1], 
                  c='red', label='Signal', alpha=0.6, s=50)
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Latent Space at Step {step} ({method.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f"latent_space_step{step:05d}_{method}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def on_train_end(self, trainer, pl_module):
        """Save summary metrics when training ends."""
        metrics_path = self.log_dir / "catastrophic_forgetting_tracking" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "silhouette": {str(k): v for k, v in self.metrics_history["silhouette"].items()},
            "davies_bouldin": {str(k): v for k, v in self.metrics_history["davies_bouldin"].items()},
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nLatent space tracking complete! Saved to {metrics_path}")


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
    
    # Catastrophic Forgetting Study arguments
    parser.add_argument("--study_catastrophic_forgetting", action="store_true", 
                       help="Enable catastrophic forgetting study with golden batch injection")
    parser.add_argument("--golden_injection_step", type=int, default=100,
                       help="Step at which to inject golden batch (default: 100)")
    parser.add_argument("--signal_test_path", type=str, default=None,
                       help="Path to test signal H5 file (e.g., sn_50k_SR_test.h5) for tracking set")
    parser.add_argument("--bg_test_path", type=str, default=None,
                       help="Path to test background H5 file (e.g., bg_200k_SR_test.h5) for tracking set")
    parser.add_argument("--tracking_set_size", type=int, default=100,
                       help="Number of signal and background jets in tracking set (default: 100 each)")
    
    args = parser.parse_args()

    # ============================================================
    # 0. Initialize Experiment Logger
    # ============================================================
    exp_logger = ExperimentLogger(log_dir=args.log_dir, naming_identifier=args.naming_identifier)
    print(f"Experiment: {exp_logger.run_name}")
    print(f"Log directory: {exp_logger.run_dir}")
    
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

    signal_path = os.path.join(args.dataset_path, "sn_25k_SR_train.h5")
    supp_background_path = os.path.join(args.dataset_path, "bg_100k_SR_supp.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    
    h5_files_all = [signal_path, supp_background_path, background_path]
    print("n_jets_train:", args.n_jets_train)
    print("Using Jet:", args.jet_name)


    # Log data configuration
    data_config = {
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
    }
    
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

    # Load golden batch and tracking set for catastrophic forgetting study
    golden_batch = None
    tracking_set = None
    if args.study_catastrophic_forgetting:
        golden_batch = load_golden_batch(
            signal_path=signal_path,
            supp_bg_path=supp_background_path,
            feature_dict=input_features_dict,
            batch_size=args.batch_size,
            n_signal=args.batch_size // 2,
            n_background=args.batch_size // 2,
            max_sequence_len=128,
            mom4_format="epxpypz",
            jet_name=args.jet_name,
        )
        
        # Validate test paths
        if args.signal_test_path is None or args.bg_test_path is None:
            raise ValueError("For catastrophic forgetting study, must provide --signal_test_path and --bg_test_path")
        
        tracking_set = load_tracking_set(
            signal_test_path=args.signal_test_path,
            bg_test_path=args.bg_test_path,
            feature_dict=input_features_dict,
            n_signal=args.tracking_set_size,
            n_background=args.tracking_set_size,
            max_sequence_len=128,
            mom4_format="epxpypz",
            jet_name=args.jet_name,
        )
        
        # Update data config with study info
        data_config["catastrophic_forgetting_study"] = {
            "enabled": True,
            "golden_batch_injection_step": args.golden_injection_step,
            "tracking_set_size": args.tracking_set_size + args.tracking_set_size,
        }

    # ============================================================
    # 3. Create Model
    # ============================================================

    # Calculate class weights for imbalanced dataset
    model_kwargs = create_model_config(input_features_dict, args)
    
    if args.use_class_weights:
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
    
    # Early stopping enabled
    early_stop_callback = EarlyStopping(
        monitor="val_argos",
        patience=30,
        mode="max",
    )

    # AUC callback: computes ROC AUC on validation set each epoch and logs it
    auc_callback = AUCCallback()

    # ARGOS callback: computes ARGOS metric on validation set each epoch and logs it
    argos_callback = ARGOSCallback()

    # Latent space tracking callback for catastrophic forgetting study
    latent_space_callback = None
    if args.study_catastrophic_forgetting and tracking_set is not None:
        print(f"\n{'='*80}")
        print(f"Catastrophic Forgetting Study ENABLED")
        print(f"  Golden batch injection at step: {args.golden_injection_step}")
        print(f"  Tracking set size: {2 * args.tracking_set_size} jets (signal + background)")
        print(f"{'='*80}\n")
        
        latent_space_callback = LatentSpaceCallback(
            tracking_set=tracking_set,
            device=device,
            jet_name=args.jet_name,
            log_dir=str(exp_logger.run_dir),
            tracking_steps=[1, args.golden_injection_step - 1, args.golden_injection_step, 
                           args.golden_injection_step + 1, args.max_steps // 2, args.max_steps],
            embedding_dim=args.embedding_dim,
        )

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
    
    # Prepare callbacks list
    callbacks_list = [checkpoint_callback, auc_callback, argos_callback, early_stop_callback]
    if latent_space_callback is not None:
        callbacks_list.append(latent_space_callback)
    
    # Ensure Lightning uses the GPU requested via --gpu_id.
    # Use explicit accelerator and devices so PL selects the correct device
    # (accelerator="auto", devices=1 will pick the first visible GPU i.e. GPU 0).
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=[args.gpu_id],
        logger=loggers if loggers else False,
        callbacks=callbacks_list,
        log_every_n_steps=20,
        gradient_clip_val=1,
        precision="32",
        num_nodes=1,
    )

    # ============================================================
    # 4. Training Loop with Optional Golden Batch Injection
    # ============================================================
    
    # Create wrapper to inject golden batch at specific step
    original_train_loader = train_loader
    golden_injected = False
    
    class GoldenBatchInjector:
        """Wrapper around train_loader to inject golden batch at step 100."""
        def __init__(self, loader, golden_batch, injection_step):
            self.loader = loader
            self.golden_batch = golden_batch
            self.injection_step = injection_step
            self.current_step = 0
            self.batch_iter = iter(loader)
        
        def __iter__(self):
            self.batch_iter = iter(self.loader)
            return self
        
        def __next__(self):
            # Inject golden batch at specific step
            if args.study_catastrophic_forgetting and self.current_step == self.injection_step:
                print(f"\n{'='*80}")
                print(f"🟡 INJECTING GOLDEN BATCH at Step {self.current_step}")
                print(f"{'='*80}\n")
                self.current_step += 1
                # Convert golden batch to same format as train batches
                return self.golden_batch
            
            batch = next(self.batch_iter)
            self.current_step += 1
            return batch
    
    if args.study_catastrophic_forgetting and golden_batch is not None:
        train_loader = GoldenBatchInjector(original_train_loader, golden_batch, args.golden_injection_step)
    
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
