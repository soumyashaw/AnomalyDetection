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
import umap
import json
import torch
import argparse
import numpy as np
import awkward as ak
import lightning as L
import awkward as ak
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import wandb
import matplotlib.pyplot as plt

# gabbro imports
from gabbro.utils.arrays import ak_pad
from gabbro.data.data_utils import create_custom_lhco_h5_dataloaders, create_lhco_h5_dataloaders, create_lhco_h5_test_loader
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

load_dotenv()  # Load environment variables from .env file (for W&B API key, etc.)

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


class EmbeddingVisualizationCallback(Callback):
    """Extract embeddings and generate t-SNE/UMAP plots after each training batch.
    
    Saves visualizations to plots/catastrophic_forgetting/{batch_idx}/ for tracking
    how embeddings evolve during training.
    """

    def __init__(self, test_loader, output_base_dir="plots/catastrophic_forgetting", log_frequency=1):
        """Initialize callback.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test dataloader to extract embeddings from
        output_base_dir : str
            Base directory for saving visualizations
        log_frequency : int
            Log visualization every N batches (default: 1, every batch)
        """
        self.test_loader = test_loader
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.batch_count = 0

    def extract_embeddings(self, pl_module, device):
        """Extract embeddings from test loader."""
        all_embeddings = []
        all_labels = []

        pl_module.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                labels = batch["jet_type_labels"]

                if isinstance(pl_module, (BackboneDijetClassificationLightning, BackboneAachenClassificationLightning)):
                    X1 = batch["part_features"].to(device)
                    X2 = batch["part_features_jet2"].to(device)
                    mask1 = batch["part_mask"].to(device)
                    mask2 = batch["part_mask_jet2"].to(device)

                    emb1 = pl_module.backbone(X1, mask1)
                    emb2 = pl_module.backbone(X2, mask2)

                    mask1_bool = mask1.bool()
                    mask2_bool = mask2.bool()

                    emb1_masked = emb1 * mask1_bool.unsqueeze(-1)
                    emb1_sum = emb1_masked.sum(dim=1)
                    valid_count1 = mask1_bool.sum(dim=1, keepdim=True).clamp(min=1)
                    emb1_pooled = emb1_sum / valid_count1

                    emb2_masked = emb2 * mask2_bool.unsqueeze(-1)
                    emb2_sum = emb2_masked.sum(dim=1)
                    valid_count2 = mask2_bool.sum(dim=1, keepdim=True).clamp(min=1)
                    emb2_pooled = emb2_sum / valid_count2

                    embeddings = torch.cat([emb1_pooled, emb2_pooled], dim=1)
                else:
                    X = batch["part_features"].to(device)
                    mask = batch["part_mask"].to(device)

                    emb = pl_module.backbone(X, mask)
                    mask_bool = mask.bool()
                    emb_masked = emb * mask_bool.unsqueeze(-1)
                    emb_sum = emb_masked.sum(dim=1)
                    valid_count = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
                    embeddings = emb_sum / valid_count

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        return embeddings_np, labels_np

    def plot_tsne(self, embeddings, labels, save_dir):
        """Generate t-SNE plot."""
        try:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                       random_state=42, n_jobs=-1, verbose=0)
            embeddings_tsne = tsne.fit_transform(embeddings_scaled)

            plt.figure(figsize=(10, 8))
            bg_mask = labels == 0
            sig_mask = labels == 1

            plt.scatter(embeddings_tsne[bg_mask, 0], embeddings_tsne[bg_mask, 1],
                       c='#FA8072', label='Background', alpha=0.6, s=5)
            plt.scatter(embeddings_tsne[sig_mask, 0], embeddings_tsne[sig_mask, 1],
                       c='#4FFFB0', label='Signal', alpha=0.6, s=5)

            plt.xlabel('t-SNE 1', fontsize=12)
            plt.ylabel('t-SNE 2', fontsize=12)
            plt.title(f'Jet Embeddings (t-SNE)', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = save_dir / 'embeddings_tsne.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating t-SNE plot: {e}")

    def plot_umap(self, embeddings, labels, save_dir):
        """Generate UMAP plot."""
        try:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            reducer = umap.UMAP(n_components=2, n_neighbors=15,
                               min_dist=0.1, random_state=42, n_jobs=1)
            embeddings_umap = reducer.fit_transform(embeddings_scaled)

            plt.figure(figsize=(10, 8))
            bg_mask = labels == 0
            sig_mask = labels == 1

            plt.scatter(embeddings_umap[bg_mask, 0], embeddings_umap[bg_mask, 1],
                       c='#FA8072', label='Background', alpha=0.6, s=5)
            plt.scatter(embeddings_umap[sig_mask, 0], embeddings_umap[sig_mask, 1],
                       c='#4FFFB0', label='Signal', alpha=0.6, s=5)

            plt.xlabel('UMAP 1', fontsize=12)
            plt.ylabel('UMAP 2', fontsize=12)
            plt.title(f'Jet Embeddings (UMAP)', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = save_dir / 'embeddings_umap.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating UMAP plot: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Visualize embeddings at specified frequency."""
        self.batch_count += 1

        if self.batch_count % self.log_frequency != 0:
            return

        if self.test_loader is None:
            return

        device = pl_module.device if hasattr(pl_module, 'device') else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        # Create batch-specific directory
        batch_dir = self.output_base_dir / f"batch_{self.batch_count:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Batch {self.batch_count}] Extracting embeddings and generating visualizations...")

        # Extract embeddings
        embeddings, labels = self.extract_embeddings(pl_module, device)

        # Generate plots
        self.plot_tsne(embeddings, labels, batch_dir)
        self.plot_umap(embeddings, labels, batch_dir)

        print(f"  Plots saved to: {batch_dir}")

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


class TestROCCallback(Callback):
    """Compute ROC AUC on the test set after each training batch and log it.

    Note: this is computationally expensive because it runs inference over the
    entire test loader after every train batch.
    """

    def __init__(self, test_loader):
        self.test_loader = test_loader

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.test_loader is None:
            return

        device = pl_module.device if hasattr(pl_module, 'device') else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        all_preds = []
        all_labels = []
        was_training = pl_module.training

        pl_module.eval()
        with torch.no_grad():
            for test_batch in self.test_loader:
                labels = test_batch["jet_type_labels"].to(device)

                if isinstance(pl_module, (BackboneDijetClassificationLightning, BackboneAachenClassificationLightning)):
                    X1 = test_batch["part_features"].to(device)
                    X2 = test_batch["part_features_jet2"].to(device)
                    mask1 = test_batch["part_mask"].to(device)
                    mask2 = test_batch["part_mask_jet2"].to(device)
                    logits = pl_module(X1, mask1, X2, mask2)
                else:
                    X = test_batch["part_features"].to(device)
                    mask = test_batch["part_mask"].to(device)
                    logits = pl_module(X, mask)

                if logits.dim() == 1:
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        if was_training:
            pl_module.train()

        if len(all_preds) == 0:
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        try:
            test_roc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            test_roc = float('nan')

        pl_module.log("test_roc", test_roc, on_step=True, on_epoch=False, logger=True, prog_bar=False)


class SignalInjectionCallback(Callback):
    """Inject real signal jets into training data every N batches.
    
    This callback implements the catastrophic forgetting experiment by:
    1. Starting training with minimal signal (n_jets_train_artificial=[1, ...])
    2. After every inject_freq batches, injecting real signal jets (--num_signal_jets)
       loaded via load_injectable_signal_jets()
    3. Replacing Label 1 (polluted background) samples with injected signal samples
    
    This tests if the model can recognize real signal despite initial weak supervision.
    """
    
    def __init__(self, signal_path, injection_signal_jets, num_signal_jets, input_features_dict, inject_freq=10):
        """Initialize the signal injection callback.
        
        Parameters
        ----------
        signal_path : str
            Path to the signal H5 file
        num_signal_jets : int
            Number of signal jets to inject per injection event
        input_features_dict : dict
            Feature preprocessing dictionary
        inject_freq : int
            Inject signal every N batches
        """
        self.signal_path = signal_path
        self.injection_signal_jets = injection_signal_jets
        self.num_signal_jets = num_signal_jets
        self.input_features_dict = input_features_dict
        self.inject_freq = inject_freq
        self.batch_count = 0
        self.injectable_signal_jets = None
        self.signal_index = 0
        
    def on_train_start(self, trainer, pl_module):
        """Load injectable signal jets at the start of training."""
        print(f"\n[SignalInjectionCallback] Loading injectable signal jets from {self.signal_path}...")
        try:
            self.injectable_signal_jets = load_injectable_signal_jets(
                self.signal_path, 
                self.injection_signal_jets,
                self.input_features_dict
            )
            print(f"[SignalInjectionCallback] Loaded signal jets for injection.")
        except Exception as e:
            print(f"[SignalInjectionCallback] Warning: Failed to load injectable signal jets: {e}")
            self.injectable_signal_jets = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Handle signal injection every inject_freq batches.
        
        This method is called after each training batch. Every inject_freq batches,
        it replaces some Label 1 (polluted background) samples with real signal jets.
        """
        self.batch_count += 1
        
        if self.batch_count % self.inject_freq == 0 and self.injectable_signal_jets is not None:
            try:
                print(f"\n[Batch {self.batch_count}] Injecting {self.num_signal_jets} real signal jets into Label 1...")
                
                # Note: on_train_batch_end is called AFTER the backward pass, so modifications
                # here affect the next training iteration. For immediate effect, this callback
                # should be paired with a custom training loop or DataLoader wrapper.
                
                # Extract batch components
                if isinstance(batch, dict):
                    part_features = batch.get("part_features")
                    part_mask = batch.get("part_mask")
                    jet_type_labels = batch.get("jet_type_labels")
                else:
                    # Try to extract from batch - might be a namedtuple or custom object
                    return
                
                if part_features is None or part_mask is None or jet_type_labels is None:
                    print(f"[Batch {self.batch_count}] Could not extract batch components")
                    return
                
                # Convert to numpy if on GPU/GPU tensors
                if hasattr(part_features, 'cpu'):
                    part_features = part_features.cpu().numpy()
                if hasattr(part_mask, 'cpu'):
                    part_mask = part_mask.cpu().numpy()
                if hasattr(jet_type_labels, 'cpu'):
                    jet_type_labels = jet_type_labels.cpu().numpy()
                
                batch_size = len(jet_type_labels)
                
                # Find indices of Label 1 samples (polluted background) in the batch
                label_1_indices = np.where(jet_type_labels == 1)[0]
                
                if len(label_1_indices) == 0:
                    print(f"[Batch {self.batch_count}] No Label 1 samples in batch to replace")
                    return
                
                # Determine number of samples to inject (min of requested and available)
                n_to_inject = min(self.num_signal_jets, len(label_1_indices))
                n_available_signal = self.injectable_signal_jets['n_jets_loaded']
                
                # Randomly select indices to replace from Label 1 samples
                np.random.seed(self.batch_count)  # For reproducibility
                replace_indices = np.random.choice(label_1_indices, size=n_to_inject, replace=False)
                
                # Randomly select signal jets to inject
                signal_indices = np.random.choice(n_available_signal, size=n_to_inject, replace=True)
                
                # Get injectable signal data
                injectable_features = self.injectable_signal_jets['part_features']  # (n_jets, max_particles, n_features)
                injectable_mask = self.injectable_signal_jets['part_mask']  # (n_jets, max_particles)
                
                # Check shape compatibility
                batch_max_particles = part_features.shape[1]
                injectable_max_particles = injectable_features.shape[1]
                
                # Perform injection: replace selected batch samples with signal jets
                for batch_idx_to_replace, signal_idx in zip(replace_indices, signal_indices):
                    # Handle particle dimension mismatch
                    if injectable_max_particles >= batch_max_particles:
                        # Trim injectable jet to match batch size
                        part_features[batch_idx_to_replace, :, :] = injectable_features[
                            signal_idx, :batch_max_particles, :
                        ]
                        part_mask[batch_idx_to_replace, :] = injectable_mask[
                            signal_idx, :batch_max_particles
                        ]
                    else:
                        # Pad injectable jet to match batch size
                        part_features[batch_idx_to_replace, :injectable_max_particles, :] = injectable_features[
                            signal_idx, :, :
                        ]
                        part_features[batch_idx_to_replace, injectable_max_particles:, :] = 0.0
                        part_mask[batch_idx_to_replace, :injectable_max_particles] = injectable_mask[
                            signal_idx, :
                        ]
                        part_mask[batch_idx_to_replace, injectable_max_particles:] = 0
                    
                    # Keep label as 1 (signal jets are labeled as 1)
                    jet_type_labels[batch_idx_to_replace] = 1
                
                # Update batch with modified arrays (convert back to tensor if needed)
                batch["part_features"] = part_features
                batch["part_mask"] = part_mask
                batch["jet_type_labels"] = jet_type_labels
                
                self.signal_index = (self.signal_index + n_to_inject) % n_available_signal
                
                print(f"[Batch {self.batch_count}] ✓ Injected {n_to_inject} real signal jets")
                print(f"[Batch {self.batch_count}] Replaced indices: {replace_indices}")
                print(f"[Batch {self.batch_count}] Signal indices used: {signal_indices}")
                
            except Exception as e:
                print(f"[Batch {self.batch_count}] Error during signal injection: {e}")
                import traceback
                traceback.print_exc()


def load_injectable_signal_jets(signal_path, num_signal_jets, input_features_dict):
    """Load a small number of signal jets for injection into the training data.
    
    This function loads signal jets from an H5 file and prepares them for injection
    into the training batches during the catastrophic forgetting experiment.
    
    Parameters
    ----------
    signal_path : str
        Path to the signal H5 file (e.g., "sn_25k_SR_train.h5")
    num_signal_jets : int
        Number of signal jets to load for injection. If the file has fewer signal
        jets than requested, all available signal jets are returned.
    input_features_dict : dict
        Feature preprocessing dictionary (e.g., {"part_pt": {...}, "part_etarel": {...}, ...})
    
    Returns
    -------
    dict or None
        Dictionary containing preprocessed signal jets with the following keys:
        - 'part_features': np.ndarray of shape (n_signal_jets, max_n_particles, n_features)
          Preprocessed particle-level features (pt, etarel, phirel, etc.)
        - 'part_mask': np.ndarray of shape (n_signal_jets, max_n_particles)
          Binary mask indicating valid particles (1) vs padding (0)
        - 'labels': np.ndarray of shape (n_signal_jets,)
          Array of ones (all are signal jets) for verification
        - 'n_jets_loaded': int
          Actual number of signal jets loaded
        - 'feature_names': list
          Names of features in consistent order
        - 'max_n_particles': int
          Maximum number of particles in any jet
        
        Returns None if loading fails or no signal jets are found.
    
    Notes
    -----
    The loaded signal jets are filtered to include only true signal events (label=1).
    These jets can then be randomly sampled during training to replace weakly labeled
    events in batches, enabling the model to learn from real signal.
    """
    try:
        print(f"[load_injectable_signal_jets] Loading signal jets from {signal_path}...")
        
        # Load all jets from the signal file
        # jet_name="both" loads both jets and returns (jet1_features, jet2_features, labels)
        jet1_features, jet2_features, labels = load_lhco_jets_from_h5(
            h5_filename=signal_path,
            feature_dict=input_features_dict,
            n_jets=None,  # Load all available jets
            jet_name="both",
            mom4_format="epxpypz",
            use_h5_features=True,
        )
        
        # Concatenate both jets along the particle axis to create a combined jet representation
        # jet1_features and jet2_features are both awkward arrays with structure (n_events, n_particles, n_features)
        # We concatenate them along the particle axis: (n_events, n_particles_jet1 + n_particles_jet2, n_features)
        preprocessed_features = ak.concatenate([jet1_features, jet2_features], axis=1)
        
        # Filter to keep only signal jets (label=1)
        signal_mask = labels == 1
        signal_features = preprocessed_features[signal_mask]
        signal_labels = labels[signal_mask]
        
        n_signal_available = len(signal_labels)
        n_to_load = min(num_signal_jets, n_signal_available)
        
        print(f"[load_injectable_signal_jets] Found {n_signal_available} signal jets in {signal_path}")
        print(f"[load_injectable_signal_jets] Loading {n_to_load} signal jets for injection")
        
        if n_to_load == 0:
            if num_signal_jets == 0:
                print(f"[load_injectable_signal_jets] Using baseline (no signal injection requested)")
            else:
                print(f"[load_injectable_signal_jets] Warning: No signal jets found in {signal_path}")
            return None
        
        # Select the first n_to_load signal jets
        selected_features = signal_features[:n_to_load]
        selected_labels = signal_labels[:n_to_load]
        
        # Convert awkward arrays to padded numpy arrays
        
        # Get the maximum number of particles across all selected jets
        n_particles_per_jet = ak.count(selected_features, axis=1)
        max_n_particles = int(ak.max(n_particles_per_jet))
        
        print(f"[load_injectable_signal_jets] Max particles per jet: {max_n_particles}")
        
        # Create placeholder for padded features
        n_features = len(input_features_dict)  # Number of feature types
        padded_features = np.zeros((n_to_load, max_n_particles, n_features), dtype=np.float32)
        padded_mask = np.zeros((n_to_load, max_n_particles), dtype=np.int32)
        
        # Get feature names in consistent order
        feature_names = sorted(input_features_dict.keys())
        
        # Fill the padded arrays
        for jet_idx in range(n_to_load):
            jet_features = selected_features[jet_idx]
            n_particles_in_jet = len(jet_features)
            
            # Fill features for valid particles
            for feat_idx, feat_name in enumerate(feature_names):
                try:
                    padded_features[jet_idx, :n_particles_in_jet, feat_idx] = np.array(
                        jet_features[feat_name], dtype=np.float32
                    )
                except Exception as e:
                    print(f"[load_injectable_signal_jets] Warning: Could not load feature {feat_name}: {e}")
            
            # Set mask for valid particles
            padded_mask[jet_idx, :n_particles_in_jet] = 1
        
        injectable_jets = {
            'part_features': padded_features,  # (n_jets, max_particles, n_features)
            'part_mask': padded_mask,  # (n_jets, max_particles)
            'labels': np.ones(n_to_load, dtype=np.int32),  # All signal jets
            'n_jets_loaded': n_to_load,
            'feature_names': feature_names,
            'max_n_particles': max_n_particles,
        }
        
        print(f"[load_injectable_signal_jets] Successfully loaded {n_to_load} signal jets")
        return injectable_jets
        
    except Exception as e:
        print(f"[load_injectable_signal_jets] Error loading injectable signal jets: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    

def main():
    parser = argparse.ArgumentParser(description="OmniJet-alpha catastrophic forgetting script")
    parser.add_argument("--dataset_path", default=str(os.getenv("DATASET_PATH")), type=str, help="Path to the LHCO dataset")
    parser.add_argument("--gpu_id", type=int, default=int(os.getenv("GPU_ID")), help="GPU ID to use for computation")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED")), help="Random seed for reproducibility")
    parser.add_argument("--jet_name", type=str, default=str(os.getenv("JET_NAME")), choices=["jet1", "jet2", "both"], help="Name of the jet to use from the dataset")
    parser.add_argument("--merge_strategy", type=str, default=str(os.getenv("MERGE_STRATEGY")), choices=["concat", "average", "weighted_sum", "attention"], help="Merge strategy for dijet model")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE")), help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE")), help="Learning rate")
    parser.add_argument("--train_val_split", type=float, default=float(os.getenv("TRAIN_VAL_SPLIT")), help="Train/validation split ratio")
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM")), help="Embedding dimension")
    parser.add_argument("--naming_identifier", type=str, default="", help="Optional identifier to add to the run name for easier tracking")
    parser.add_argument("--log_dir", type=str, default=str(os.getenv("LOG_DIR_AACHEN")), help="Directory for experiment logs")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == 'true', default=True, help="Use automatic class weighting for imbalanced data (default: True)")
    
    # Signal injection parameters for catastrophic forgetting
    parser.add_argument("--inject_freq", type=int, default=10, help="Inject signal jets every N batches (catastrophic forgetting experiment)")
    parser.add_argument("--num_signal_jets", type=int, default=0, help="Number of signal jets to inject into label 1 on injection batches")

    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="catastrophic-forgetting", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional, auto-generated if not provided)")
    
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

    n_jets_train = [100, 10000, 20000]               # [signal, supp, background]
    n_jets_train_artificial = [1, 10000, 20000]       # [signal, supp, background]
    n_jets_test = [1000, 1000]                          # [signal, background]
    injection_signal_jets = 25000  # Number of real signal jets to inject into label 1 during training

    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }

    # Loading the full training dataset (signal + polluted background + clean background)
    signal_path = os.path.join(args.dataset_path, "sn_25k_SR_train.h5")
    supp_background_path = os.path.join(args.dataset_path, "bg_100k_SR_supp.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    h5_files_all = [signal_path, supp_background_path, background_path]
    print("n_jets_train:",  n_jets_train)
    print("Using Jet:", args.jet_name)

    # Loading the test dataset (separate signal and background files)
    test_signal_path = os.path.join(args.dataset_path, "sn_50k_SR_test.h5")
    test_background_path = os.path.join(args.dataset_path, "bg_200k_SR_test.h5")
    h5_files_test = [test_signal_path, test_background_path]

    # Log data configuration
    data_config = {
        "dataset_path": args.dataset_path,
        "signal_file": signal_path,
        "supp_background_file": supp_background_path,
        "background_file": background_path,
        "n_jets_train_full": n_jets_train,
        "n_jets_train_artificial": n_jets_train_artificial,
        "training_strategy": "Catastrophic Forgetting with Signal Injection",
        "training_composition": f"n_jets_train_artificial={n_jets_train_artificial} (minimal signal for initial training)",
        "validation_composition": f"n_jets_train={n_jets_train} (full signal count for validation)",
        "batch_size": args.batch_size,
        "max_sequence_len": 128,
        "mom4_format": "epxpypz",
        "train_val_split": args.train_val_split,
        "features": list(input_features_dict.keys()),
        "feature_preprocessing": input_features_dict,
        "shuffle_train": True,
        "jet_name": args.jet_name,
        "signal_injection_enabled": True,
        "signal_injection_schedule": f"Inject {args.num_signal_jets} real signal jets every {args.inject_freq} batches",
        "inject_signal_every_n_batches": args.inject_freq,
        "num_signal_jets_per_injection": args.num_signal_jets,
        "initial_signal_count": n_jets_train_artificial[0],
        "full_signal_count": n_jets_train[0],
    }
    
    train_loader, val_loader = create_custom_lhco_h5_dataloaders(
        h5_files_train=h5_files_all,
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=args.batch_size,
        n_jets_train=n_jets_train,  # [signal, supp, background]
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        train_val_split=args.train_val_split,
        shuffle_train=True,
        num_workers=1,
    )

    train_loader_artificial, val_loader_artificial = create_custom_lhco_h5_dataloaders(
        h5_files_train=h5_files_all,
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=args.batch_size,
        n_jets_train=n_jets_train_artificial,  # [signal, supp, background]
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        train_val_split=args.train_val_split,
        shuffle_train=True,
        num_workers=1,
    )

    test_loader = create_lhco_h5_test_loader(
        h5_files_test=h5_files_test,
        feature_dict=input_features_dict,
        batch_size=args.batch_size,
        n_jets_test=n_jets_test,  # [signal, background]
        max_sequence_len=128,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        shuffle_test=False,
        num_workers=1,
    )

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
        n_label_1 = n_jets_train[0] + n_jets_train[1]  # signal + supp background
        n_label_0 = n_jets_train[2]  # clean background
        total = n_label_1 + n_label_0
        
        # Weight = total / (n_classes * n_samples_per_class)
        # Higher weight for minority class
        weight_label_0 = total / (2.0 * n_label_0)  # Weight for class 0 (clean background)
        weight_label_1 = total / (2.0 * n_label_1)  # Weight for class 1 (signal + polluted)
        # PyTorch CrossEntropyLoss expects weights in class order: [weight_for_class_0, weight_for_class_1]
        class_weights = [weight_label_0, weight_label_1]
        
        print(f"\n=== Weak Supervision Label Distribution ===")
        print(f"Label 0 (clean background): {n_label_0} jets → weight={weight_label_0:.4f}")
        print(f"Label 1 (signal + polluted bg): {n_label_1} jets → weight={weight_label_1:.4f}")
        print(f"  - True signal: {n_jets_train[0]}")
        print(f"  - Polluted background: {n_jets_train[1]}")
        print(f"Weight ratio (Label_1/Label_0): {weight_label_1/weight_label_0:.4f}")
        print(f"Class weights array: {class_weights}\n")
        model_kwargs["class_weights"] = class_weights
    else:
        print("Class weighting disabled - using standard CrossEntropyLoss")
        model_kwargs["class_weights"] = None


    # For constant learning rate, use ConstantLR
    scheduler_with_params = torch.optim.lr_scheduler.ConstantLR

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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Log model configuration
    model_config = {
        "architecture": "BackboneAachenClassificationLightning",
        "class_head_type": "aachen",
        "merge_strategy": args.merge_strategy,
        "use_continuous_input": True,
        "num_parameters": num_params,
        "embedding_dim": args.embedding_dim,
        "n_transformer_blocks": 8,
        "num_attention_heads": 8,
        "max_sequence_len": 128,
        "n_output_classes": 2,
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
        "max_epochs": 1,
        "test_roc_logged_per_train_batch": True,
        "embedding_visualization_per_batch": True,
        "embedding_visualization_dir": "plots/catastrophic_forgetting",
        "signal_injection_enabled": True,
        "signal_injection_freq": args.inject_freq,
        "num_signal_jets_per_injection": args.num_signal_jets,
        "gradient_clip_val": 1.0,
        "precision": "32",
        "early_stopping_patience": 15,
        "early_stopping_monitor": "val_argos",
        "checkpoint_monitor": "val_argos",
        "checkpoint_mode": "max",
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

    # AUC callback: computes ROC AUC on validation set each epoch and logs it
    auc_callback = AUCCallback()

    # ARGOS callback: computes ARGOS metric on validation set each epoch and logs it
    argos_callback = ARGOSCallback()

    # Test ROC callback: computes ROC AUC on test set after each train batch and logs it
    test_roc_callback = TestROCCallback(test_loader=test_loader)

    # Signal injection callback: injects real signal jets every N batches
    signal_injection_callback = SignalInjectionCallback(
        signal_path=signal_path,
        injection_signal_jets=injection_signal_jets,
        num_signal_jets=args.num_signal_jets,
        input_features_dict=input_features_dict,
        inject_freq=args.inject_freq,
    )

    # Embedding visualization callback: extract embeddings and plot t-SNE/UMAP after each batch
    embedding_viz_callback = EmbeddingVisualizationCallback(
        test_loader=test_loader,
        output_base_dir="plots/catastrophic_forgetting",
        log_frequency=1  # Visualize after every batch; set to >1 to reduce frequency
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
    # Ensure Lightning uses the GPU requested via --gpu_id.
    # Use explicit accelerator and devices so PL selects the correct device
    # (accelerator="auto", devices=1 will pick the first visible GPU i.e. GPU 0).
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[args.gpu_id],
        logger=loggers if loggers else False,
        callbacks=[auc_callback, argos_callback, test_roc_callback, signal_injection_callback], # embedding_viz_callback, early_stop_callback],
        log_every_n_steps=1,
        val_check_interval=1,
        gradient_clip_val=1,
        precision="32",
        num_nodes=1,
    )

    # ============================================================
    # 4. Training Loop
    # ============================================================
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION:")
    print(f"  Train loader: n_jets_train_artificial={n_jets_train_artificial} (minimal signal)")
    print(f"  Val loader: n_jets_train={n_jets_train} (full signal count)")
    print(f"  Signal injection: Every {args.inject_freq} batches, inject {args.num_signal_jets} real signal jets")
    print("="*80 + "\n")
    
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader_artificial,
            val_dataloaders=val_loader,
        )
        
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