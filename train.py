""" Python file to train the OmniJet-alpha model with continuous tokens.
COMPONENTS USED:
├── Data Loading
│   ├── load_multiple_h5_files
│   ├── ak_select_and_preprocess
│   ├── ak_pad
│   └── JetDataset
│
├── Model Architecture
│   ├── BackboneClassificationLightning
│   │   ├── BackboneTransformer (use_continuous_input=True)
│   │   └── ClassificationHead (class_head_type="attention")
│   │
│   └── Loss: CrossEntropyLoss
│
└── Training
    ├── PyTorch Lightning Trainer
    └── AdamW + Scheduler

Meant for training the supervised anomaly detection model on LHCO datasets. (Trained on 200k bkg + 25k signal jets) (Tested on 200k bkg + 50k signal jets)
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
from sklearn.metrics import roc_auc_score
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from pathlib import Path

# gabbro imports
from gabbro.utils.arrays import ak_pad
from gabbro.data.data_utils import create_lhco_h5_dataloaders
from gabbro.models.backbone import BackboneClassificationLightning, BackboneDijetClassificationLightning, BackboneAachenClassificationLightning
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files


class ExperimentLogger:
    """Handles logging of experiment configuration and results."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    def get_tensorboard_dir(self):
        """Get tensorboard log directory for this run."""
        return str(self.run_dir / "tensorboard")



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
        
        # Transformer configuration
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
        
        # Classification head settings (for class_attention type omnijet alpha)
        "class_head_hidden_dim": 128,
        "class_head_num_heads": 8,
        "class_head_num_CA_blocks": 2,
        "class_head_num_SA_blocks": 0,
        "class_head_dropout_rate": 0.1,

        # Anomaly detection head settings (for Aachen method)
        # "class_head_hidden_dim": 128,
        # "class_head_num_heads": 2,
        # "class_head_num_CA_blocks": 2,
        # "class_head_num_SA_blocks": 0,
        # "class_head_dropout_rate": 0.1,
        
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
                if isinstance(pl_module, BackboneDijetClassificationLightning):
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


def main():
    parser = argparse.ArgumentParser(description="OmniJet-alpha Anomaly Detection Training Script")
    parser.add_argument("--dataset_path", default="/.automount/net_rw/net__data_ttk/soshaw", type=str, help="Path to the LHCO dataset")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--jet_name", type=str, default="jet1", choices=["jet1", "jet2", "both"], help="Name of the jet to use from the dataset")
    parser.add_argument("--merge_strategy", type=str, default="concat", choices=["concat", "average", "weighted_sum", "attention"], help="Merge strategy for dijet model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_val_split", type=float, default=0.7, help="Train/validation split ratio")
    parser.add_argument("--n_jets_train", type=list, default=[25000, 200000], help="Number of jets per class for training [signal, background]")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--log_dir", type=str, default="dijet_expts", help="Directory for experiment logs")
    parser.add_argument("--use_class_weights", type=lambda x: x.lower() == 'true', default=True, help="Use automatic class weighting for imbalanced data (default: True)")
    args = parser.parse_args()

    # ============================================================
    # 0. Initialize Experiment Logger
    # ============================================================
    exp_logger = ExperimentLogger(log_dir=args.log_dir)
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
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_train.h5")
    
    h5_files_all = [signal_path, background_path]

    print("n_jets_train:", args.n_jets_train)
    print("Using Jet:", args.jet_name)

    # Log data configuration
    data_config = {
        "dataset_path": args.dataset_path,
        "signal_file": signal_path,
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
    
    train_loader, val_loader = create_lhco_h5_dataloaders(
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
    # 3. Create Model
    # ============================================================

    # Calculate class weights for imbalanced dataset
    model_kwargs = create_model_config(input_features_dict, args)
    
    if args.use_class_weights:
        # n_jets_train = [signal, background] = [25000, 200000]
        n_signal = args.n_jets_train[0]
        n_background = args.n_jets_train[1]
        total = n_signal + n_background
        
        # Weight = total / (n_classes * n_samples_per_class)
        # Higher weight for minority class (signal)
        # Note: In LHCO files, signal=1, background=0
        weight_background = total / (2.0 * n_background)  # Weight for class 0
        weight_signal = total / (2.0 * n_signal)  # Weight for class 1
        # PyTorch CrossEntropyLoss expects weights in class order: [weight_for_class_0, weight_for_class_1]
        class_weights = [weight_background, weight_signal]
        
        print(f"\n=== Supervised Training Label Distribution ===")
        print(f"Class 0 (Background): {n_background} jets → weight={weight_background:.4f}")
        print(f"Class 1 (Signal): {n_signal} jets → weight={weight_signal:.4f}")
        print(f"Weight ratio (Signal/Background): {weight_signal/weight_background:.4f}")
        print(f"Class weights array: {class_weights}\n")
        model_kwargs["class_weights"] = class_weights
    else:
        print("Class weighting disabled - using standard CrossEntropyLoss")
        model_kwargs["class_weights"] = None

    scheduler_with_params = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=1000000,
        eta_min=1e-6, # minimum learning rate
    )

    # -------------------------------------------------------------------------
    # ---------------------- Single Jet Data Model ----------------------------
    # -------------------------------------------------------------------------

    # Initialize the Backbone + Classification Head (Single Jet)
    # model = BackboneClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": 1e-3,
    #         "weight_decay": 1e-2,
    #         "partial": True,
    #     },
    #     scheduler=scheduler_with_params,
    #     class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_loss",
    #         "mode": "min",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    # -------------------------------------------------------------------------
    # ------------------------- DiJet Data Model ------------------------------
    # -------------------------------------------------------------------------

    model = BackboneDijetClassificationLightning(
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "partial": True,
        },
        scheduler=scheduler_with_params,
        merge_strategy=args.merge_strategy,  # other options: "average", "weighted_sum", "attention"
        class_head_type="class_attention",  # other options: "linear_average_pool", "summation", "flatten"
        model_kwargs=model_kwargs,
        use_continuous_input=True,
        scheduler_lightning_kwargs={
            "monitor": "val_loss",
            "mode": "min",
            "interval": "step",
            "frequency": 1,
        },
    )

    # -------------------------------------------------------------------------
    # ------------------ Aachen Anomaly Detection Model -----------------------
    # -------------------------------------------------------------------------

    # model = BackboneAachenClassificationLightning(
    #     optimizer=torch.optim.AdamW,
    #     optimizer_kwargs={
    #         "lr": 1e-3,
    #         "weight_decay": 1e-2,
    #         "partial": True,
    #     },
    #     scheduler=scheduler_with_params,
    #     merge_strategy=args.merge_strategy,  # other options: "average", "weighted_sum", "attention"
    #     model_kwargs=model_kwargs,
    #     use_continuous_input=True,
    #     scheduler_lightning_kwargs={
    #         "monitor": "val_loss",
    #         "mode": "min",
    #         "interval": "step",
    #         "frequency": 1,
    #     },
    # )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Log model configuration
    model_config = {
        "architecture": "BackboneDijetClassificationLightning",
        "merge_strategy": args.merge_strategy,
        "class_head_type": "class_attention",
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
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "partial": True,
        },
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {
            "T_max": 1000000,
            "eta_min": 1e-6,
        },
        "max_steps": args.max_steps,
        "gradient_clip_val": 1.0,
        "precision": "32",
        "early_stopping_patience": 15,
        "early_stopping_monitor": "val_loss",
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
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_logger.get_checkpoint_dir(),
        filename="anomaly_detector_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Alternatively, monitor AUC instead of loss
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=exp_logger.get_checkpoint_dir(),
    #     filename="anomaly_detector_{epoch:02d}_{val_auc:.4f}",
    #     monitor="val_auc",
    #     mode="max",
    #     save_top_k=3,
    #     save_last=True,
    # )
    
    # Early stopping disabled
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=5,
    #     mode="min",
    # )

    # AUC callback: computes ROC AUC on validation set each epoch and logs it
    auc_callback = AUCCallback()

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=exp_logger.get_tensorboard_dir(),
        name="",
        version="",
    )

    # Create trainer
    print("Starting training...")
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, auc_callback],  # early_stop_callback removed
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
        print("=" * 80 + "\n")
        
    except Exception as e:
        # Log error if training fails
        error_info = {
            "training_completed": False,
            "error": str(e),
            "timestamp_error": datetime.now().isoformat(),
        }
        exp_logger.log_results(error_info)
        print(f"Training failed! Error logged to: {exp_logger.run_dir}")
        raise


if __name__ == "__main__":
    main()
