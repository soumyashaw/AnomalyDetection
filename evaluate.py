"""Evaluate trained anomaly detection model with comprehensive metrics and plots."""

import os
import json
import torch
import argparse
import numpy as np
import pickle
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader, TensorDataset

from gabbro.models.backbone import (
    BackboneClassificationLightning,
    BackboneDijetClassificationLightning,
    BackboneAachenClassificationLightning,
)
from gabbro.data.data_utils import create_lhco_h5_test_loader


class DataCache:
    """Cache loaded HDF5 data to avoid repeated disk I/O."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Generate unique cache key based on data configuration."""
        # Create a string representation of the configuration
        config_str = f"{h5_files}_{n_jets}_{feature_dict}_{max_sequence_len}_{model_type}"
        # Hash it to get a short, unique filename
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()
    
    def get_cache_path(self, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Generate cache filename."""
        cache_key = self._get_cache_key(h5_files, n_jets, feature_dict, max_sequence_len, model_type)
        file_str = "_".join([Path(f).stem for f in h5_files])
        n_jets_str = "_".join(map(str, n_jets))
        type_str = "dijet" if model_type == "dijet" else "single"
        return self.cache_dir / f"data_{type_str}_{file_str}_{n_jets_str}_{cache_key}.pkl"
    
    def load(self, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Load dataset from cache if available."""
        cache_path = self.get_cache_path(h5_files, n_jets, feature_dict, max_sequence_len, model_type)
        if cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Successfully loaded {len(cached_data['labels'])} samples from cache")
                return cached_data
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}). Loading from HDF5 files.")
                return None
        return None
    
    def save(self, data_dict, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Save dataset to cache."""
        cache_path = self.get_cache_path(h5_files, n_jets, feature_dict, max_sequence_len, model_type)
        print(f"Saving data to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Successfully cached {len(data_dict['labels'])} samples")
            # Print cache file size
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"Cache file size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")


def extract_data_from_loader(dataloader, model_type="single"):
    """Extract all data from a DataLoader into tensors for caching.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to extract data from
    model_type : str
        Type of model: "single", "dijet", or "aachen"
        
    Returns
    -------
    dict
        Dictionary with tensors for features, masks, labels
        For dijet: includes 'features', 'features_jet2', 'masks', 'masks_jet2', 'labels'
        For single: includes 'features', 'masks', 'labels'
    """
    all_features = []
    all_features_jet2 = []
    all_masks = []
    all_masks_jet2 = []
    all_labels = []
    
    print("Extracting data from DataLoader...")
    for batch_idx, batch in enumerate(dataloader):
        all_features.append(batch["part_features"])
        all_masks.append(batch["part_mask"])
        all_labels.append(batch["jet_type_labels"])
        
        # For dijet models, also extract jet2 data
        if model_type == "dijet" and "part_features_jet2" in batch:
            all_features_jet2.append(batch["part_features_jet2"])
            all_masks_jet2.append(batch["part_mask_jet2"])
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Extracted {batch_idx + 1}/{len(dataloader)} batches")
    
    result = {
        "features": torch.cat(all_features, dim=0),
        "masks": torch.cat(all_masks, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    
    # Add jet2 data for dijet models
    if model_type == "dijet" and all_features_jet2:
        result["features_jet2"] = torch.cat(all_features_jet2, dim=0)
        result["masks_jet2"] = torch.cat(all_masks_jet2, dim=0)
    
    return result


def create_loader_from_cached_data(cached_data, batch_size, model_type="single"):
    """Create a DataLoader from cached tensor data.
    
    Parameters
    ----------
    cached_data : dict
        Dictionary with cached tensors
        For single: 'features', 'masks', 'labels'
        For dijet: 'features', 'features_jet2', 'masks', 'masks_jet2', 'labels'
    batch_size : int
        Batch size for DataLoader
    model_type : str
        Type of model: "single", "dijet", or "aachen"
        
    Returns
    -------
    DataLoader
        DataLoader wrapping the cached data
    """
    if model_type == "dijet":
        # Dijet dataset with both jets
        class DijetCachedDataset(torch.utils.data.Dataset):
            def __init__(self, features, features_jet2, masks, masks_jet2, labels):
                self.features = features
                self.features_jet2 = features_jet2
                self.masks = masks
                self.masks_jet2 = masks_jet2
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    "part_features": self.features[idx],
                    "part_features_jet2": self.features_jet2[idx],
                    "part_mask": self.masks[idx],
                    "part_mask_jet2": self.masks_jet2[idx],
                    "jet_type_labels": self.labels[idx],
                    "jet_features": torch.tensor([]),
                }
        
        dataset = DijetCachedDataset(
            cached_data["features"],
            cached_data["features_jet2"],
            cached_data["masks"],
            cached_data["masks_jet2"],
            cached_data["labels"]
        )
    else:
        # Single-jet dataset
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, features, masks, labels):
                self.features = features
                self.masks = masks
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    "part_features": self.features[idx],
                    "part_mask": self.masks[idx],
                    "jet_type_labels": self.labels[idx],
                    "jet_features": torch.tensor([]),
                }
        
        dataset = CachedDataset(
            cached_data["features"],
            cached_data["masks"],
            cached_data["labels"]
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No need for workers with cached data
        pin_memory=True,
    )


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, checkpoint_path, gpu_id, model_type="single", output_dir="evaluation_results"):
        """Initialize evaluator.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint
        gpu_id : int
            GPU device ID
        model_type : str
            Model architecture type: 'single', 'dijet', or 'aachen'
        output_dir : str
            Directory to save evaluation results
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory with timestamp
        self.eval_dir = self.output_dir / f"eval_{self.checkpoint_path.split('/')[1]}_{self.timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model based on type
        print(f"Loading {model_type} model from {checkpoint_path}")
        if self.model_type == "dijet":
            self.model = BackboneDijetClassificationLightning.load_from_checkpoint(checkpoint_path)
        elif self.model_type == "aachen":
            self.model = BackboneAachenClassificationLightning.load_from_checkpoint(checkpoint_path)
        elif self.model_type == "single":
            self.model = BackboneClassificationLightning.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from 'single', 'dijet', or 'aachen'")
        
        self.model.eval()
        
        # Validate GPU availability
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if gpu_id >= n_gpus:
                print(f"Warning: GPU {gpu_id} not available (only {n_gpus} GPU(s) found). Using GPU 0.")
                gpu_id = 0
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            print("Warning: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
        
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Storage for results
        self.results = {}
        
    def predict(self, dataloader):
        """Generate predictions on data.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for test data
            
        Returns
        -------
        predictions : np.ndarray
            Predicted probabilities for signal class
        labels : np.ndarray
            True labels
        logits : np.ndarray
            Raw model outputs (logits)
        """
        all_preds = []
        all_labels = []
        all_logits = []
        
        print("Generating predictions...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                labels = batch["jet_type_labels"]
                
                # Forward pass based on model type
                if self.model_type == "dijet":
                    # Dijet model needs both jets
                    X1 = batch["part_features"].to(self.device)
                    X2 = batch["part_features_jet2"].to(self.device)
                    mask1 = batch["part_mask"].to(self.device)
                    mask2 = batch["part_mask_jet2"].to(self.device)
                    logits = self.model(X1, mask1, X2, mask2)
                else:
                    # Single-jet or Aachen model needs only one jet
                    X = batch["part_features"].to(self.device)
                    mask = batch["part_mask"].to(self.device)
                    logits = self.model(X, mask)
                
                probs = torch.softmax(logits, dim=1)
                
                # Collect predictions (probability of signal class)
                all_preds.append(probs[:, 1].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        predictions = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        logits = np.concatenate(all_logits)
        
        print(f"Generated predictions for {len(predictions)} samples")
        return predictions, labels, logits
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """Calculate classification metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        threshold : float
            Classification threshold
            
        Returns
        -------
        metrics : dict
            Dictionary of metrics
        """
        # Binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred_binary, 
            target_names=['Background', 'Signal'],
            output_dict=True
        )
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        metrics = {
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "threshold_used": float(threshold),
            "optimal_threshold": float(optimal_threshold),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "n_samples": len(y_true),
            "n_signal": int(np.sum(y_true)),
            "n_background": int(len(y_true) - np.sum(y_true)),
        }
        
        return metrics, (fpr, tpr, roc_thresholds), (precision, recall, pr_thresholds)
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")
    
    def plot_precision_recall_curve(self, precision, recall, avg_precision):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curve saved to {save_path}")
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Background', 'Signal'],
                   yticklabels=['Background', 'Signal'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_score_distribution(self, y_true, y_pred):
        """Plot distribution of prediction scores."""
        plt.figure(figsize=(10, 6))
        
        # Separate scores by class
        signal_scores = y_pred[y_true == 1]
        background_scores = y_pred[y_true == 0]
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        plt.hist(background_scores, bins=bins, alpha=0.6, label='Background', 
                color='blue', density=True)
        plt.hist(signal_scores, bins=bins, alpha=0.6, label='Signal', 
                color='red', density=True)
        
        plt.xlabel('Prediction Score (Signal Probability)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Prediction Scores', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.eval_dir / 'score_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score distribution saved to {save_path}")
    
    def plot_threshold_analysis(self, fpr, tpr, thresholds):
        """Plot metrics vs threshold."""
        # Skip the last threshold (inf)
        thresholds = thresholds[:-1]
        tpr = tpr[:-1]
        fpr = fpr[:-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: TPR and FPR vs threshold
        ax1.plot(thresholds, tpr, label='True Positive Rate', linewidth=2)
        ax1.plot(thresholds, fpr, label='False Positive Rate', linewidth=2)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Rate', fontsize=12)
        ax1.set_title('TPR and FPR vs Threshold', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Youden's J statistic
        j_scores = tpr - fpr
        ax2.plot(thresholds, j_scores, linewidth=2, color='green')
        optimal_idx = np.argmax(j_scores)
        ax2.axvline(thresholds[optimal_idx], color='red', linestyle='--', 
                   label=f'Optimal = {thresholds[optimal_idx]:.3f}')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel("Youden's J Statistic", fontsize=12)
        ax2.set_title("Threshold Optimization", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.eval_dir / 'threshold_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Threshold analysis saved to {save_path}")
    
    def save_results(self, metrics, y_true, y_pred):
        """Save evaluation results to JSON."""
        results = {
            "checkpoint_path": str(self.checkpoint_path),
            "evaluation_timestamp": self.timestamp,
            "metrics": metrics,
            "device": str(self.device),
        }
        
        # Save to JSON
        results_path = self.eval_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results saved to {results_path}")
        
        # Save predictions
        predictions_path = self.eval_dir / 'predictions.npz'
        np.savez(predictions_path, 
                y_true=y_true, 
                y_pred=y_pred)
        print(f"Predictions saved to {predictions_path}")
        
        # Create summary log
        summary_path = self.eval_dir / 'evaluation_summary.log'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANOMALY DETECTION MODEL EVALUATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            f.write("-" * 80 + "\n")
            f.write("METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"ROC AUC:              {metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision:    {metrics['average_precision']:.4f}\n")
            f.write(f"Accuracy:             {metrics['accuracy']:.4f}\n")
            f.write(f"Sensitivity (TPR):    {metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity (TNR):    {metrics['specificity']:.4f}\n")
            f.write(f"Threshold Used:       {metrics['threshold_used']:.4f}\n")
            f.write(f"Optimal Threshold:    {metrics['optimal_threshold']:.4f}\n\n")
            f.write("-" * 80 + "\n")
            f.write("DATA STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples:        {metrics['n_samples']}\n")
            f.write(f"Signal Jets:          {metrics['n_signal']}\n")
            f.write(f"Background Jets:      {metrics['n_background']}\n\n")
            f.write("-" * 80 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write(f"True Negatives:  {metrics['confusion_matrix'][0][0]}\n")
            f.write(f"False Positives: {metrics['confusion_matrix'][0][1]}\n")
            f.write(f"False Negatives: {metrics['confusion_matrix'][1][0]}\n")
            f.write(f"True Positives:  {metrics['confusion_matrix'][1][1]}\n")
        print(f"Summary saved to {summary_path}")
    
    def evaluate(self, test_loader, threshold=0.5):
        """Run full evaluation pipeline.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        threshold : float
            Classification threshold
        """
        print("\n" + "=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80 + "\n")
        
        # Generate predictions
        y_pred, y_true, logits = self.predict(test_loader)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics, (fpr, tpr, roc_thresholds), (precision, recall, pr_thresholds) = \
            self.calculate_metrics(y_true, y_pred, threshold)
        
        # Print key metrics
        print("\n" + "-" * 80)
        print("KEY METRICS")
        print("-" * 80)
        print(f"ROC AUC:              {metrics['roc_auc']:.4f}")
        print(f"Average Precision:    {metrics['average_precision']:.4f}")
        print(f"Accuracy:             {metrics['accuracy']:.4f}")
        print(f"Sensitivity (TPR):    {metrics['sensitivity']:.4f}")
        print(f"Specificity (TNR):    {metrics['specificity']:.4f}")
        print("-" * 80 + "\n")
        
        # Generate plots
        print("Generating plots...")
        self.plot_roc_curve(fpr, tpr, metrics['roc_auc'])
        self.plot_precision_recall_curve(precision, recall, metrics['average_precision'])
        self.plot_confusion_matrix(np.array(metrics['confusion_matrix']))
        self.plot_score_distribution(y_true, y_pred)
        self.plot_threshold_analysis(fpr, tpr, roc_thresholds)
        
        # Save results
        print("\nSaving results...")
        self.save_results(metrics, y_true, y_pred)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.eval_dir}")
        print("=" * 80 + "\n")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection Model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="single",
                       choices=["single", "dijet", "aachen"],
                       help="Model architecture type")
    parser.add_argument("--dataset_path", type=str, 
                       default="/.automount/net_rw/net__data_ttk/soshaw",
                       help="Path to LHCO dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--n_jets_test", type=int, nargs='+', default=[50000, 200000],
                       help="Number of jets per class for testing [signal, background]")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save evaluation results")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use for evaluation")
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear cached data and reload from HDF5 files")
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dir = Path(".cache/evaluation")
        if cache_dir.exists():
            import shutil
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared.")
    
    # Define input features (same as training)
    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }
    
    # Load test data
    # NOTE: Order must match training! Signal first, then background
    signal_path = os.path.join(args.dataset_path, "sn_50k_SR_test.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_test.h5")
    h5_files_test = [signal_path, background_path]
    
    # Determine jet_name based on model type
    if args.model_type == "dijet":
        jet_name = "both"
        print("Loading both jets for dijet model evaluation...")
    else:
        jet_name = "jet1"
        print("Loading single jet for evaluation...")
    
    # Initialize cache
    cache = DataCache(cache_dir=".cache/evaluation")
    
    # Try to load from cache (works for both single-jet and dijet models now)
    print("Checking for cached data...")
    cached_data = cache.load(
        h5_files=h5_files_test,
        n_jets=args.n_jets_test,
        feature_dict=input_features_dict,
        max_sequence_len=128,
        model_type=args.model_type
    )
    
    if cached_data is not None:
        # Create DataLoader from cached data
        print("Creating DataLoader from cached data...")
        test_loader = create_loader_from_cached_data(cached_data, args.batch_size, args.model_type)
    else:
        # Load from HDF5 files
        print(f"Loading test data from HDF5 files (jet_name={jet_name})...")
        test_loader = create_lhco_h5_test_loader(
            h5_files_test=h5_files_test,
            feature_dict=input_features_dict,
            batch_size=args.batch_size,
            n_jets_test=args.n_jets_test,
            max_sequence_len=128,
            mom4_format="epxpypz",
            jet_name=jet_name,
            shuffle_test=False,
            num_workers=1,
        )
        
        # Extract and cache the data for future runs
        print("Caching loaded data for future evaluations...")
        cached_data = extract_data_from_loader(test_loader, model_type=args.model_type)
        cache.save(
            data_dict=cached_data,
            h5_files=h5_files_test,
            n_jets=args.n_jets_test,
            feature_dict=input_features_dict,
            max_sequence_len=128,
            model_type=args.model_type
        )
        
        # Recreate loader from cached data to ensure consistent behavior
        test_loader = create_loader_from_cached_data(cached_data, args.batch_size, args.model_type)
    
    # Initialize evaluator and run evaluation
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        gpu_id=args.gpu_id,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    metrics = evaluator.evaluate(test_loader, threshold=args.threshold)

if __name__ == "__main__":
    main()
