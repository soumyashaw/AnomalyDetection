"""
Visualize clustering patterns in signal vs background data using dimensionality reduction.

This script loads signal and background jet data using the same preprocessing pipeline
as train.py and applies multiple dimensionality reduction techniques to check for 
emergent clustering before training.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
import argparse

# gabbro imports - same as train.py
from gabbro.data.data_utils import create_lhco_h5_dataloaders

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)


def flatten_jet_features(data):
    """Flatten jet constituent features into 1D vectors."""
    # data shape: (n_jets, n_constituents, n_features)
    n_jets = data.shape[0]
    
    # Flatten each jet's constituents
    flattened = data.reshape(n_jets, -1)
    
    print(f"Flattened shape: {flattened.shape}")
    return flattened


def aggregate_jet_features(data):
    """Create aggregate features per jet (mean, std, max, min of constituents)."""
    print("Creating aggregate features...")
    
    # data shape: (n_jets, n_constituents, n_features)
    features = []
    
    # Aggregate statistics across constituents (ignoring padding zeros)
    # Compute mask for non-zero entries (padding is all zeros)
    mask = np.any(data != 0, axis=-1, keepdims=True)  # (n_jets, n_constituents, 1)
    
    # Masked aggregation
    masked_data = np.where(mask, data, np.nan)
    
    features.append(np.nanmean(masked_data, axis=1))  # Mean across constituents
    features.append(np.nanstd(masked_data, axis=1))   # Std across constituents
    features.append(np.nanmax(masked_data, axis=1))   # Max across constituents
    features.append(np.nanmin(masked_data, axis=1))   # Min across constituents
    
    # Count non-padding constituents
    n_constituents = np.sum(mask[:, :, 0], axis=1, keepdims=True)
    features.append(n_constituents)
    
    # Concatenate all aggregate features
    aggregated = np.concatenate(features, axis=1)
    
    # Replace any remaining NaN with 0
    aggregated = np.nan_to_num(aggregated, nan=0.0)
    
    print(f"Aggregated shape: {aggregated.shape}")
    return aggregated


def plot_dimensionality_reduction(X, y, method_name, perplexity=30, n_neighbors=15, 
                                   min_dist=0.1, save_path=None):
    """Apply dimensionality reduction and plot results."""
    
    print(f"\n{'='*60}")
    print(f"Running {method_name}...")
    print(f"{'='*60}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method_name == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                      verbose=1, n_iter=1000)
        X_reduced = reducer.fit_transform(X_scaled)
        title_suffix = f"(perplexity={perplexity})"
        
    elif method_name == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        variance = reducer.explained_variance_ratio_
        title_suffix = f"(Var: {variance[0]:.2%} + {variance[1]:.2%} = {sum(variance):.2%})"
        print(f"Explained variance: PC1={variance[0]:.4f}, PC2={variance[1]:.4f}")
        
    elif method_name == "UMAP":
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42, verbose=True)
        X_reduced = reducer.fit_transform(X_scaled)
        title_suffix = f"(n_neighbors={n_neighbors}, min_dist={min_dist})"
        
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Scatter with transparency
    ax = axes[0]
    scatter = ax.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], 
                        c='blue', label='Background', alpha=0.3, s=10, edgecolors='none')
    scatter = ax.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], 
                        c='red', label='Signal', alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)
    ax.set_title(f"{method_name} Projection {title_suffix}", fontsize=14, fontweight='bold')
    ax.legend(markerscale=2, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 2D histogram/density
    ax = axes[1]
    
    # Create density plots
    from scipy.stats import gaussian_kde
    
    # Background density
    if np.sum(y == 0) > 100:
        xy_bg = X_reduced[y == 0].T
        z_bg = gaussian_kde(xy_bg)(xy_bg)
        ax.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], 
                  c=z_bg, cmap='Blues', label='Background', 
                  alpha=0.3, s=10, edgecolors='none')
    
    # Signal density
    if np.sum(y == 1) > 100:
        xy_sig = X_reduced[y == 1].T
        z_sig = gaussian_kde(xy_sig)(xy_sig)
        scatter_sig = ax.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], 
                                c=z_sig, cmap='Reds', label='Signal', 
                                alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)
    ax.set_title(f"{method_name} Density View", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    return X_reduced


def compute_separation_metrics(X_reduced, y):
    """Compute metrics to quantify signal/background separation."""
    
    signal_data = X_reduced[y == 1]
    background_data = X_reduced[y == 0]
    
    # Compute centroids
    signal_centroid = np.mean(signal_data, axis=0)
    background_centroid = np.mean(background_data, axis=0)
    
    # Distance between centroids
    centroid_distance = np.linalg.norm(signal_centroid - background_centroid)
    
    # Average distance within each class
    signal_spread = np.mean(np.linalg.norm(signal_data - signal_centroid, axis=1))
    background_spread = np.mean(np.linalg.norm(background_data - background_centroid, axis=1))
    
    # Separation score (centroid distance / average spread)
    separation_score = centroid_distance / (0.5 * (signal_spread + background_spread))
    
    print(f"\nSeparation Metrics:")
    print(f"  Centroid distance: {centroid_distance:.4f}")
    print(f"  Signal spread: {signal_spread:.4f}")
    print(f"  Background spread: {background_spread:.4f}")
    print(f"  Separation score: {separation_score:.4f} (higher = better separation)")
    
    return {
        'centroid_distance': centroid_distance,
        'signal_spread': signal_spread,
        'background_spread': background_spread,
        'separation_score': separation_score
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize clustering in signal vs background data')
    parser.add_argument('--signal_path', type=str, 
                       default='/.automount/net_rw/net__data_ttk/soshaw/sn_25k_SR_train.h5',
                       help='Path to signal h5 file')
    parser.add_argument('--background_path', type=str,
                       default='/.automount/net_rw/net__data_ttk/soshaw/bg_100k_SR_supp.h5',
                       help='Path to background h5 file')
    parser.add_argument('--n_signal', type=int, default=5000,
                       help='Number of signal samples to load')
    parser.add_argument('--n_background', type=int, default=100000,
                       help='Number of background samples to load')
    parser.add_argument('--feature_type', type=str, default='aggregate',
                       choices=['flatten', 'aggregate'],
                       help='Feature extraction method')
    parser.add_argument('--output_dir', type=str, default='plots/clustering',
                       help='Directory to save plots')
    parser.add_argument('--max_sequence_len', type=int, default=128,
                       help='Maximum sequence length for padding')
    parser.add_argument('--jet_name', type=str, default='jet1',
                       help='Name of jet in h5 file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CLUSTERING VISUALIZATION")
    print("="*60)
    
    # Define feature preprocessing (same as train.py)
    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }
    
    print("\nFeature preprocessing configuration:")
    for feat_name, feat_config in input_features_dict.items():
        print(f"  {feat_name}: {feat_config}")
    
    # Load data using the EXACT same pipeline as train.py (with real labels)
    print("\nLoading data using create_lhco_h5_dataloaders...")
    h5_files = [args.signal_path, args.background_path]
    n_jets = [args.n_signal, args.n_background]
    
    # We don't need the DataLoader, just the data
    # So we'll load with train_val_split=1.0 to get all data in train split
    # This preserves real labels: signal=1, background=0
    train_loader, val_loader = create_lhco_h5_dataloaders(
        h5_files_train=h5_files,
        h5_files_val=None,
        feature_dict=input_features_dict,
        batch_size=10000,  # Large batch to get all data at once
        n_jets_train=n_jets,
        max_sequence_len=args.max_sequence_len,
        mom4_format="epxpypz",
        jet_name=args.jet_name,
        train_val_split=1.0,  # All data goes to "train" split
        shuffle_train=False,  # Don't shuffle for visualization
        num_workers=0,
    )
    
    # Extract all data from the loader
    all_features = []
    all_labels = []
    
    for batch in train_loader:
        all_features.append(batch["part_features"].numpy())
        all_labels.append(batch["jet_type_labels"].numpy())
    
    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nLoaded data shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Signal (label=1): {np.sum(all_labels == 1)}")
    print(f"Background (label=0): {np.sum(all_labels == 0)}")
    
    # Extract features
    print(f"\nExtracting features using '{args.feature_type}' method...")
    if args.feature_type == 'flatten':
        features = flatten_jet_features(all_features)
    else:  # aggregate
        features = aggregate_jet_features(all_features)
    
    # Use labels directly (no need to create separate signal/background arrays)
    X = features
    y = all_labels
    
    print(f"\nDataset for visualization:")
    print(f"  Total samples: {len(X)}")
    print(f"  Signal: {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
    print(f"  Background: {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # For faster computation with large datasets, subsample for visualization
    if len(X) > 50000:
        print(f"\nSubsampling to 50k for visualization (keeping signal/background ratio)...")
        n_signal_vis = min(5000, np.sum(y == 1))
        n_background_vis = min(45000, np.sum(y == 0))
        
        signal_idx = np.where(y == 1)[0]
        background_idx = np.where(y == 0)[0]
        
        selected_signal = np.random.choice(signal_idx, n_signal_vis, replace=False)
        selected_background = np.random.choice(background_idx, n_background_vis, replace=False)
        
        selected_idx = np.concatenate([selected_signal, selected_background])
        np.random.shuffle(selected_idx)
        
        X_vis = X[selected_idx]
        y_vis = y[selected_idx]
    else:
        X_vis = X
        y_vis = y
    
    print(f"\nVisualization dataset size: {len(X_vis)}")
    
    # 1. PCA (fast, linear)
    print("\n" + "="*60)
    print("1. Principal Component Analysis (PCA)")
    print("="*60)
    X_pca = plot_dimensionality_reduction(
        X_vis, y_vis, "PCA",
        save_path=output_dir / f"pca_{args.feature_type}.png"
    )
    metrics_pca = compute_separation_metrics(X_pca, y_vis)
    
    # 2. t-SNE (slower, nonlinear, good for local structure)
    print("\n" + "="*60)
    print("2. t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    print("="*60)
    
    # Try different perplexities
    for perplexity in [30, 50]:
        X_tsne = plot_dimensionality_reduction(
            X_vis, y_vis, "t-SNE", perplexity=perplexity,
            save_path=output_dir / f"tsne_perp{perplexity}_{args.feature_type}.png"
        )
        metrics_tsne = compute_separation_metrics(X_tsne, y_vis)
    
    # 3. UMAP (fast, preserves both local and global structure)
    print("\n" + "="*60)
    print("3. Uniform Manifold Approximation and Projection (UMAP)")
    print("="*60)
    
    # Try different hyperparameters
    for n_neighbors in [15, 50]:
        for min_dist in [0.1, 0.5]:
            X_umap = plot_dimensionality_reduction(
                X_vis, y_vis, "UMAP", 
                n_neighbors=n_neighbors, min_dist=min_dist,
                save_path=output_dir / f"umap_n{n_neighbors}_d{min_dist}_{args.feature_type}.png"
            )
            metrics_umap = compute_separation_metrics(X_umap, y_vis)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Plots saved to: {output_dir.absolute()}")
    print("\nInterpretation:")
    print("  - Clear clustering = Signal is distinguishable from background")
    print("  - Mixed/overlapping = Signal is embedded within background (harder task)")
    print("  - Separation score > 1 = Good separation")
    print("  - Separation score < 0.5 = Poor separation (weak supervision will struggle)")


if __name__ == "__main__":
    main()
