"""
Script to load trained VQ-VAE model and plot original vs reconstructed jet distributions.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak

from gabbro.data.loading import load_lhco_jets_from_h5
from gabbro.models.vqvae import VQVAETransformer
from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess


def load_trained_model(checkpoint_path, device='cuda:0'):
    """Load a trained VQ-VAE model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the saved model checkpoint
    device : str
        Device to load model on
        
    Returns
    -------
    model : VQVAETransformer
        Loaded model in eval mode
    config : dict
        Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    
    # Create model with saved config
    model = VQVAETransformer(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        conditional_dim=config['conditional_dim'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        vq_kwargs=config['vq_kwargs'],
        causal_decoder=config['causal_decoder'],
        max_sequence_len=config['max_sequence_len'],
        input_features_dict=config['input_features_dict'],
        old_transformer_implementation=config.get('old_transformer_implementation', False),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Config: latent_dim={config['latent_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_codes={config['vq_kwargs']['num_codes']}")
    
    return model, config


def reconstruct_jets(model, x, mask, device='cuda:0'):
    """Reconstruct jets using the VQ-VAE model.
    
    Parameters
    ----------
    model : VQVAETransformer
        Trained model
    x : torch.Tensor
        Input jet features (batch, seq_len, n_features)
    mask : torch.Tensor
        Particle mask (batch, seq_len)
    device : str
        Device to run on
        
    Returns
    -------
    x_reco : torch.Tensor
        Reconstructed features
    vq_out : dict
        VQ-VAE output dictionary
    """
    x = x.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        x_reco, vq_out = model(x=x, mask=mask)
    
    return x_reco.cpu(), vq_out


def plot_feature_distributions(
    x_original,
    x_reconstructed,
    mask,
    feature_names,
    feature_dict,
    output_dir='plots',
    n_bins=50,
):
    """Plot distributions of original vs reconstructed features.
    
    Parameters
    ----------
    x_original : torch.Tensor or np.ndarray
        Original features (batch, seq_len, n_features)
    x_reconstructed : torch.Tensor or np.ndarray
        Reconstructed features (batch, seq_len, n_features)
    mask : torch.Tensor or np.ndarray
        Particle mask (batch, seq_len)
    feature_names : list
        List of feature names
    feature_dict : dict
        Feature preprocessing dictionary
    output_dir : str
        Directory to save plots
    n_bins : int
        Number of histogram bins
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if torch.is_tensor(x_original):
        x_original = x_original.cpu().numpy()
    if torch.is_tensor(x_reconstructed):
        x_reconstructed = x_reconstructed.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    n_features = len(feature_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(feature_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get feature data (flatten and mask)
        orig_flat = x_original[:, :, idx][mask.astype(bool)]
        reco_flat = x_reconstructed[:, :, idx][mask.astype(bool)]
        
        # Determine range
        vmin = min(orig_flat.min(), reco_flat.min())
        vmax = max(orig_flat.max(), reco_flat.max())
        bins = np.linspace(vmin, vmax, n_bins)
        
        # Plot histograms
        ax.hist(orig_flat, bins=bins, alpha=0.5, label='Original', 
                color='blue', density=True, histtype='stepfilled')
        ax.hist(reco_flat, bins=bins, alpha=0.5, label='Reconstructed', 
                color='red', density=True, histtype='stepfilled')
        
        # Labels
        ax.set_xlabel(feat_name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add preprocessing info as title
        if feat_name in feature_dict and feature_dict[feat_name] is not None:
            if 'func' in feature_dict[feat_name] and feature_dict[feat_name]['func']:
                ax.set_title(f"{feat_name} ({feature_dict[feat_name]['func']})")
            else:
                ax.set_title(feat_name)
        else:
            ax.set_title(feat_name)
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150)
    print(f"Saved feature distributions to {output_dir}/feature_distributions.png")
    plt.close()


def plot_particle_multiplicity(mask, output_dir='plots'):
    """Plot particle multiplicity distribution.
    
    Parameters
    ----------
    mask : torch.Tensor or np.ndarray
        Particle mask (batch, seq_len)
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Count particles per jet
    n_particles = mask.sum(axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.hist(n_particles, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of particles per jet')
    plt.ylabel('Number of jets')
    plt.title('Jet Particle Multiplicity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_multiplicity.png'), dpi=150)
    print(f"Saved particle multiplicity to {output_dir}/particle_multiplicity.png")
    plt.close()


def plot_reconstruction_error(
    x_original,
    x_reconstructed,
    mask,
    feature_names,
    output_dir='plots',
):
    """Plot reconstruction error per feature.
    
    Parameters
    ----------
    x_original : torch.Tensor or np.ndarray
        Original features
    x_reconstructed : torch.Tensor or np.ndarray
        Reconstructed features
    mask : torch.Tensor or np.ndarray
        Particle mask
    feature_names : list
        Feature names
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.is_tensor(x_original):
        x_original = x_original.cpu().numpy()
    if torch.is_tensor(x_reconstructed):
        x_reconstructed = x_reconstructed.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Calculate MSE per feature
    mse_per_feature = []
    for idx in range(len(feature_names)):
        orig_flat = x_original[:, :, idx][mask.astype(bool)]
        reco_flat = x_reconstructed[:, :, idx][mask.astype(bool)]
        mse = np.mean((orig_flat - reco_flat) ** 2)
        mse_per_feature.append(mse)
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), mse_per_feature, edgecolor='black', alpha=0.7)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Error per Feature')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'), dpi=150)
    print(f"Saved reconstruction error to {output_dir}/reconstruction_error.png")
    plt.close()
    
    # Print MSE values
    print("\nReconstruction MSE per feature:")
    for name, mse in zip(feature_names, mse_per_feature):
        print(f"  {name:15s}: {mse:.6f}")


def plot_2d_correlations(
    x_original,
    x_reconstructed,
    mask,
    feature_names,
    output_dir='plots',
):
    """Plot 2D scatter plots of original vs reconstructed for each feature.
    
    Parameters
    ----------
    x_original : torch.Tensor or np.ndarray
        Original features
    x_reconstructed : torch.Tensor or np.ndarray
        Reconstructed features
    mask : torch.Tensor or np.ndarray
        Particle mask
    feature_names : list
        Feature names
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.is_tensor(x_original):
        x_original = x_original.cpu().numpy()
    if torch.is_tensor(x_reconstructed):
        x_reconstructed = x_reconstructed.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(feature_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get feature data
        orig_flat = x_original[:, :, idx][mask.astype(bool)]
        reco_flat = x_reconstructed[:, :, idx][mask.astype(bool)]
        
        # Subsample for plotting (too many points)
        n_samples = min(10000, len(orig_flat))
        indices = np.random.choice(len(orig_flat), n_samples, replace=False)
        orig_sample = orig_flat[indices]
        reco_sample = reco_flat[indices]
        
        # 2D histogram
        ax.hexbin(orig_sample, reco_sample, gridsize=50, cmap='Blues', mincnt=1)
        
        # Add diagonal line (perfect reconstruction)
        lims = [
            min(orig_sample.min(), reco_sample.min()),
            max(orig_sample.max(), reco_sample.max()),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect reconstruction')
        
        ax.set_xlabel(f'Original {feat_name}')
        ax.set_ylabel(f'Reconstructed {feat_name}')
        ax.set_title(feat_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_plots.png'), dpi=150)
    print(f"Saved correlation plots to {output_dir}/correlation_plots.png")
    plt.close()


def main():
    # ============================================================
    # Configuration
    # ============================================================
    checkpoint_path = 'checkpoints/vqvae_lhco_h5_final.pt'
    data_dir = "/.automount/net_rw/net__data_ttk/hreyes/LHCO/processed_jg/original"
    output_dir = 'plots/reconstruction_h5'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # Load Model
    # ============================================================
    print("\n" + "="*60)
    print("Loading trained model...")
    print("="*60)
    model, config = load_trained_model(checkpoint_path, device=device)
    
    feature_dict = config['input_features_dict']
    feature_names = list(feature_dict.keys())
    max_sequence_len = config['max_sequence_len']
    
    # ============================================================
    # Load Test Data
    # ============================================================
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    # Load a test file (using signal file as example)
    h5_file = os.path.join(data_dir, "sn_N100.h5")
    
    features, labels = load_lhco_jets_from_h5(
        h5_filename=h5_file,
        feature_dict=feature_dict,
        n_jets=5000,  # Load 5000 jets for plotting
        jet_name="jet1",
        mom4_format="epxpypz",
    )
    
    print(f"Loaded {len(features)} jets")
    
    # ============================================================
    # Prepare Data
    # ============================================================
    print("\n" + "="*60)
    print("Preparing data...")
    print("="*60)
    
    # Pad features
    features_padded, mask = ak_pad(
        features, maxlen=max_sequence_len, axis=1, fill_value=0.0, return_mask=True
    )
    
    # Stack features into array
    features_stacked = ak.concatenate(
        [features_padded[feat][..., np.newaxis] for feat in feature_names],
        axis=-1
    )
    
    # Convert to tensors
    x_original = torch.from_numpy(ak.to_numpy(features_stacked)).float()
    mask_tensor = torch.from_numpy(ak.to_numpy(mask)).float()
    
    print(f"Data shape: {x_original.shape}")
    print(f"Mask shape: {mask_tensor.shape}")
    
    # ============================================================
    # Reconstruct
    # ============================================================
    print("\n" + "="*60)
    print("Reconstructing jets...")
    print("="*60)
    
    batch_size = 512
    n_jets = x_original.shape[0]
    x_reco_list = []
    
    for i in range(0, n_jets, batch_size):
        end_idx = min(i + batch_size, n_jets)
        x_batch = x_original[i:end_idx]
        mask_batch = mask_tensor[i:end_idx]
        
        x_reco_batch, vq_out = reconstruct_jets(model, x_batch, mask_batch, device=device)
        x_reco_list.append(x_reco_batch)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {end_idx}/{n_jets} jets")
    
    x_reconstructed = torch.cat(x_reco_list, dim=0)
    print(f"Reconstruction complete: {x_reconstructed.shape}")
    
    # ============================================================
    # Plot
    # ============================================================
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Plot feature distributions
    plot_feature_distributions(
        x_original,
        x_reconstructed,
        mask_tensor,
        feature_names,
        feature_dict,
        output_dir=output_dir,
    )
    
    # Plot particle multiplicity
    plot_particle_multiplicity(mask_tensor, output_dir=output_dir)
    
    # Plot reconstruction error
    plot_reconstruction_error(
        x_original,
        x_reconstructed,
        mask_tensor,
        feature_names,
        output_dir=output_dir,
    )
    
    # Plot 2D correlations
    plot_2d_correlations(
        x_original,
        x_reconstructed,
        mask_tensor,
        feature_names,
        output_dir=output_dir,
    )
    
    # ============================================================
    # Summary Statistics
    # ============================================================
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # Overall reconstruction error
    mask_bool = mask_tensor.cpu().numpy().astype(bool)
    overall_mse = np.mean(
        (x_original.cpu().numpy()[mask_bool] - x_reconstructed.cpu().numpy()[mask_bool]) ** 2
    )
    print(f"\nOverall MSE: {overall_mse:.6f}")
    
    # Unique codes used
    print("\nRunning final batch to check codebook usage...")
    with torch.no_grad():
        x_sample = x_original[:512].to(device)
        mask_sample = mask_tensor[:512].to(device)
        _, vq_out = model(x_sample, mask_sample)
        unique_codes = torch.unique(vq_out['q']).cpu().numpy()
        print(f"Unique codes used: {len(unique_codes)}/{config['vq_kwargs']['num_codes']}")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
