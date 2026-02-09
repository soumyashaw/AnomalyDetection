"""
Training script for VQ-VAE tokenizer on LHCO dataset from HDF5 files.
Adapted to use the H5 data format with jet1/4mom, jet1/mask, and signal labels.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import awkward as ak
import numpy as np

# Import the H5 loader we created
from gabbro.data.loading import load_lhco_jets_from_h5, load_multiple_h5_files

# Import the VQVAETransformer class
from gabbro.models.vqvae import VQVAETransformer
from gabbro.utils.optimizer.ranger import Ranger
from gabbro.utils.arrays import ak_pad


def create_lhco_h5_dataloaders(
    h5_files_train,
    h5_files_val,
    feature_dict,
    batch_size=512,
    n_jets_train=None,
    n_jets_val=None,
    pad_length=128,
    mom4_format="epxpypz",
    train_val_split=None,
):
    """Create PyTorch DataLoaders from LHCO HDF5 files.
    
    Parameters
    ----------
    h5_files_train : list
        List of HDF5 file paths for training
    h5_files_val : list
        List of HDF5 file paths for validation
    feature_dict : dict
        Feature preprocessing dictionary
    batch_size : int
        Batch size for dataloaders
    n_jets_train : int, optional
        Number of jets to load for training (per file)
    n_jets_val : int, optional
        Number of jets to load for validation (per file)
    pad_length : int
        Maximum sequence length (padding)
    mom4_format : str
        4-momentum format in HDF5 files
    train_val_split : float, optional
        If provided (e.g., 0.8), will split the same data into train/val
        with this fraction for training. Ignores h5_files_val in this case.
        
    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    if train_val_split is not None:
        # Load all data and split
        print(f"Loading data from H5 files and splitting {train_val_split:.1%}/{1-train_val_split:.1%} train/val...")
        all_features, all_labels = load_multiple_h5_files(
            h5_files_train,
            feature_dict,
            n_jets_per_file=n_jets_train,
            mom4_format=mom4_format,
        )
        
        # Calculate split index
        n_total = len(all_features)
        n_train = int(n_total * train_val_split)
        
        print(f"Splitting {n_total} jets into {n_train} train / {n_total - n_train} val")
        
        # Split the data
        train_features = all_features[:n_train]
        val_features = all_features[n_train:]
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:]
        
    else:
        # Load training data
        print("Loading training data from H5 files...")
        train_features, train_labels = load_multiple_h5_files(
            h5_files_train,
            feature_dict,
            n_jets_per_file=n_jets_train,
            mom4_format=mom4_format,
        )
        
        # Load validation data
        print("Loading validation data from H5 files...")
        val_features, val_labels = load_multiple_h5_files(
            h5_files_val,
            feature_dict,
            n_jets_per_file=n_jets_val,
            mom4_format=mom4_format,
        )
    
    # Pad and convert to numpy
    print("Padding and converting to tensors...")
    train_features_padded, train_mask = ak_pad(
        train_features, maxlen=pad_length, axis=1, fill_value=0.0, return_mask=True
    )
    val_features_padded, val_mask = ak_pad(
        val_features, maxlen=pad_length, axis=1, fill_value=0.0, return_mask=True
    )
    
    # Stack fields into a single array (n_jets, pad_length, n_features)
    # Get feature names from the dict keys
    feature_names = list(feature_dict.keys())
    
    # Stack training features
    train_features_stacked = ak.concatenate(
        [train_features_padded[feat][..., np.newaxis] for feat in feature_names],
        axis=-1
    )
    # Stack validation features
    val_features_stacked = ak.concatenate(
        [val_features_padded[feat][..., np.newaxis] for feat in feature_names],
        axis=-1
    )
    
    # Convert to numpy then to torch tensors
    train_x = torch.from_numpy(ak.to_numpy(train_features_stacked)).float()
    train_mask_t = torch.from_numpy(ak.to_numpy(train_mask)).float()
    val_x = torch.from_numpy(ak.to_numpy(val_features_stacked)).float()
    val_mask_t = torch.from_numpy(ak.to_numpy(val_mask)).float()
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Validation data shape: {val_x.shape}")
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_x, train_mask_t)
    val_dataset = TensorDataset(val_x, val_mask_t)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader


def main():
    # ============================================================
    # 1. Configuration
    # ============================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model hyperparameters
    alpha = 10  # Weight for VQ loss
    latent_dim = 4
    hidden_dim = 128
    conditional_dim = 0
    num_heads = 8
    num_blocks = 4
    causal_decoder = True
    max_sequence_len = 128
    old_transformer_implementation = False

    # Data configuration
    # Data directory
    data_dir = "/.automount/net_rw/net__data_ttk/hreyes/LHCO/processed_jg/original"
    
    # Background files (label=0 in the file means background)
    h5_files_train = [
        os.path.join(data_dir, "bg_N100.h5"),
    ]
    
    # Signal files (label=1 in the file means signal)
    h5_files_train += [
        os.path.join(data_dir, "sn_N100.h5"),
    ]
    
    # Train/val split ratio (set to None to use separate val files)
    train_val_split = 0.8  # 80% train, 20% val
    
    # If using separate validation files (only used if train_val_split=None)
    h5_files_val = h5_files_train  # Ignored when train_val_split is set
    
    # Feature dictionary - matching LHCO parquet loader
    input_features_dict = {
        "part_pt": {"func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": None,
        "part_phirel": None,
        "part_energy": {"func": "signed_log", "inv_func": "signed_exp"},
        "part_ptrel": None,
        "part_erel": None,
        "part_deltaR": None,
        "part_mass": {"func": "signed_log", "inv_func": "signed_exp"},
    }

    # Training hyperparameters
    batch_size = 512
    num_epochs = 10
    n_jets_train = None  # None = load all jets from files
    n_jets_val = None    # Only used if train_val_split=None
    
    # 4-momentum format in your H5 files
    # Based on the inspection, it looks like [E, px, py, pz] format
    mom4_format = "epxpypz"
    
    # VQ configuration
    vq_kwargs = {
        'num_codes': 512,
        'beta': 0.9,
        'kmeans_init': False,
        'groups': 1,
        'norm': None,
        'cb_norm': None,
        'affine_lr': 2,
        'sync_nu': 1,
        'replace_freq': 100
    }
    
    # ============================================================
    # 2. Create Model
    # ============================================================
    model = VQVAETransformer(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        conditional_dim=conditional_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vq_kwargs=vq_kwargs,
        causal_decoder=causal_decoder,
        max_sequence_len=max_sequence_len,
        input_features_dict=input_features_dict,
        old_transformer_implementation=old_transformer_implementation,
    )
    
    model = model.to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ============================================================
    # 3. Load Data
    # ============================================================
    train_loader, val_loader = create_lhco_h5_dataloaders(
        h5_files_train=h5_files_train,
        h5_files_val=h5_files_val,
        feature_dict=input_features_dict,
        batch_size=batch_size,
        n_jets_train=n_jets_train,
        n_jets_val=n_jets_val,
        pad_length=max_sequence_len,
        mom4_format=mom4_format,
        train_val_split=train_val_split,
    )
    
    # ============================================================
    # 4a. Setup Optimizer
    # ============================================================
    optimizer = Ranger(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-2,
        betas=(0.95, 0.999),
        eps=1e-5,
        alpha=0.5,
        k=6
    )
    
    # ============================================================
    # 4b. Setup Scheduler
    # ============================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # ============================================================
    # 5. Training Loop
    # ============================================================
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_reco_loss = 0.0
        train_vq_loss = 0.0
        train_batches = 0
        
        for batch_idx, (x, mask) in enumerate(train_loader):
            x = x.to(device)
            mask = mask.to(device)
            
            # Forward pass
            x_reco, vq_out = model(x=x, mask=mask)
            
            # Calculate reconstruction loss
            reco_loss = torch.sum(
                (x_reco * mask.unsqueeze(-1) - x * mask.unsqueeze(-1)) ** 2
            ) / torch.sum(mask)
            
            # Get VQ loss (commitment loss)
            vq_loss = vq_out['loss'].mean()
            
            # Total loss
            loss = reco_loss + alpha * vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Optional: gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_reco_loss += reco_loss.item()
            train_vq_loss += vq_loss.item()
            train_batches += 1
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}, "
                      f"Reco: {reco_loss.item():.4f}, "
                      f"VQ: {vq_loss.item():.4f}")
        
        # Calculate average training losses
        train_loss /= train_batches
        train_reco_loss /= train_batches
        train_vq_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_reco_loss = 0.0
        val_vq_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, mask in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                
                # Forward pass
                x_reco, vq_out = model(x=x, mask=mask)
                
                # Calculate losses
                reco_loss = torch.sum(
                    (x_reco * mask.unsqueeze(-1) - x * mask.unsqueeze(-1)) ** 2
                ) / torch.sum(mask)
                
                vq_loss = vq_out['loss'].mean()
                loss = reco_loss + alpha * vq_loss
                
                val_loss += loss.item()
                val_reco_loss += reco_loss.item()
                val_vq_loss += vq_loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_reco_loss /= val_batches
        val_vq_loss /= val_batches
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        print(f"  Train - Loss: {train_loss:.4f}, Reco: {train_reco_loss:.4f}, VQ: {train_vq_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Reco: {val_reco_loss:.4f}, VQ: {val_vq_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/vqvae_lhco_h5_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved to '{checkpoint_path}'")
    
    # ============================================================
    # 6. Save Final Model
    # ============================================================
    final_save_path = 'checkpoints/vqvae_lhco_h5_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'conditional_dim': conditional_dim,
            'num_heads': num_heads,
            'num_blocks': num_blocks,
            'vq_kwargs': vq_kwargs,
            'input_features_dict': input_features_dict,
            'causal_decoder': causal_decoder,
            'max_sequence_len': max_sequence_len,
            'old_transformer_implementation': old_transformer_implementation,
        }
    }, final_save_path)
    
    print(f"\nFinal model saved to '{final_save_path}'")
    
    # ============================================================
    # 7. Test Inference
    # ============================================================
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_x, sample_mask = next(iter(val_loader))
        sample_x = sample_x.to(device)
        sample_mask = sample_mask.to(device)
        
        # Encode
        z_embed, _ = model.encode(sample_x, sample_mask)
        print(f"Encoded shape: {z_embed.shape}")
        
        # Quantize
        z_q, vq_out = model.quantize(z_embed)
        print(f"Quantized shape: {z_q.shape}")
        print(f"Code indices shape: {vq_out['q'].shape}")
        
        # Decode
        x_reco = model.decode(z_q, sample_mask)
        print(f"Reconstructed shape: {x_reco.shape}")
        
        # Calculate reconstruction error
        reco_error = torch.mean((x_reco * sample_mask.unsqueeze(-1) - sample_x * sample_mask.unsqueeze(-1)) ** 2).item()
        print(f"Mean reconstruction error: {reco_error:.6f}")


if __name__ == "__main__":
    main()
