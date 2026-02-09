# imports
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dataloader imports
from gabbro.data.iterable_dataset_jetclass import IterableDatamodule, CustomIterableDataset

# Import the VQVAETransformer class
from gabbro.models.vqvae import VQVAETransformer
from gabbro.utils.optimizer.ranger import Ranger

def data_loader(dataset_kwargs_train, dataset_kwargs_val, dataset_kwargs_test, dataset_kwargs_common, batch_size):
    """Create dataloaders using IterableDatamodule."""
    datamodule = IterableDatamodule(
        dataset_kwargs_train=dataset_kwargs_train,
        dataset_kwargs_val=dataset_kwargs_val,
        dataset_kwargs_test=dataset_kwargs_test,
        dataset_kwargs_common=dataset_kwargs_common,
        batch_size=batch_size,
    )
    
    # Setup for training
    datamodule.setup(stage="fit")
    
    # Get the dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    return train_loader, val_loader


def main():    
    # ============================================================
    # 1. Configuration
    # ============================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    data_dir = "/.automount/net_rw/net__data_ttk/soshaw/JetClass"

    files_dict = {
        'train': {
            'QCD': [os.path.join(data_dir, 'train_100M/ZJetsToNuNu_*.root')],
            'Hbb': [os.path.join(data_dir, 'train_100M/HToBB_*.root')],
            'Hcc': [os.path.join(data_dir, 'train_100M/HToCC_*.root')],
            'Hgg': [os.path.join(data_dir, 'train_100M/HToGG_*.root')],
            'H4q': [os.path.join(data_dir, 'train_100M/HToWW4Q_*.root')],
            'Hqql': [os.path.join(data_dir, 'train_100M/HToWW2Q1L_*.root')],
            'Zqq': [os.path.join(data_dir, 'train_100M/ZToQQ_*.root')],
            'Wqq': [os.path.join(data_dir, 'train_100M/WToQQ_*.root')],
            'Tbqq': [os.path.join(data_dir, 'train_100M/TTBar_*.root')],
            'Tbl': [os.path.join(data_dir, 'train_100M/TTBarLep_*.root')],
        },
        'val': {
            'QCD': [os.path.join(data_dir, 'val_5M/ZJetsToNuNu_*.root')],
            'Hbb': [os.path.join(data_dir, 'val_5M/HToBB_*.root')],
            'Hcc': [os.path.join(data_dir, 'val_5M/HToCC_*.root')],
            'Hgg': [os.path.join(data_dir, 'val_5M/HToGG_*.root')],
            'H4q': [os.path.join(data_dir, 'val_5M/HToWW4Q_*.root')],
            'Hqql': [os.path.join(data_dir, 'val_5M/HToWW2Q1L_*.root')],
            'Zqq': [os.path.join(data_dir, 'val_5M/ZToQQ_*.root')],
            'Wqq': [os.path.join(data_dir, 'val_5M/WToQQ_*.root')],
            'Tbqq': [os.path.join(data_dir, 'val_5M/TTBar_*.root')],
            'Tbl': [os.path.join(data_dir, 'val_5M/TTBarLep_*.root')],
        },
        'test': {
            'QCD': [os.path.join(data_dir, 'test_20M/ZJetsToNuNu_*.root')],
            'Hbb': [os.path.join(data_dir, 'test_20M/HToBB_*.root')],
            'Hcc': [os.path.join(data_dir, 'test_20M/HToCC_*.root')],
            'Hgg': [os.path.join(data_dir, 'test_20M/HToGG_*.root')],
            'H4q': [os.path.join(data_dir, 'test_20M/HToWW4Q_*.root')],
            'Hqql': [os.path.join(data_dir, 'test_20M/HToWW2Q1L_*.root')],
            'Zqq': [os.path.join(data_dir, 'test_20M/ZToQQ_*.root')],
            'Wqq': [os.path.join(data_dir, 'test_20M/WToQQ_*.root')],
            'Tbqq': [os.path.join(data_dir, 'test_20M/TTBar_*.root')],
            'Tbl': [os.path.join(data_dir, 'test_20M/TTBarLep_*.root')],
        }
    }

    labels_to_load = [
        'label_QCD',
        'label_Hbb',
        'label_Hcc',
        'label_Hgg',
        'label_H4q',
        'label_Hqql',
        'label_Zqq',
        'label_Wqq',
        'label_Tbqq',
        'label_Tbl',
    ]

    # Input features configuration (converted from YAML)
    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3},
        "part_mass": {"clip_min_input_space": 0},
        "part_charge": None,
        "part_isChargedHadron": None,
        "part_isNeutralHadron": None,
        "part_isPhoton": None,
        "part_isElectron": None,
        "part_isMuon": None,
        "part_d0val": {"func": "signed_log", "inv_func": "signed_exp"},
        "part_dzval": {"func": "signed_log", "inv_func": "signed_exp"},
        "part_d0err": None,
        "part_dzerr": None,
    }


    # Batch sizes (not used, overwritten in iter_dataset_jetclass)
    # batch_size = {
    #     'train': 500,
    #     'val': 2000,
    #     'test': 2000
    # }

    batch_size = 512
    # num_epochs = int(1000000 / batch_size)
    num_epochs = 1

    # Dataset kwargs - common parameters
    dataset_kwargs_common = {
        'shuffle_particles': True,
        'n_files_at_once': 10,
        'load_only_once': False,
        'pad_length': 128,
        'feature_dict': input_features_dict,
        'feature_dict_jet': None,
        'pad_fill_value': 0.0,
        'collate': False,
        'labels_to_load': labels_to_load,
    }

    # Dataset kwargs - training specific
    dataset_kwargs_train = {
        'n_files_at_once': 10,
        'load_only_once': False,
        'n_jets_per_file': 100_000,
        'max_n_files_per_type': None,
        'n_jets_per_file': None,
        'files_dict': files_dict['train']
    }

    # Dataset kwargs - validation specific
    dataset_kwargs_val = {
        'n_files_at_once': 10,
        'load_only_once': True,
        'shuffle_only_once': True,
        'n_jets_per_file': 10_000,
        'max_n_files_per_type': 1,
        'files_dict': files_dict['val']
    }

    # Dataset kwargs - test specific
    dataset_kwargs_test = {
        'n_files_at_once': 10,
        'load_only_once': True,
        'shuffle_only_once': True,
        'n_jets_per_file': 20_000,
        'max_n_files_per_type': 1,
        'files_dict': files_dict['test']
    }
    
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
    # Training hyperparameters
    # input_dim = len(input_features_dict)  # 15 features
    # batch_size = 32
    # num_epochs = 50

    train_loader, val_loader = data_loader(
        dataset_kwargs_train=dataset_kwargs_train,
        dataset_kwargs_val=dataset_kwargs_val,
        dataset_kwargs_test=dataset_kwargs_test,
        dataset_kwargs_common=dataset_kwargs_common,
        batch_size=batch_size
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
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1,
        total_iters=1
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
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data from batch dictionary
            x = batch['part_features'].to(device)      # (batch_size, 128, 15)
            mask = batch['part_mask'].to(device)       # (batch_size, 128)
            
            # Optional: extract jet-level features if using conditioning
            x_jet = batch.get('jet_features', None)
            if x_jet is not None:
                x_jet = x_jet.to(device)
            
            # Forward pass
            x_reco, vq_out = model(
                x=x, 
                mask=mask, 
                x_conditional=x_jet if model.conditional_dim > 0 else None
            )
            
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
            
            # Print progress every N batches
            if (batch_idx + 1) % 100 == 0:
                len_train = None
                try:
                    len_train = len(train_loader)
                except TypeError:
                    pass
                if len_train:
                    print(f"  Batch {batch_idx+1}/{len_train} - "
                          f"Loss: {loss.item():.4f}, "
                          f"Reco: {reco_loss.item():.4f}, "
                          f"VQ: {vq_loss.item():.4f}")
                else:
                    print(f"  Batch {batch_idx+1} - "
                          f"Loss: {loss.item():.4f}, "
                          f"Reco: {reco_loss.item():.4f}, "
                          f"VQ: {vq_loss.item():.4f}")
        
        # Calculate average training losses
        if train_batches > 0:
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
            for batch in val_loader:
                # Extract data from batch dictionary
                x = batch['part_features'].to(device)
                mask = batch['part_mask'].to(device)
                x_jet = batch.get('jet_features', None)
                if x_jet is not None:
                    x_jet = x_jet.to(device)
                
                # Forward pass
                x_reco, vq_out = model(
                    x=x, 
                    mask=mask, 
                    x_conditional=x_jet if model.conditional_dim > 0 else None
                )
                
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
        
        if val_batches > 0:
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
    
    # ============================================================
    # 6. Save Model
    # ============================================================
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
    }, 'checkpoints/vqvae_model.pt')
    
    print("\nModel saved to 'checkpoints/vqvae_model.pt'")
    
    # ============================================================
    # 7. Test Inference
    # ============================================================
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        # Get a sample
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
        reco_error = torch.mean((x_reco * sample_mask.unsqueeze(-1) - sample_x * sample_mask.unsqueeze(-1)) ** 2)
        print(f"Mean Reconstruction error: {reco_error.item()}")

if __name__ == "__main__":
    main()













"""import torch
from torch.utils.data import DataLoader
from pathlib import Path

from gabbro.data.iterable_dataset_jetclass import IterableDatamodule, CustomIterableDataset

# Define the feature dictionary (from feature_dict_all_without_cuts.yaml)
feature_dict = {
    'part_pt': {'multiply_by': 1, 'subtract_by': 1.8, 'func': "signed_log", 'inv_func': "signed_exp"},
    'part_etarel': {'multiply_by': 3},
    'part_phirel': {'multiply_by': 3},
    'part_mass': {"clip_min_input_space": 0},
    'part_charge': None,
    'part_isChargedHadron': None,
    'part_isNeutralHadron': None,
    'part_isPhoton': None,
    'part_isElectron': None,
    'part_isMuon': None,
    'part_d0val': {'func': "signed_log", 'inv_func': "signed_exp"},
    'part_dzval': {'func': "signed_log", 'inv_func': "signed_exp"},
    'part_d0err': None,
    'part_dzerr': None,
}

# Data directory
data_dir = "/data/dust/user/birkjosc/datasets/jetclass/JetClass"

# Batch sizes
batch_size = {
    'train': 500,
    'val': 2000,
    'test': 2000
}

# Dataset kwargs - common parameters
dataset_kwargs_common = {
    'shuffle_particles': True,
    'n_files_at_once': 10,
    'load_only_once': False,
    'pad_length': 128,
    'feature_dict': feature_dict,
    'feature_dict_jet': None,
    'pad_fill_value': 0.0,
    'collate': False,
}

# Dataset kwargs - training specific
dataset_kwargs_train = {
    'n_files_at_once': 10,
    'load_only_once': False,
    'n_jets_per_file': 100_000,
}

# Dataset kwargs - validation specific
dataset_kwargs_val = {
    'n_files_at_once': 10,
    'load_only_once': True,
    'shuffle_only_once': True,
    'n_jets_per_file': 10_000,
}

# Dataset kwargs - test specific
dataset_kwargs_test = {
    'n_files_at_once': 10,
    'load_only_once': True,
    'shuffle_only_once': True,
    'n_jets_per_file': 20_000,
}

# Create the data module
datamodule = IterableDatamodule(
    dataset_kwargs_train=dataset_kwargs_train,
    dataset_kwargs_val=dataset_kwargs_val,
    dataset_kwargs_test=dataset_kwargs_test,
    dataset_kwargs_common=dataset_kwargs_common,
    batch_size=batch_size,
)

# Setup for training
datamodule.setup(stage="fit")

# Get the dataloaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Iterate through training data
print("Starting training data iteration...")
for batch_idx, batch in enumerate(train_loader):
    # batch is a dictionary with:
    # - 'part_features': torch.Tensor of shape (500, 128, 15)
    # - 'part_mask': torch.BoolTensor of shape (500, 128)
    # - 'jet_type_labels_one_hot': torch.Tensor
    # - 'jet_type_labels': torch.LongTensor
    
    part_features = batch['part_features']  # (500, 128, 15)
    part_mask = batch['part_mask']          # (500, 128)
    
    print(f"Batch {batch_idx}:")
    print(f"  part_features shape: {part_features.shape}")
    print(f"  part_mask shape: {part_mask.shape}")
    print(f"  Number of real particles per jet (first 5): {part_mask.sum(dim=1)[:5]}")
    
    # Process your batch here
    # ...
    
    if batch_idx >= 2:  # Just show first 3 batches
        break

print("\nStarting validation data iteration...")
for batch_idx, batch in enumerate(val_loader):
    part_features = batch['part_features']  # (2000, 128, 15)
    part_mask = batch['part_mask']          # (2000, 128)
    
    print(f"Validation Batch {batch_idx}:")
    print(f"  part_features shape: {part_features.shape}")
    print(f"  part_mask shape: {part_mask.shape}")
    
    if batch_idx >= 1:  # Just show first 2 validation batches
        break"""