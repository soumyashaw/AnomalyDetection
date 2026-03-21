"""Visualize jet embeddings using t-SNE and UMAP."""
import os
import torch
import argparse
import numpy as np
import pickle
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from torch.utils.data import DataLoader

from gabbro.models.backbone import (
    BackboneClassificationLightning,
    BackboneDijetClassificationLightning,
    BackboneAachenClassificationLightning,
)
from gabbro.data.data_utils import create_lhco_h5_test_loader

load_dotenv()


class DataCache:
    """Cache loaded HDF5 data to avoid repeated disk I/O."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Generate unique cache key based on data configuration."""
        config_str = f"{h5_files}_{n_jets}_{feature_dict}_{max_sequence_len}_{model_type}"
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()
    
    def get_cache_path(self, h5_files, n_jets, feature_dict, max_sequence_len, model_type="single"):
        """Generate cache filename."""
        cache_key = self._get_cache_key(h5_files, n_jets, feature_dict, max_sequence_len, model_type)
        file_str = "_".join([Path(f).stem for f in h5_files])
        n_jets_str = "_".join(map(str, n_jets))
        type_str = "dijet" if model_type in ["dijet", "aachen"] else "single"
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
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"Cache file size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")


def extract_data_from_loader(dataloader, model_type="single"):
    """Extract all data from a DataLoader into tensors for caching."""
    all_features = []
    all_features_jet2 = []
    all_masks = []
    all_masks_jet2 = []
    all_labels = []
    
    print("Extracting data from loader...")
    for batch_idx, batch in enumerate(dataloader):
        all_features.append(batch["part_features"])
        all_masks.append(batch["part_mask"])
        all_labels.append(batch["jet_type_labels"])
        
        if model_type in ["dijet", "aachen"] and "part_features_jet2" in batch:
            all_features_jet2.append(batch["part_features_jet2"])
            all_masks_jet2.append(batch["part_mask_jet2"])
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  {batch_idx + 1}/{len(dataloader)} batches")
    
    result = {
        "features": torch.cat(all_features, dim=0),
        "masks": torch.cat(all_masks, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    
    if model_type in ["dijet", "aachen"] and all_features_jet2:
        result["features_jet2"] = torch.cat(all_features_jet2, dim=0)
        result["masks_jet2"] = torch.cat(all_masks_jet2, dim=0)
    
    return result


def create_loader_from_cached_data(cached_data, batch_size, model_type="single"):
    """Create a DataLoader from cached tensor data."""
    if model_type in ["dijet", "aachen"]:
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
        num_workers=0,
        pin_memory=True,
    )


class EmbeddingExtractor:
    """Extract and visualize embeddings from trained model."""
    
    def __init__(self, checkpoint_path, gpu_id, model_type="single", output_dir="plots"):
        """Initialize extractor."""
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading {model_type} model...")
        if self.model_type == "dijet":
            self.model = BackboneDijetClassificationLightning.load_from_checkpoint(checkpoint_path, map_location='cpu')
        elif self.model_type == "aachen":
            self.model = BackboneAachenClassificationLightning.load_from_checkpoint(checkpoint_path, map_location='cpu')
        else:
            self.model = BackboneClassificationLightning.load_from_checkpoint(checkpoint_path, map_location='cpu')
        
        self.model.eval()
        
        # Setup device
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if gpu_id >= n_gpus:
                gpu_id = 0
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def extract_embeddings(self, dataloader):
        """Extract embeddings from model backbone."""
        all_embeddings = []
        all_labels = []
        
        print("Extracting embeddings...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                labels = batch["jet_type_labels"]
                
                if self.model_type in ["dijet", "aachen"]:
                    X1 = batch["part_features"].to(self.device)
                    X2 = batch["part_features_jet2"].to(self.device)
                    mask1 = batch["part_mask"].to(self.device)
                    mask2 = batch["part_mask_jet2"].to(self.device)
                    
                    emb1 = self.model.backbone(X1, mask1)
                    emb2 = self.model.backbone(X2, mask2)
                    
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
                    X = batch["part_features"].to(self.device)
                    mask = batch["part_mask"].to(self.device)
                    
                    emb = self.model.backbone(X, mask)
                    mask_bool = mask.bool()
                    emb_masked = emb * mask_bool.unsqueeze(-1)
                    emb_sum = emb_masked.sum(dim=1)
                    valid_count = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
                    embeddings = emb_sum / valid_count
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Extracted {batch_idx + 1}/{len(dataloader)} batches")
        
        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        
        print(f"Extracted {len(embeddings_np)} embeddings with shape {embeddings_np.shape}")
        return embeddings_np, labels_np
    
    def plot_tsne(self, embeddings, labels, perplexity=30):
        """Plot t-SNE visualization."""
        print("Computing t-SNE...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, 
                    random_state=42, n_jobs=-1, verbose=1)
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
        
        save_path = self.output_dir / 'embeddings_tsne.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_umap(self, embeddings, labels, n_neighbors=15, min_dist=0.1):
        """Plot UMAP visualization."""
        if not HAS_UMAP:
            print("Skipping UMAP (not installed)")
            return
        
        print("Computing UMAP...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42, n_jobs=-1)
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
        
        save_path = self.output_dir / 'embeddings_umap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def visualize(self, test_loader):
        """Extract and visualize embeddings."""
        print("\nExtracting embeddings...")
        embeddings, labels = self.extract_embeddings(test_loader)
        
        print("\nGenerating plots...")
        self.plot_tsne(embeddings, labels)
        self.plot_umap(embeddings, labels)
        
        print(f"\nPlots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Jet Embeddings with t-SNE and UMAP")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["single", "dijet", "aachen"],
                       help="Model architecture type")
    parser.add_argument("--dataset_path", type=str, 
                       default=os.getenv("DATASET_PATH"),
                       help="Path to LHCO dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--n_jets_test", type=int, nargs='+', default=[10000, 20000],
                       help="Number of jets per class [signal, background]")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID")
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear cache and reload from HDF5")
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dir = Path(".cache/evaluation")
        if cache_dir.exists():
            import shutil
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Input features
    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }
    
    # Load test data
    signal_path = os.path.join(args.dataset_path, "sn_50k_SR_test.h5")
    background_path = os.path.join(args.dataset_path, "bg_200k_SR_test.h5")
    h5_files_test = [signal_path, background_path]
    
    # Determine jet_name
    jet_name = "both" if args.model_type in ["dijet", "aachen"] else "jet1"
    
    # Initialize cache
    cache = DataCache(cache_dir=".cache/evaluation")
    
    # Try to load from cache
    print("Checking for cached data...")
    cached_data = cache.load(
        h5_files=h5_files_test,
        n_jets=args.n_jets_test,
        feature_dict=input_features_dict,
        max_sequence_len=128,
        model_type=args.model_type
    )
    
    if cached_data is not None:
        print("Creating DataLoader from cache...")
        test_loader = create_loader_from_cached_data(cached_data, args.batch_size, args.model_type)
    else:
        print(f"Loading from HDF5 (jet_name={jet_name})...")
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
        
        # Extract and cache
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
        
        # Recreate loader from cache
        test_loader = create_loader_from_cached_data(cached_data, args.batch_size, args.model_type)
    
    # Extract and visualize
    extractor = EmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        gpu_id=args.gpu_id,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    extractor.visualize(test_loader)


if __name__ == "__main__":
    main()
