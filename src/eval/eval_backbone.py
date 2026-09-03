"""Visualize jet embeddings using t-SNE and UMAP."""
import os
import torch
import argparse
import numpy as np
import pickle
import hashlib
import logging
from collections.abc import Mapping
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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
from gabbro.models.backbone_base import BackboneTransformer
from gabbro.data.data_utils import create_lhco_h5_test_loader

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


def _to_plain_dict(obj):
    """Convert config-like containers (e.g., DictConfig) to plain dict recursively."""
    if isinstance(obj, Mapping):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(v) for v in obj)
    return obj


class AttrDict(dict):
    """Dict with attribute-style access (obj.key) while keeping dict behavior."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _to_attrdict(obj):
    """Recursively convert mappings to AttrDict for cfg objects like embed_cfg."""
    if isinstance(obj, Mapping):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_attrdict(v) for v in obj)
    return obj


def _complete_backbone_hparams(hparams, state_dict=None):
    """
    Add missing hyperparameters needed for BackboneTransformer with sensible defaults.
    
    This is useful when loading checkpoints from models that only saved minimal hyperparameters
    (e.g., contrastive learning models that only saved embedding_dim and particle_features_dict).
    
    Parameters
    ----------
    hparams : dict
        Existing hyperparameters from checkpoint
    state_dict : dict, optional
        Model state_dict to infer architecture from (e.g., number of blocks)
    
    Returns
    -------
    dict
        Complete hyperparameters with defaults added for missing keys
    """
    hparams = dict(hparams)  # Copy to avoid modifying original
    
    # Infer n_blocks from state_dict if available
    n_blocks = 8  # default
    if state_dict is not None:
        block_keys = [k for k in state_dict.keys() if k.startswith('transformer.blocks.')]
        if block_keys:
            block_indices = [int(k.split('.')[2]) for k in block_keys if k.split('.')[2].isdigit()]
            if block_indices:
                n_blocks = max(block_indices) + 1
    
    # Infer n_registers from state_dict
    n_registers = 0
    if state_dict is not None and 'registers' in state_dict:
        # registers shape is (1, n_registers, embedding_dim)
        n_registers = state_dict['registers'].shape[1]
    
    # Check if model has jet features
    has_jet_features = False
    if state_dict is not None:
        jet_keys = [k for k in state_dict.keys() if 'embed_jet' in k]
        has_jet_features = len(jet_keys) > 0
    
    # Check what's missing and add defaults
    defaults = {
        'max_sequence_len': 128,
        'vocab_size': 8194,
        'n_registers': n_registers,
        'apply_causal_mask': False,  # Most evaluation tasks don't need causal mask
        'embed_cfg': {
            'type': 'continuous_project_add',
            'intermediate_dim': None,
        },
        'transformer_cfg': {
            'dim': hparams.get('embedding_dim', 128),
            'n_blocks': n_blocks,
            'norm_after_blocks': True,
            'residual_cfg': {
                'gate_type': 'local',
                'init_value': 1,
            },
            'attn_cfg': {
                'num_heads': 8,
                'dropout_rate': 0.1,
                'norm_before': True,
                'norm_after': False,
            },
            'mlp_cfg': {
                'dropout_rate': 0.0,
                'norm_before': True,
                'expansion_factor': 4,
                'activation': 'GELU',
            },
        },
        'interaction_cfg': None,
        'jet_features_dict': None,  # Will be set if model has jet features
        'feature_drop_cfg': None,
    }
    
    # Only add defaults for keys that are missing
    for key, default_value in defaults.items():
        if key not in hparams:
            hparams[key] = default_value
    
    return hparams


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
        type_str = "dijet" if model_type in ["dijet", "aachen", "pretrained"] else "single"
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
        
        if model_type in ["dijet", "aachen", "pretrained"] and "part_features_jet2" in batch:
            all_features_jet2.append(batch["part_features_jet2"])
            all_masks_jet2.append(batch["part_mask_jet2"])
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  {batch_idx + 1}/{len(dataloader)} batches")
    
    result = {
        "features": torch.cat(all_features, dim=0),
        "masks": torch.cat(all_masks, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    
    if model_type in ["dijet", "aachen", "pretrained"] and all_features_jet2:
        result["features_jet2"] = torch.cat(all_features_jet2, dim=0)
        result["masks_jet2"] = torch.cat(all_masks_jet2, dim=0)
    
    return result


def create_loader_from_cached_data(cached_data, batch_size, model_type="single"):
    """Create a DataLoader from cached tensor data."""
    if model_type in ["dijet", "aachen", "pretrained"]:
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
        """Initialize extractor.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint
        gpu_id : int
            GPU device ID
        model_type : str
            Model architecture type: 'single', 'dijet', 'aachen', or 'pretrained'
            'pretrained' loads BackboneTransformer directly
        output_dir : str
            Directory to save plots
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading {model_type} model...")
        if self.model_type == "pretrained":
            # Load BackboneTransformer directly from checkpoint (Lightning or extracted)
            print("Loading BackboneTransformer from checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle both Lightning and extracted backbone formats
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                
                # Try to get hyperparameters from multiple possible locations
                hparams = ckpt.get('hyper_parameters', {})
                if not hparams:
                    hparams = ckpt.get('hparams', {})
                if not hparams:
                    hparams = ckpt.get('hparams_dict', {})
                if isinstance(hparams, Mapping):
                    hparams = _to_plain_dict(hparams)
                
                # Handle nested backbone_cfg (in case checkpoint was saved with nested structure)
                if hparams and 'backbone_cfg' in hparams and 'embedding_dim' not in hparams:
                    print("Warning: Detected nested backbone_cfg. Extracting flattened config...")
                    if isinstance(hparams['backbone_cfg'], Mapping):
                        hparams = _to_plain_dict(hparams['backbone_cfg'])
                        print(f"✓ Extracted backbone_cfg with {len(hparams)} parameters")
                
                # Also check model_kwargs nesting
                if hparams and 'model_kwargs' in hparams and 'embedding_dim' not in hparams:
                    model_kwargs = hparams.get('model_kwargs', {})
                    if isinstance(model_kwargs, Mapping) and 'backbone_cfg' in model_kwargs:
                        print("Warning: Detected nested model_kwargs.backbone_cfg. Extracting...")
                        hparams = _to_plain_dict(model_kwargs['backbone_cfg'])
                        print(f"✓ Extracted backbone_cfg with {len(hparams)} parameters")
                
                # Extract backbone state dict (remove 'backbone.' prefix if present)
                backbone_state = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_key = k.replace('backbone.', '', 1)
                        backbone_state[new_key] = v
                    else:
                        # Already a backbone state dict (from extraction)
                        backbone_state[k] = v
                
                if not backbone_state:
                    raise ValueError("Checkpoint has empty state_dict")
                
                # Create BackboneTransformer with hyperparameters
                if not hparams:
                    # Fallback: Load as Lightning module, then extract backbone
                    print("\nWarning: Checkpoint missing hyperparameters.")
                    print("Loading full Lightning module and extracting backbone...")
                    try:
                        # Try loading as single jet model (most common)
                        temp_model = BackboneClassificationLightning.load_from_checkpoint(
                            checkpoint_path, map_location='cpu'
                        )
                        # Get backbone from the loaded model
                        self.model = temp_model.backbone
                        print("✓ Extracted backbone from Lightning module")
                    except Exception as e:
                        # Fallback: Load state_dict directly and try to infer model type
                        print(f"Lightning load failed ({type(e).__name__}), trying direct state_dict load...")
                        try:
                            # Load state_dict directly
                            ckpt_raw = torch.load(checkpoint_path, map_location='cpu')
                            state_dict_raw = ckpt_raw.get('state_dict', ckpt_raw)
                            
                            # Try different Lightning module types
                            model_types_to_try = [
                                ('single', BackboneClassificationLightning),
                                ('dijet', BackboneDijetClassificationLightning),
                                ('aachen', BackboneAachenClassificationLightning),
                            ]
                            
                            loaded = False
                            for model_name, ModelClass in model_types_to_try:
                                try:
                                    # Create model and load state_dict
                                    temp_model = ModelClass()
                                    temp_model.load_state_dict(state_dict_raw, strict=False)
                                    self.model = temp_model.backbone
                                    print(f"✓ Extracted backbone from {model_name} Lightning module")
                                    loaded = True
                                    break
                                except Exception:
                                    continue
                            
                            if not loaded:
                                raise RuntimeError(
                                    "Could not load checkpoint as any Lightning model type "
                                    "(single/dijet/aachen)"
                                )
                        except Exception as e2:
                            logger.error("=" * 80)
                            logger.error("ERROR: Could not load checkpoint!")
                            logger.error("=" * 80)
                            logger.error(f"\nOriginal error: {e}")
                            logger.error(f"Fallback error: {e2}")
                            logger.error("\nTo use this checkpoint, you can:")
                            logger.error("\n  Option 1: Use the full Lightning module (specify correct type)")
                            logger.error(f"     python -m src.eval.eval_backbone \\")
                            logger.error(f"       --checkpoint {checkpoint_path} \\")
                            logger.error(f"       --model_type single  # or dijet/aachen if applicable")
                            logger.error("\n  Option 2: Extract backbone weights first (for transfer learning)")
                            logger.error(f"     python -m src.eval.extract_backbone \\")
                            logger.error(f"       --checkpoint {checkpoint_path} \\")
                            logger.error(f"       --output_name my_backbone \\")
                            logger.error(f"       --verify --metadata")
                            logger.error("\n     Then use:")
                            logger.error(f"     python -m src.eval.eval_backbone \\")
                            logger.error(f"       --checkpoint backbone_weights/my_backbone/backbone.pt \\")
                            logger.error(f"       --model_type pretrained")
                            logger.error("=" * 80 + "\n")
                            raise
                else:
                    print(f"Instantiating BackboneTransformer with {len(hparams)} hyperparameters...")
                    print(f"Hyperparameters keys: {list(hparams.keys())}")
                    
                    # Add missing hyperparameters with sensible defaults
                    # This is needed for checkpoints that only saved minimal hparams (e.g., contrastive models)
                    if 'embed_cfg' not in hparams or 'transformer_cfg' not in hparams:
                        print(f"\n⚠ Warning: Checkpoint missing configuration keys (embed_cfg, transformer_cfg, etc.)")
                        print(f"   This is common for contrastive/pretraining checkpoints.")
                        print(f"   Adding default configuration based on typical architecture...")
                        hparams = _complete_backbone_hparams(hparams, state_dict=backbone_state)
                        print(f"   ✓ Configuration completed. New keys: {list(hparams.keys())}")
                    
                    try:
                        # BackboneTransformer expects cfg objects with attribute access (e.g. embed_cfg.type).
                        hparams_for_model = _to_attrdict(hparams)
                        self.model = BackboneTransformer(**hparams_for_model)
                    except TypeError as e:
                        print(f"\nError: {e}")
                        print(f"\nAvailable hparams ({len(hparams)} total):")
                        for k, v in hparams.items():
                            print(f"  {k}: {type(v).__name__}")
                        
                        # Try to reconstruct hparams if they're nested
                        if isinstance(hparams, dict):
                            # Check if hparams are nested (e.g., from a config object)
                            for k, v in list(hparams.items())[:5]:
                                if isinstance(v, dict):
                                    print(f"\nNote: '{k}' appears to be a nested dict. Checking contents...")
                                    for subk, subv in v.items():
                                        print(f"  {k}.{subk}: {type(subv).__name__}")
                        
                        raise ValueError(
                            f"Missing required hyperparameters for BackboneTransformer. "
                            f"Expected 'embedding_dim' and other parameters, but got: {list(hparams.keys())}"
                        )
                    
                    # Load state_dict with strict=False to handle minor mismatches
                    # (e.g., jet features or registers that may differ from checkpoint)
                    missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state, strict=False)
                    if missing_keys:
                        print(f"⚠ Warning: Missing keys in checkpoint (will use random initialization):")
                        for key in missing_keys:
                            print(f"  - {key}")
                    if unexpected_keys:
                        print(f"⚠ Warning: Unexpected keys in checkpoint (will be ignored):")
                        for key in unexpected_keys:
                            print(f"  - {key}")
                    print(f"✓ Loaded {len(backbone_state)} backbone weight tensors")
            else:
                raise ValueError(
                    f"Checkpoint must contain 'state_dict' key. "
                    f"Found keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
                )
        else:
            # Load Lightning modules
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
                
                if self.model_type == "pretrained":
                    # BackboneTransformer with both jets - called directly (not via .backbone())
                    X1 = batch["part_features"].to(self.device)
                    X2 = batch["part_features_jet2"].to(self.device)
                    mask1 = batch["part_mask"].to(self.device)
                    mask2 = batch["part_mask_jet2"].to(self.device)
                    
                    emb1 = self.model(X1, mask1)
                    emb2 = self.model(X2, mask2)
                    
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
                elif self.model_type in ["dijet", "aachen"]:
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
                    # Single jet (default)
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
        
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, 
                    random_state=42, n_jobs=-1, verbose=1)
        embeddings_tsne = tsne.fit_transform(embeddings_scaled)
        
        plt.figure(figsize=(10, 8))
        bg_mask = labels == 0
        sig_mask = labels == 1
        
        plt.scatter(embeddings_tsne[bg_mask, 0], embeddings_tsne[bg_mask, 1], 
                   c='#D6938A', label='Background', alpha=0.6, s=5)
        plt.scatter(embeddings_tsne[sig_mask, 0], embeddings_tsne[sig_mask, 1], 
                   c='#1b9e77', label='Signal', alpha=0.6, s=5)
        
        ax = plt.gca()
        ax.set_xlabel('t-SNE 1', fontsize=15)
        ax.set_ylabel('t-SNE 2', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
        ax.minorticks_on()
        ax.legend(fontsize=15, framealpha=0.9)
        plt.tight_layout()
        
        save_path_png = self.output_dir / 'embeddings_tsne.png'
        save_path_pdf = self.output_dir / 'embeddings_tsne.pdf'
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
    
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
                   c='#D6938A', label='Background', alpha=0.6, s=5)
        plt.scatter(embeddings_umap[sig_mask, 0], embeddings_umap[sig_mask, 1], 
                   c='#1b9e77', label='Signal', alpha=0.6, s=5)
        
        ax = plt.gca()
        ax.set_xlabel('UMAP 1', fontsize=15)
        ax.set_ylabel('UMAP 2', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
        ax.minorticks_on()
        ax.legend(fontsize=15, framealpha=0.9)
        plt.tight_layout()
        
        save_path_png = self.output_dir / 'embeddings_umap.png'
        save_path_pdf = self.output_dir / 'embeddings_umap.pdf'
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
    
    def visualize(self, test_loader):
        """Extract and visualize embeddings."""
        embeddings, labels = self.extract_embeddings(test_loader)

        # Compute silhouette score on standardized embeddings and save to text file.
        silhouette_path = self.output_dir / 'silhouette_score.txt'
        silhouette_value = None
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
            message = (
                "Silhouette Score: N/A\n"
                "Reason: Need at least 2 clusters and more samples than number of clusters.\n"
            )
        else:
            try:
                embeddings_scaled = StandardScaler().fit_transform(embeddings)
                silhouette_value = silhouette_score(embeddings_scaled, labels)
                message = f"Silhouette Score: {silhouette_value:.6f}\n"
            except Exception as e:
                message = f"Silhouette Score: N/A\nReason: {e}\n"

        with open(silhouette_path, 'w') as f:
            f.write(message)

        if silhouette_value is not None:
            print(f"Saved silhouette score: {silhouette_value:.6f} -> {silhouette_path}")
        else:
            print(f"Saved silhouette score info -> {silhouette_path}")
        
        print("\nGenerating plots...")
        self.plot_tsne(embeddings, labels)
        self.plot_umap(embeddings, labels)
        
        print(f"\nPlots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Jet Embeddings with t-SNE and UMAP")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["single", "dijet", "aachen", "pretrained"],
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
    
    # Determine jet_name (pretrained and dijet/aachen use both jets)
    jet_name = "both" if args.model_type in ["dijet", "aachen", "pretrained"] else "jet1"
    
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
