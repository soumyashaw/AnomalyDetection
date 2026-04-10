#!/usr/bin/env python3
"""
Extract BackboneTransformer weights from Lightning checkpoints for transfer learning.

This script extracts the shared backbone weights from trained Lightning modules and saves them
in a format that can be easily loaded into new models for fine-tuning or transfer learning.

Key Features:
- Extracts backbone weights from any Lightning checkpoint with a backbone attribute
- Validates weight integrity and structure
- Creates metadata for reproducibility
- Provides utilities to load weights with or without strict mode
- Handles both old-style (module.) and new-style (backbone.) prefixes

Usage:
    # Extract and save backbone weights
    python extract_backbone.py \\
        --checkpoint path/to/best.ckpt \\
        --output_dir backbone_weights \\
        --verify \\
        --metadata \\
        --verbose

    # Load weights into a new model
    from extract_backbone_weights import load_backbone_weights
    load_backbone_weights(new_model, 'backbone_weights.pt', strict=True)
"""

import argparse
import json
import logging
from pathlib import Path
from collections.abc import Mapping
from typing import Dict, Optional, Tuple
import yaml

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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


class BackboneWeightExtractor:
    """Extract and manage BackboneTransformer weights from Lightning checkpoints."""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cpu'):
        """
        Initialize the weight extractor.
        
        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt)
            device: Device to load checkpoint on ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint has invalid format
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=device)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")
        
        self.state_dict = self.checkpoint.get('state_dict', {})
        if not self.state_dict:
            self.state_dict = self.checkpoint  # Handle raw state_dict saves
        
        # Extract hyperparameters for model reconstruction
        self.hparams = self._extract_hparams()
        
        # Verify extraction was successful
        if not self.hparams:
            logger.error("Failed to extract hyperparameters. This will cause issues later.")
        elif 'embedding_dim' not in self.hparams:
            logger.warning(
                f"Extracted hparams but missing 'embedding_dim'. "
                f"Keys: {list(self.hparams.keys())[:10]}. "
                f"This indicates incomplete extraction from nested structure."
            )
    
    def _extract_hparams(self) -> dict:
        """
        Extract hyperparameters needed to reconstruct BackboneTransformer.
        Handles both flat and nested hyperparameter structures.
        
        Tries multiple locations:
        - hyper_parameters (extracted format)
        - hparams (Lightning format)
        - hparams_name (some Lightning versions)
        - hparams_dict (other Lightning versions)
        
        Also handles nested configs:
        - backbone_cfg (nested BackboneTransformer config)
        - model_kwargs (another common nesting location)
        
        Returns:
            Dictionary with hyperparameters for BackboneTransformer
        """
        hparams = {}
        
        # Try multiple possible locations in checkpoint
        hparam_keys = ['hyper_parameters', 'hparams', 'hparams_name', 'hparams_dict']
        
        for key in hparam_keys:
            if key in self.checkpoint:
                candidate = self.checkpoint[key]
                # Handle different types
                if isinstance(candidate, Mapping):
                    hparams = _to_plain_dict(candidate)
                    logger.info(f"Found hyperparameters under key '{key}'")
                    break
        
        if not hparams:
            logger.warning(
                "No hyperparameters found in checkpoint. "
                "Tried keys: " + ", ".join(hparam_keys) + ". "
                "Available checkpoint keys: " + ", ".join(list(self.checkpoint.keys())[:10])
            )
            return {}
        
        # Check if backbone config is nested under 'backbone_cfg'
        if 'backbone_cfg' in hparams and isinstance(hparams['backbone_cfg'], Mapping):
            logger.info("Detected nested BackboneTransformer config under 'backbone_cfg'")
            backbone_hparams = _to_plain_dict(hparams['backbone_cfg'])
            logger.info(f"✓ Extracted {len(backbone_hparams)} hyperparameters from backbone_cfg")
            return backbone_hparams
        
        # Check if backbone config is under 'model_kwargs'
        if 'model_kwargs' in hparams and isinstance(hparams['model_kwargs'], Mapping):
            logger.info("Detected nested BackboneTransformer config under 'model_kwargs'")
            model_kwargs = hparams['model_kwargs']
            # Check if it has backbone config
            if 'backbone_cfg' in model_kwargs and isinstance(model_kwargs['backbone_cfg'], Mapping):
                backbone_hparams = _to_plain_dict(model_kwargs['backbone_cfg'])
                logger.info(f"✓ Extracted {len(backbone_hparams)} hyperparameters from model_kwargs.backbone_cfg")
                return backbone_hparams
        
        # Strategy 1: Try to extract only backbone-related keys from flat structure
        required_keys = ['embedding_dim']
        backbone_keys = [
            'embedding_dim', 'apply_causal_mask', 'max_sequence_len', 'vocab_size',
            'n_registers', 'embed_cfg', 'transformer_cfg', 'interaction_cfg',
            'particle_features_dict', 'jet_features_dict', 'feature_drop_cfg'
        ]
        
        backbone_hparams = {}
        for key in backbone_keys:
            if key in hparams:
                backbone_hparams[key] = hparams[key]
        
        # Strategy 2: If we're missing embedding_dim (required), keep ALL hparams
        if 'embedding_dim' not in backbone_hparams:
            logger.warning(
                f"Missing 'embedding_dim' in filtered hyperparameters. "
                f"Found backbone keys: {list(backbone_hparams.keys())}. "
                f"Nested config keys in hparams: {[k for k in hparams.keys() if isinstance(hparams[k], dict)]}. "
                f"Unable to locate proper BackboneTransformer configuration."
            )
            backbone_hparams = hparams
        else:
            logger.info(f"✓ Extracted {len(backbone_hparams)} BackboneTransformer hyperparameters")
        
        return backbone_hparams
        
    def extract_backbone_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract BackboneTransformer weights from checkpoint state_dict.
        
        The function handles multiple naming conventions:
        - backbone.* - Modern Lightning module naming
        - module.backbone.* - Distributed training naming
        - Standalone BackboneTransformer weights
        
        Returns:
            Dictionary of backbone weights with prefixes removed
            
        Raises:
            ValueError: If no backbone weights found
        """
        if not self.state_dict:
            raise ValueError("Checkpoint has no 'state_dict' key and is not a raw state_dict")
        
        backbone_weights = {}
        
        # Strategy 1: Look for backbone.* prefixed keys (modern Lightning)
        for key, value in self.state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '', 1)
                backbone_weights[new_key] = value
        
        # Strategy 2: Look for module.backbone.* keys (distributed training)
        if not backbone_weights:
            for key, value in self.state_dict.items():
                if 'module.backbone.' in key:
                    # Remove 'module.backbone.' prefix
                    new_key = key.split('module.backbone.')[1]
                    backbone_weights[new_key] = value
        
        # Strategy 3: Check if this is a standalone backbone state_dict
        # (has transformer, embedding layers, but no other heads)
        if not backbone_weights:
            # Check for typical BackboneTransformer keys
            transformer_keys = [k for k in self.state_dict.keys() 
                              if 'transformer' in k or 'embed' in k or 'input_projection' in k]
            if len(transformer_keys) > 0:
                logger.warning(
                    "Detected standalone BackboneTransformer state_dict. "
                    "Using all weights as backbone weights."
                )
                backbone_weights = dict(self.state_dict)
        
        if not backbone_weights:
            self._log_available_keys()
            raise ValueError(
                "No backbone weights found in checkpoint. "
                "Expected keys starting with 'backbone.', 'module.backbone.', "
                "or standalone backbone state_dict."
            )
        
        logger.info(f"✓ Extracted {len(backbone_weights)} backbone weight tensors")
        return backbone_weights
    
    def _log_available_keys(self) -> None:
        """Log available keys in the checkpoint for debugging."""
        logger.error("Available keys in checkpoint:")
        for i, key in enumerate(sorted(self.state_dict.keys())[:20]):
            logger.error(f"  {key}")
        if len(self.state_dict) > 20:
            logger.error(f"  ... and {len(self.state_dict) - 20} more keys")
    
    def validate_backbone_weights(self, weights: Dict[str, torch.Tensor]) -> bool:
        """
        Validate that extracted weights represent a valid BackboneTransformer.
        
        Checks for:
        - Transformer blocks (attention, MLP)
        - Embedding layers
        - Layer normalization
        
        Args:
            weights: Dictionary of backbone weights
            
        Returns:
            True if structure looks valid
        """
        if not weights:
            logger.error("Empty weights dictionary")
            return False
        
        # Look for key components
        key_patterns = {
            'transformer': 'transformer encoder blocks',
            'embed': 'embedding layers',
            'norm': 'layer normalization',
            'attn': 'attention layers',
            'mlp': 'feed-forward network',
        }
        
        found_patterns = {}
        for pattern, description in key_patterns.items():
            matching_keys = [k for k in weights.keys() if pattern in k.lower()]
            found_patterns[description] = len(matching_keys)
        
        logger.info("Backbone structure validation:")
        for description, count in found_patterns.items():
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} {description}: {count} components")
        
        # Require at minimum transformer blocks or (attention + mlp)
        has_transformer = found_patterns.get('transformer encoder blocks', 0) > 0
        has_attention = found_patterns.get('attention layers', 0) > 0
        has_mlp = found_patterns.get('feed-forward network', 0) > 0
        
        is_valid = has_transformer or (has_attention and has_mlp)
        
        if is_valid:
            logger.info("✓ Backbone structure validation passed")
        else:
            logger.warning("⚠ Backbone structure validation failed - no transformer components found")
        
        return is_valid
    
    def get_weight_statistics(self, weights: Dict[str, torch.Tensor]) -> Dict:
        """
        Calculate statistics about the extracted weights.
        
        Args:
            weights: Dictionary of backbone weights
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_tensors': len(weights),
            'total_parameters': sum(t.numel() for t in weights.values()),
            'parameter_breakdown': {},
            'dtype_breakdown': {},
            'size_mb': sum(t.element_size() * t.numel() for t in weights.values()) / (1024 * 1024),
        }
        
        # Parameter breakdown by layer
        for key, tensor in weights.items():
            layer_name = key.split('.')[0]
            num_params = tensor.numel()
            if layer_name not in stats['parameter_breakdown']:
                stats['parameter_breakdown'][layer_name] = 0
            stats['parameter_breakdown'][layer_name] += num_params
        
        # Data type breakdown
        for key, tensor in weights.items():
            dtype = str(tensor.dtype)
            if dtype not in stats['dtype_breakdown']:
                stats['dtype_breakdown'][dtype] = 0
            stats['dtype_breakdown'][dtype] += tensor.numel()
        
        return stats
    
    def get_weight_shapes(self, weights: Dict[str, torch.Tensor]) -> None:
        """Log shape information for all backbone weights."""
        logger.info("Backbone weight shapes:")
        total_params = 0
        for key, tensor in sorted(weights.items()):
            num_params = tensor.numel()
            total_params += num_params
            logger.info(f"  {key}: {tuple(tensor.shape)} ({num_params:,} params)")
        
        logger.info(f"Total backbone parameters: {total_params:,}")
    
    def save_weights(self, weights: Dict[str, torch.Tensor], output_path: Path) -> Path:
        """
        Save backbone weights AND hyperparameters to disk.
        
        Creates a minimal checkpoint with only what's needed for transfer learning:
        - Backbone state_dict (without head weights)
        - Hyperparameters to reconstruct BackboneTransformer
        - Minimal metadata
        
        Args:
            weights: Dictionary of backbone weights
            output_path: Path to save weights (used as base for both formats)
            
        Returns:
            Path to saved .pt weights file
            
        Raises:
            IOError: If save fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine base path and save both formats
        if output_path.suffix == '.pt':
            pt_path = output_path
            ckpt_path = output_path.with_suffix('.ckpt')
        elif output_path.suffix == '.ckpt':
            ckpt_path = output_path
            pt_path = output_path.with_suffix('.pt')
        else:
            pt_path = output_path.with_suffix('.pt')
            ckpt_path = output_path.with_suffix('.ckpt')
        
        # Create minimal checkpoint with backbone weights and hparams only
        checkpoint_data = {
            'state_dict': weights,
            'hyper_parameters': self.hparams,
            'backbone_only': True,  # Flag to indicate this is extracted backbone
            'source_checkpoint': str(self.checkpoint_path.absolute()),
        }
        
        # Log what we're saving
        logger.info(f"Saving checkpoint with:")
        logger.info(f"  - state_dict: {len(weights)} weight tensors")
        logger.info(f"  - hyper_parameters: {len(self.hparams)} parameters")
        logger.info(f"    Keys: {list(self.hparams.keys())[:10]}{'...' if len(self.hparams) > 10 else ''}")
        if 'embedding_dim' in self.hparams:
            logger.info(f"    ✓ embedding_dim={self.hparams.get('embedding_dim')}")
        else:
            logger.warning(f"    ⚠ Missing 'embedding_dim' - this will cause loading errors!")
        
        try:
            # Save as .pt (PyTorch checkpoint format with minimal info)
            torch.save(checkpoint_data, pt_path)
            logger.info(f"✓ Saved backbone weights + hparams to: {pt_path}")
            
            # Save as .ckpt (same content, different extension for compatibility)
            torch.save(checkpoint_data, ckpt_path)
            logger.info(f"✓ Saved backbone weights + hparams to: {ckpt_path}")
        except Exception as e:
            raise IOError(f"Failed to save weights: {e}")
        
        # Verify both saves
        try:
            assert pt_path.exists(), f"Weight file not created at {pt_path}"
            assert ckpt_path.exists(), f"Weight file not created at {ckpt_path}"
            
            loaded_pt = torch.load(pt_path, map_location='cpu')
            loaded_ckpt = torch.load(ckpt_path, map_location='cpu')
            
            assert 'state_dict' in loaded_pt, "Saved file missing 'state_dict'"
            assert 'hyper_parameters' in loaded_pt, "Saved file missing 'hyper_parameters'"
            assert len(loaded_pt['state_dict']) == len(weights), (
                f"PT weights count mismatch: {len(loaded_pt['state_dict'])} vs {len(weights)}"
            )
            assert len(loaded_ckpt['state_dict']) == len(weights), (
                f"CKPT weights count mismatch: {len(loaded_ckpt['state_dict'])} vs {len(weights)}"
            )
            
            size_mb = pt_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Verified saved file ({size_mb:.2f} MB)")
            logger.info(f"  - Backbone weights: {len(weights)} tensors")
            logger.info(f"  - Hyperparameters: {len(self.hparams)} params")
        except AssertionError as e:
            raise IOError(f"Verification failed: {e}")
        
        return pt_path
    
    def save_metadata(self, weights: Dict[str, torch.Tensor], 
                     output_path: Path, metadata_path: Path) -> Path:
        """
        Save extraction metadata.
        
        Args:
            weights: Dictionary of backbone weights
            output_path: Path to saved weights
            metadata_path: Path to save metadata
            
        Returns:
            Path to saved metadata file
        """
        stats = self.get_weight_statistics(weights)
        
        metadata = {
            'checkpoint_source': str(self.checkpoint_path.absolute()),
            'weights_file': str(output_path.absolute()),
            'extraction_timestamp': torch.cuda.Event(enable_timing=True).record() 
                if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'num_weight_tensors': stats['num_tensors'],
            'total_parameters': stats['total_parameters'],
            'size_mb': stats['size_mb'],
            'weight_keys': list(weights.keys()),
            'device_used': self.device,
            'parameter_breakdown': stats['parameter_breakdown'],
            'dtype_breakdown': stats['dtype_breakdown'],
        }
        
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump({k: str(v) for k, v in metadata.items()}, f, indent=2)
            logger.info(f"✓ Saved metadata to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
        
        return metadata_path
    
    def log_weight_shapes(self, weights: Dict[str, torch.Tensor], limit: int = None) -> None:
        """
        Log detailed information about weight shapes and sizes.
        
        Args:
            weights: Dictionary of backbone weights
            limit: Maximum number of weights to display (None for all)
        """
        logger.info("Backbone weight details:")
        
        total_params = 0
        items = sorted(weights.items())
        if limit:
            items = items[:limit]
        
        for key, tensor in items:
            num_params = tensor.numel()
            total_params += num_params
            size_mb = tensor.element_size() * num_params / (1024 * 1024)
            logger.info(
                f"  {key:50s} | {str(tuple(tensor.shape)):25s} | "
                f"{num_params:10,d} params | {size_mb:8.2f} MB | {str(tensor.dtype)}"
            )
        
        logger.info(f"Total parameters: {total_params:,}")


def load_backbone_weights(model: nn.Module, weights_path: str, strict: bool = True) -> None:
    """
    Load extracted backbone weights into a model.
    
    This function handles:
    - Loading weights from extracted files
    - Removing unnecessary prefixes
    - Loading into model.backbone or directly into model
    - Strict vs non-strict loading
    
    Args:
        model: PyTorch model or Lightning module with backbone
        weights_path: Path to extracted weights file
        strict: Whether to require exact state_dict match
        
    Raises:
        FileNotFoundError: If weights file not found
        RuntimeError: If loading fails
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    logger.info(f"Loading backbone weights from: {weights_path}")
    
    try:
        device = next(model.parameters()).device
        weights = torch.load(weights_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights: {e}")
    
    # Try to load into model.backbone first (Lightning module case)
    if hasattr(model, 'backbone'):
        try:
            model.backbone.load_state_dict(weights, strict=strict)
            logger.info(f"✓ Loaded weights into model.backbone (strict={strict})")
        except (RuntimeError, TypeError) as e:
            if strict:
                logger.error(f"Strict loading failed: {e}")
                raise
            else:
                logger.warning(f"Non-strict loading encountered issue: {e}")
    else:
        # Load directly into model
        try:
            model.load_state_dict(weights, strict=strict)
            logger.info(f"✓ Loaded weights into model (strict={strict})")
        except (RuntimeError, TypeError) as e:
            if strict:
                logger.error(f"Strict loading failed: {e}")
                raise
            else:
                logger.warning(f"Non-strict loading encountered issue: {e}")


def main():
    """Main extraction pipeline."""
    parser = argparse.ArgumentParser(
        description='Extract BackboneTransformer weights from Lightning checkpoints'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to Lightning checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='backbone_weights',
        help='Output directory for backbone weights (default: backbone_weights)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='backbone_weights',
        help='Folder name for weights (default: backbone_weights). Weights saved as backbone.pt inside this folder'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to load checkpoint on (default: cpu)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify extracted weights structure'
    )
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Save metadata file alongside weights'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed weight information'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Use strict mode when loading (requires exact key match)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        logger.info(f"Starting backbone weight extraction")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Output directory: {args.output_dir}")
        
        extractor = BackboneWeightExtractor(args.checkpoint, device=args.device)
        
        # Extract weights
        backbone_weights = extractor.extract_backbone_weights()
        
        # Verify structure if requested
        if args.verify:
            is_valid = extractor.validate_backbone_weights(backbone_weights)
            if not is_valid and args.strict:
                logger.error("Backbone structure validation failed in strict mode")
                return 1
        
        # Log weight details if verbose
        if args.verbose:
            extractor.log_weight_shapes(backbone_weights)
        else:
            # Show summary even in non-verbose mode
            stats = extractor.get_weight_statistics(backbone_weights)
            logger.info(
                f"Summary: {stats['num_tensors']} tensors, "
                f"{stats['total_parameters']:,} parameters, "
                f"{stats['size_mb']:.2f} MB"
            )
        
        # Save weights
        checkpoint_dir = Path(args.output_dir) / args.output_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = checkpoint_dir / 'backbone.pt'
        saved_path = extractor.save_weights(backbone_weights, weights_path)
        
        # Save metadata if requested
        if args.metadata:
            metadata_path = checkpoint_dir / 'backbone.json'
            extractor.save_metadata(backbone_weights, saved_path, metadata_path)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ EXTRACTION COMPLETE")
        logger.info("=" * 70)
        pt_path = saved_path
        ckpt_path = saved_path.with_suffix('.ckpt')
        logger.info(f"Weights saved to:")
        logger.info(f"  - {pt_path}")
        logger.info(f"  - {ckpt_path}")
        logger.info(f"File size:         {saved_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"Number of tensors: {len(backbone_weights)}")
        logger.info(f"Total parameters:  {sum(t.numel() for t in backbone_weights.values()):,}")
        
        if args.metadata:
            logger.info(f"Metadata saved to: {metadata_path}")
        
        logger.info("")
        logger.info("Usage for fine-tuning:")
        logger.info(f"  from scripts.extract_backbone_weights import load_backbone_weights")
        logger.info(f"  load_backbone_weights(model, '{saved_path}', strict={args.strict})")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
