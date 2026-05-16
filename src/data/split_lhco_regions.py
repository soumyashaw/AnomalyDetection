"""
Split LHCO dijet data into Signal Region (SR) and Side Bands (SB) categories.

This script samples dijets from the mass window 2900-4100 GeV and splits them into:
- Signal SR (3300-3700 GeV, label=1)
- Background SR (3300-3700 GeV, label=0)
- Signal SB (outside SR but within 2900-4100 GeV, label=1)
- Background SB (outside SR but within 2900-4100 GeV, label=0)
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Dict


def load_and_filter_data(
    h5_filename: str,
    mass_window: Tuple[float, float] = (2900, 4100),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data and filter by mass window.
    
    Parameters
    ----------
    h5_filename : str
        Path to input HDF5 file
    mass_window : tuple
        (min_mass, max_mass) in GeV for the mass window
        
    Returns
    -------
    indices : np.ndarray
        Original indices of events in mass window
    mjj : np.ndarray
        Dijet masses in mass window
    labels : np.ndarray
        Labels (0=background, 1=signal) in mass window
    """
    with h5py.File(h5_filename, "r") as f:
        # Load dijet mass and labels
        mjj_all = f["jet_features"][:, 6]  # mjj is at index 6
        labels_all = f["signal"][:]
        
        # Filter by mass window
        in_window = (mjj_all >= mass_window[0]) & (mjj_all <= mass_window[1])
        indices = np.where(in_window)[0]
        mjj = mjj_all[in_window]
        labels = labels_all[in_window]
        
    print(f"Loaded {len(mjj)} events in mass window [{mass_window[0]}, {mass_window[1]}] GeV")
    print(f"  - Background (label=0): {np.sum(labels == 0)}")
    print(f"  - Signal (label=1): {np.sum(labels == 1)}")
    
    return indices, mjj, labels


def categorize_events(
    mjj: np.ndarray,
    labels: np.ndarray,
    signal_region: Tuple[float, float] = (3300, 3700),
) -> Dict[str, np.ndarray]:
    """Categorize events into SR and SB for signal and background.
    
    Parameters
    ----------
    mjj : np.ndarray
        Dijet masses
    labels : np.ndarray
        Event labels (0=background, 1=signal)
    signal_region : tuple
        (min_mass, max_mass) defining the signal region
        
    Returns
    -------
    categories : dict
        Dictionary with keys: 'signal_SR', 'background_SR', 'signal_SB', 'background_SB'
        Each value is a boolean mask array
    """
    in_sr = (mjj >= signal_region[0]) & (mjj <= signal_region[1])
    in_sb = ~in_sr  # Side bands = not in SR
    
    is_signal = labels == 1
    is_background = labels == 0
    
    categories = {
        'signal_SR': in_sr & is_signal,
        'background_SR': in_sr & is_background,
        'signal_SB': in_sb & is_signal,
        'background_SB': in_sb & is_background,
    }
    
    return categories


def sample_events(
    categories: Dict[str, np.ndarray],
    n_background_total: int = 200000,
    n_signal_total: int = 1000,
    random_seed: int = 42,
    sr_fraction_std: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Randomly sample events to meet target counts.
    
    Parameters
    ----------
    categories : dict
        Dictionary of boolean masks for each category
    n_background_total : int
        Total number of background events to sample (SR + SB)
    n_signal_total : int
        Total number of signal events to sample (SR + SB)
    random_seed : int
        Random seed for reproducibility
    sr_fraction_std : float
        Standard deviation for SR/SB split variation (default 0.05 = 5%)
        
    Returns
    -------
    sampled_indices : dict
        Dictionary with sampled indices for each category
    """
    rng = np.random.default_rng(random_seed)
    
    # Get available counts
    n_bg_sr_avail = np.sum(categories['background_SR'])
    n_bg_sb_avail = np.sum(categories['background_SB'])
    n_sig_sr_avail = np.sum(categories['signal_SR'])
    n_sig_sb_avail = np.sum(categories['signal_SB'])
    
    print("\nAvailable events:")
    print(f"  Background SR: {n_bg_sr_avail}")
    print(f"  Background SB: {n_bg_sb_avail}")
    print(f"  Signal SR: {n_sig_sr_avail}")
    print(f"  Signal SB: {n_sig_sb_avail}")
    
    # Calculate base proportions from available data
    # For background
    bg_total_avail = n_bg_sr_avail + n_bg_sb_avail
    if bg_total_avail < n_background_total:
        print(f"\nWarning: Only {bg_total_avail} background events available, using all")
        n_background_total = bg_total_avail
    
    bg_sr_fraction_base = n_bg_sr_avail / bg_total_avail if bg_total_avail > 0 else 0
    
    # Add random variation to the SR fraction (seed-dependent)
    bg_sr_fraction_delta = rng.normal(0, sr_fraction_std)
    bg_sr_fraction = np.clip(bg_sr_fraction_base + bg_sr_fraction_delta, 0.1, 0.9)
    
    n_bg_sr_sample = min(int(n_background_total * bg_sr_fraction), n_bg_sr_avail)
    n_bg_sb_sample = min(n_background_total - n_bg_sr_sample, n_bg_sb_avail)
    
    # For signal
    sig_total_avail = n_sig_sr_avail + n_sig_sb_avail
    if sig_total_avail < n_signal_total:
        print(f"Warning: Only {sig_total_avail} signal events available, using all")
        n_signal_total = sig_total_avail
    
    sig_sr_fraction_base = n_sig_sr_avail / sig_total_avail if sig_total_avail > 0 else 0
    
    # Add random variation to the SR fraction (seed-dependent)
    sig_sr_fraction_delta = rng.normal(0, sr_fraction_std)
    sig_sr_fraction = np.clip(sig_sr_fraction_base + sig_sr_fraction_delta, 0.1, 0.9)
    
    n_sig_sr_sample = min(int(n_signal_total * sig_sr_fraction), n_sig_sr_avail)
    n_sig_sb_sample = min(n_signal_total - n_sig_sr_sample, n_sig_sb_avail)
    
    print(f"\nSampling strategy (seed={random_seed}):")
    print(f"  Background SR fraction: {bg_sr_fraction_base:.3f} → {bg_sr_fraction:.3f} (Δ={bg_sr_fraction_delta:+.3f})")
    print(f"  Signal SR fraction: {sig_sr_fraction_base:.3f} → {sig_sr_fraction:.3f} (Δ={sig_sr_fraction_delta:+.3f})")
    print(f"  Background SR: {n_bg_sr_sample} / {n_bg_sr_avail}")
    print(f"  Background SB: {n_bg_sb_sample} / {n_bg_sb_avail}")
    print(f"  Signal SR: {n_sig_sr_sample} / {n_sig_sr_avail}")
    print(f"  Signal SB: {n_sig_sb_sample} / {n_sig_sb_avail}")
    
    # Sample indices
    sampled_indices = {}
    
    for cat_name, mask in categories.items():
        available_indices = np.where(mask)[0]
        
        if cat_name == 'background_SR':
            n_sample = n_bg_sr_sample
        elif cat_name == 'background_SB':
            n_sample = n_bg_sb_sample
        elif cat_name == 'signal_SR':
            n_sample = n_sig_sr_sample
        elif cat_name == 'signal_SB':
            n_sample = n_sig_sb_sample
        
        if n_sample > 0 and len(available_indices) > 0:
            sampled_indices[cat_name] = rng.choice(
                available_indices, size=n_sample, replace=False
            )
        else:
            sampled_indices[cat_name] = np.array([], dtype=int)
    
    return sampled_indices


def copy_events_to_h5(
    input_file: str,
    output_file: str,
    indices: np.ndarray,
    datasets_to_copy: list = None,
    batch_size: int = 10000,
):
    """Copy selected events from input to output HDF5 file.
    
    Parameters
    ----------
    input_file : str
        Path to input HDF5 file
    output_file : str
        Path to output HDF5 file
    indices : np.ndarray
        Indices of events to copy
    datasets_to_copy : list, optional
        List of dataset paths to copy. If None, copies ALL datasets.
    batch_size : int
        Number of events to copy at once (for large datasets)
    """
    if len(indices) == 0:
        print(f"No events to write to {output_file}")
        return
    
    # Sort indices for HDF5 fancy indexing requirement
    indices_sorted = np.sort(indices)
    n_events = len(indices_sorted)
    
    print(f"  Copying {n_events} events to {output_file}...")
    
    with h5py.File(input_file, "r") as f_in:
        with h5py.File(output_file, "w") as f_out:
            # If no specific datasets specified, get all datasets
            if datasets_to_copy is None:
                datasets_to_copy = []
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets_to_copy.append(name)
                f_in.visititems(collect_datasets)
            
            print(f"    Found {len(datasets_to_copy)} datasets to copy")
            
            # Copy datasets with progress indicator
            for i, dataset_path in enumerate(datasets_to_copy):
                if dataset_path in f_in:
                    dataset = f_in[dataset_path]
                    
                    # Create parent groups if they don't exist
                    if '/' in dataset_path:
                        parent_path = '/'.join(dataset_path.split('/')[:-1])
                        if parent_path and parent_path not in f_out:
                            f_out.create_group(parent_path)
                    
                    # For large datasets, copy in batches
                    if n_events > batch_size and dataset.nbytes > 100 * 1024 * 1024:  # > 100 MB
                        print(f"    [{i+1}/{len(datasets_to_copy)}] {dataset_path} (batched, {dataset.nbytes / 1024**2:.1f} MB)")
                        
                        # Read first batch to get shape
                        first_batch = dataset[indices_sorted[:min(batch_size, n_events)]]
                        output_shape = (n_events,) + first_batch.shape[1:]
                        
                        # Create output dataset
                        out_dset = f_out.create_dataset(
                            dataset_path,
                            shape=output_shape,
                            dtype=first_batch.dtype,
                            compression='gzip',
                            compression_opts=1,  # Lower compression for speed
                            chunks=True
                        )
                        
                        # Write first batch
                        out_dset[:len(first_batch)] = first_batch
                        
                        # Process remaining batches
                        for batch_start in range(batch_size, n_events, batch_size):
                            batch_end = min(batch_start + batch_size, n_events)
                            batch_indices = indices_sorted[batch_start:batch_end]
                            batch_data = dataset[batch_indices]
                            out_dset[batch_start:batch_end] = batch_data
                            if batch_start % (batch_size * 5) == 0:
                                print(f"      Progress: {batch_end}/{n_events} events")
                    else:
                        # Small dataset or few events - copy all at once
                        data = dataset[indices_sorted]
                        f_out.create_dataset(
                            dataset_path, 
                            data=data, 
                            compression='gzip',
                            compression_opts=1  # Lower compression for speed
                        )
                        print(f"    [{i+1}/{len(datasets_to_copy)}] {dataset_path}")
                    
                    # Copy attributes
                    for attr_name, attr_value in dataset.attrs.items():
                        f_out[dataset_path].attrs[attr_name] = attr_value
            
            # Copy top-level attributes
            for key in f_in.attrs.keys():
                f_out.attrs[key] = f_in.attrs[key]
    
    print(f"  ✓ Completed {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split LHCO dijet data into SR and SB categories"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/.automount/net_rw/net__data_ttk/soshaw/bg_N100.h5",
        help="Input HDF5 file with dijet data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_split",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=200000,
        help="Total number of background events to sample"
    )
    parser.add_argument(
        "--n-signal",
        type=int,
        default=1000,
        help="Total number of signal events to sample"
    )
    parser.add_argument(
        "--mass-window",
        type=float,
        nargs=2,
        default=[2900, 4100],
        help="Mass window (min max) in GeV"
    )
    parser.add_argument(
        "--signal-region",
        type=float,
        nargs=2,
        default=[3300, 3700],
        help="Signal region (min max) in GeV"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--sr-fraction-std",
        type=float,
        default=0.01,
        help="Standard deviation for SR/SB split variation (default 0.01 = 1%%)"
    )
    parser.add_argument(
        "--jet-level-only",
        action="store_true",
        help="Copy only jet-level data (jet_features, signal, jet_coords) for speed. "
             "By default, copies ALL datasets including constituent-level data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output files. By default, skips existing files."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("LHCO Dijet Region Splitter")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Mass window: {args.mass_window[0]}-{args.mass_window[1]} GeV")
    print(f"Signal region: {args.signal_region[0]}-{args.signal_region[1]} GeV")
    print(f"Target: {args.n_background} background, {args.n_signal} signal events")
    print("=" * 70)
    
    # Load and filter data
    print("\n[1/4] Loading data...")
    indices, mjj, labels = load_and_filter_data(
        args.input_file,
        mass_window=tuple(args.mass_window)
    )
    
    # Categorize events
    print("\n[2/4] Categorizing events...")
    categories = categorize_events(
        mjj,
        labels,
        signal_region=tuple(args.signal_region)
    )
    
    # Sample events
    print("\n[3/4] Sampling events...")
    sampled_local_indices = sample_events(
        categories,
        n_background_total=args.n_background,
        n_signal_total=args.n_signal,
        random_seed=args.seed,
        sr_fraction_std=args.sr_fraction_std
    )
    
    # Convert local indices back to original file indices
    sampled_file_indices = {
        cat: indices[local_idx]
        for cat, local_idx in sampled_local_indices.items()
    }
    
    # Save to separate files
    print("\n[4/4] Writing output files...")
    
    # Determine which datasets to copy
    if args.jet_level_only:
        print("Copying only jet-level data (fast mode)...")
        datasets_to_copy = ['jet_features', 'signal', 'jet_coords']
    else:
        print("Copying all datasets including constituent-level data...")
        datasets_to_copy = None  # Will auto-discover all datasets
    
    output_files = {
        'signal_SR': output_dir / "signal_SR.h5",
        'background_SR': output_dir / "background_SR_supp.h5",
        'signal_SB': output_dir / "signal_SB_supp.h5",
        'background_SB': output_dir / "background_SB.h5",
    }
    
    for cat_name, file_path in output_files.items():
        print(f"\n{cat_name}:")
        
        # Skip if file already exists (unless --force is used)
        if file_path.exists() and not args.force:
            print(f"  ✓ File already exists: {file_path} (skipping, use --force to overwrite)")
            continue
        
        copy_events_to_h5(
            args.input_file,
            str(file_path),
            sampled_file_indices[cat_name],
            datasets_to_copy=datasets_to_copy
        )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL COUNTS")
    print("=" * 70)
    n_bg_sr = len(sampled_file_indices['background_SR'])
    n_bg_sb = len(sampled_file_indices['background_SB'])
    n_sig_sr = len(sampled_file_indices['signal_SR'])
    n_sig_sb = len(sampled_file_indices['signal_SB'])
    
    print(f"Background SR (3300-3700 GeV, label=0): {n_bg_sr:,}")
    print(f"Background SB (side bands, label=0):    {n_bg_sb:,}")
    print(f"Signal SR (3300-3700 GeV, label=1):     {n_sig_sr:,}")
    print(f"Signal SB (side bands, label=1):        {n_sig_sb:,}")
    print("-" * 70)
    print(f"Total Background: {n_bg_sr + n_bg_sb:,}")
    print(f"Total Signal:     {n_sig_sr + n_sig_sb:,}")
    print(f"Total Events:     {n_bg_sr + n_bg_sb + n_sig_sr + n_sig_sb:,}")
    print("=" * 70)
    
    print(f"\nOutput files saved to: {output_dir}")
    print("  - signal_SR.h5")
    print("  - background_SR.h5")
    print("  - signal_SB.h5")
    print("  - background_SB.h5")


if __name__ == "__main__":
    main()
