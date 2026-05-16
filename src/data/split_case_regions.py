"""
Split CASE dataset by dijet invariant mass regions.

This script reads CASE HDF5 files and extracts jets within specified dijet mass regions.
Useful for creating signal region, sideband region, or control region datasets.

Usage:
    # Extract 100k jets from signal region across multiple files
    python -m src.data.split_case_regions \
        --input background_0.h5 background_1.h5 background_2.h5 \
        --output background_signal_region.h5 \
        --mass_range 2900 3100 \
        --n_jets 100000
    
    # Extract all jets in sideband region
    python -m src.data.split_case_regions \
        --input background_0.h5 \
        --output background_sideband.h5 \
        --mass_range 2900 3100 \
        --inverse
    
    # Extract specific number from custom region
    python -m src.data.split_case_regions \
        --input signal_1.h5 signal_2.h5 signal_3.h5 \
        --output signal_combined.h5 \
        --mass_range 2725 3331 \
        --n_jets 50000
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def load_case_dijet_mass(h5_filename: str) -> np.ndarray:
    """Load dijet invariant mass from CASE HDF5 file.
    
    Parameters
    ----------
    h5_filename : str
        Path to CASE HDF5 file.
    
    Returns
    -------
    np.ndarray
        Dijet invariant mass (m_jj) in GeV, shape (N,).
    """
    with h5py.File(h5_filename, "r") as f:
        # jet_kinematics has shape (N, 14), m_jj is at index 0
        m_jj = f["jet_kinematics"][:, 0]
    return m_jj


def create_mass_mask(
    m_jj: np.ndarray,
    mass_range: Tuple[float, float],
    inverse: bool = False,
) -> np.ndarray:
    """Create boolean mask for mass selection.
    
    Parameters
    ----------
    m_jj : np.ndarray
        Dijet invariant mass array.
    mass_range : Tuple[float, float]
        Mass range (min, max) in GeV.
    inverse : bool, optional
        If True, select events OUTSIDE the mass range.
    
    Returns
    -------
    np.ndarray
        Boolean mask, True for selected events.
    """
    min_mass, max_mass = mass_range
    
    if inverse:
        # Select events outside [min_mass, max_mass]
        mask = (m_jj < min_mass) | (m_jj > max_mass)
    else:
        # Select events inside [min_mass, max_mass]
        mask = (m_jj >= min_mass) & (m_jj <= max_mass)
    
    return mask


def split_case_h5_by_mass(
    input_files: List[str],
    output_file: str,
    mass_range: Tuple[float, float],
    inverse: bool = False,
    max_events: int = None,
) -> None:
    """Split CASE HDF5 file(s) by dijet mass region.
    
    Extracts dijet events from one or more input files that fall within (or outside)
    a specified mass range. Useful for creating signal region, sideband, or control
    region datasets. Can limit total number of extracted events.
    
    Parameters
    ----------
    input_files : List[str]
        List of input CASE HDF5 file paths. Files are processed in order until
        the desired number of events is reached.
    output_file : str
        Output HDF5 file path.
    mass_range : Tuple[float, float]
        Mass range (min, max) in GeV to extract.
    inverse : bool, optional
        If True, extract events OUTSIDE the mass range (for sidebands).
    max_events : int, optional
        Maximum number of events to extract across all input files.
        None extracts all matching events from all files.
    """
    # CASE dataset keys
    dataset_keys = [
        "jet1_PFCands",     # (N, 100, 4) float16
        "jet2_PFCands",     # (N, 100, 4) float16
        "jet1_extraInfo",   # (N, 7) float32
        "jet2_extraInfo",   # (N, 7) float32
        "jet_kinematics",   # (N, 14) float32
        "truth_label",      # (N, 1) int
    ]
    
    # Collect masks from all input files
    all_masks = []
    file_sizes = []
    
    print(f"Analyzing {len(input_files)} input file(s)...")
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        m_jj = load_case_dijet_mass(input_file)
        mask = create_mass_mask(m_jj, mass_range, inverse)
        
        all_masks.append(mask)
        file_sizes.append(len(mask))
        
        n_selected = np.sum(mask)
        print(f"  {os.path.basename(input_file)}: {n_selected:,}/{len(mask):,} events in range")
    
    total_selected = sum(np.sum(mask) for mask in all_masks)
    
    if total_selected == 0:
        print(f"\nWarning: No events found in mass range {mass_range} GeV")
        print("No output file created.")
        return
    
    # Apply max_events limit if specified
    if max_events is not None and total_selected > max_events:
        print(f"\nTotal available: {total_selected:,} events")
        print(f"Extracting: {max_events:,} events (as requested)")
        total_to_save = max_events
    else:
        total_to_save = total_selected
        print(f"\nTotal available: {total_selected:,} events")
        print(f"Extracting: all {total_to_save:,} events")
    
    region_type = "outside" if inverse else "within"
    print(f"Mass range: [{mass_range[0]:.1f}, {mass_range[1]:.1f}] GeV ({region_type})")
    print(f"Output file: {output_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write selected events to output file
    with h5py.File(output_file, "w") as f_out:
        # Determine data types and shapes from first input file
        with h5py.File(input_files[0], "r") as f_in:
            dtypes = {key: f_in[key].dtype for key in dataset_keys}
            shapes = {key: f_in[key].shape[1:] for key in dataset_keys}  # Shape without batch dimension
        
        # Create datasets in output file
        datasets_out = {}
        for key in dataset_keys:
            full_shape = (total_to_save,) + shapes[key]
            datasets_out[key] = f_out.create_dataset(
                key,
                shape=full_shape,
                dtype=dtypes[key],
                compression="gzip",
                compression_opts=4,
            )
        
        # Copy data from input files
        write_idx = 0
        events_written = 0
        
        for input_file, mask in zip(input_files, all_masks):
            n_selected_file = np.sum(mask)
            
            if n_selected_file == 0:
                continue
            
            # Determine how many events to take from this file
            events_to_take = min(n_selected_file, total_to_save - events_written)
            
            if events_to_take == 0:
                break
            
            with h5py.File(input_file, "r") as f_in:
                # Get indices of selected events
                selected_indices = np.where(mask)[0]
                
                # If we need to limit, take first N events
                if events_to_take < len(selected_indices):
                    selected_indices = selected_indices[:events_to_take]
                
                # Copy each dataset
                file_basename = os.path.basename(input_file)
                for key in tqdm(
                    dataset_keys,
                    desc=f"Copying {events_to_take:,} events from {file_basename}",
                ):
                    data_selected = f_in[key][selected_indices]
                    datasets_out[key][write_idx : write_idx + events_to_take] = data_selected
                
                write_idx += events_to_take
                events_written += events_to_take
        
        # Verify label distribution
        labels = datasets_out["truth_label"][:]
        labels = np.clip(labels, 0, None).squeeze()  # Handle negative labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\nOutput file statistics:")
        print(f"  Total events: {len(labels)}")
        for label, count in zip(unique_labels, counts):
            label_name = "background" if label == 0 else "signal"
            print(f"  Label {label} ({label_name}): {count} ({100*count/len(labels):.1f}%)")
    
    print(f"\n✓ Successfully created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split CASE dataset by dijet invariant mass regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Extract 100k jets from signal region across multiple files
            python -m src.data.split_case_regions -i bg_0.h5 bg_1.h5 bg_2.h5 -o bg_signal.h5 -m 2900 3100 -n 100000
            
            # Extract all jets in sideband (outside signal region)
            python -m src.data.split_case_regions -i background.h5 -o bg_sideband.h5 -m 2900 3100 --inverse
            
            # Combine specific number from multiple signal files
            python -m src.data.split_case_regions -i sig_1.h5 sig_2.h5 sig_3.h5 -o signal_SR.h5 -m 2900 3100 -n 50000
            
            # Extract all matching jets from multiple files
            python -m src.data.split_case_regions -i file1.h5 file2.h5 file3.h5 -o combined.h5 -m 2725 3331
        """,
    )
    
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input CASE HDF5 file(s). Multiple files will be processed in order.",
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output HDF5 file path",
    )
    
    parser.add_argument(
        "-m", "--mass_range",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="Mass range [min, max] in GeV (e.g., -m 2900 3100 for signal region)",
    )
    
    parser.add_argument(
        "-n", "--n_jets",
        type=int,
        default=None,
        dest="max_events",
        help="Number of jets to extract across all input files (default: extract all matching jets)",
    )
    
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Extract events OUTSIDE the mass range (for sideband selection)",
    )
    
    args = parser.parse_args()
    
    # Validate mass range
    if args.mass_range[0] >= args.mass_range[1]:
        raise ValueError(f"Invalid mass range: min ({args.mass_range[0]}) must be < max ({args.mass_range[1]})")
    
    # Run splitting
    split_case_h5_by_mass(
        input_files=args.input,
        output_file=args.output,
        mass_range=tuple(args.mass_range),
        inverse=args.inverse,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
