#!/usr/bin/env python3
"""
Script to create three copies of an HDF5 dataset, each with a configurable
number of jets (defaults to 200k per file).
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def split_h5_dataset(input_file, output_dir, n_jets=200000, n_jets_2=None, n_jets_3=None, seed=42):
    """
    Create three copies of an HDF5 dataset with specified number of jets.

    Args:
        input_file: Path to input HDF5 file
        output_dir: Directory to save output files
        n_jets: Number of jets for first output file (default: 200000)
        n_jets_2: Number of jets for second output file (default: same as n_jets)
        n_jets_3: Number of jets for third output file (default: same as n_jets)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    if n_jets_2 is None:
        n_jets_2 = n_jets
    if n_jets_3 is None:
        n_jets_3 = n_jets
    
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the input file
    print(f"Reading from: {input_file}")
    with h5py.File(input_file, 'r') as f_in:
        # Get total number of jets
        total_jets = f_in['signal'].shape[0]
        print(f"Total jets in input file: {total_jets}")
        
        total_needed = n_jets + n_jets_2 + n_jets_3
        if total_jets < total_needed:
            raise ValueError(
                f"Input file has only {total_jets} jets, but need {total_needed} "
                f"({n_jets} + {n_jets_2} + {n_jets_3})"
            )

        # Generate random indices for three non-overlapping subsets
        all_indices = np.arange(total_jets)
        np.random.shuffle(all_indices)

        indices_1 = np.sort(all_indices[:n_jets])
        indices_2 = np.sort(all_indices[n_jets:n_jets + n_jets_2])
        indices_3 = np.sort(all_indices[n_jets + n_jets_2:n_jets + n_jets_2 + n_jets_3])

        print(f"Creating three datasets with {n_jets}, {n_jets_2} and {n_jets_3} jets...")
        
        # Create two output files with batch processing for speed
        batch_size = 20000  # Process 20k jets at a time
        
        for idx, (indices, suffix, n_jets_file) in enumerate(
            [(indices_1, '1', n_jets), (indices_2, '2', n_jets_2), (indices_3, '3', n_jets_3)], 1
        ):
            output_file = output_dir / f"{input_path.stem}_copy{suffix}.h5"
            print(f"\nCreating file {idx}/3: {output_file} ({n_jets_file} jets)")
            
            with h5py.File(output_file, 'w') as f_out:
                # Copy jet1 group with batching
                print("  Copying jet1 group...")
                jet1_group = f_out.create_group('jet1')
                for dataset_name in ['4mom', 'coords', 'features', 'mask']:
                    src_data = f_in['jet1'][dataset_name]
                    shape = (n_jets_file,) + src_data.shape[1:]
                    
                    # Create dataset
                    dst_data = jet1_group.create_dataset(
                        dataset_name,
                        shape=shape,
                        dtype=src_data.dtype,
                        maxshape=(None,) + src_data.shape[1:],
                        chunks=(min(1000, n_jets_file),) + src_data.shape[1:],
                        compression='gzip',
                        compression_opts=1
                    )
                    
                    # Copy in batches
                    for i in range(0, n_jets_file, batch_size):
                        end = min(i + batch_size, n_jets_file)
                        dst_data[i:end] = src_data[indices[i:end]]
                
                # Copy jet2 group with batching
                print("  Copying jet2 group...")
                jet2_group = f_out.create_group('jet2')
                for dataset_name in ['4mom', 'coords', 'features', 'mask']:
                    src_data = f_in['jet2'][dataset_name]
                    shape = (n_jets_file,) + src_data.shape[1:]
                    
                    dst_data = jet2_group.create_dataset(
                        dataset_name,
                        shape=shape,
                        dtype=src_data.dtype,
                        maxshape=(None,) + src_data.shape[1:],
                        chunks=(min(1000, n_jets_file),) + src_data.shape[1:],
                        compression='gzip',
                        compression_opts=1
                    )
                    
                    for i in range(0, n_jets_file, batch_size):
                        end = min(i + batch_size, n_jets_file)
                        dst_data[i:end] = src_data[indices[i:end]]
                
                # Copy top-level datasets with batching
                print("  Copying top-level datasets...")
                for dataset_name in ['jet_coords', 'jet_features', 'signal']:
                    src_data = f_in[dataset_name]
                    if src_data.ndim > 1:
                        shape = (n_jets_file,) + src_data.shape[1:]
                        maxshape = (None,) + src_data.shape[1:]
                        chunks = (min(1000, n_jets_file),) + src_data.shape[1:]
                    else:
                        shape = (n_jets_file,)
                        maxshape = (None,)
                        chunks = (min(1000, n_jets_file),)
                    
                    dst_data = f_out.create_dataset(
                        dataset_name,
                        shape=shape,
                        dtype=src_data.dtype,
                        maxshape=maxshape,
                        chunks=chunks,
                        compression='gzip',
                        compression_opts=1
                    )
                    
                    for i in range(0, n_jets_file, batch_size):
                        end = min(i + batch_size, n_jets_file)
                        dst_data[i:end] = src_data[indices[i:end]]
            
        print(f"  ✓ Saved to: {output_file}")

    print(f"\n✅ Successfully created 3 files ({n_jets}, {n_jets_2}, {n_jets_3} jets)!")


def main():
    parser = argparse.ArgumentParser(
        description='Split HDF5 dataset into three copies with specified number of jets each'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/.automount/net_rw/net__data_ttk/hreyes/LHCO/processed_jg/original/bg_N100_SR_extra.h5',
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/.automount/net_rw/net__data_ttk/soshaw',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--n-jets',
        type=int,
        default=200000,
        help='Number of jets for first output file (default: 200000)'
    )
    parser.add_argument(
        '--n-jets-2',
        type=int,
        default=None,
        help='Number of jets for second output file (default: same as --n-jets)'
    )
    parser.add_argument(
        '--n-jets-3',
        type=int,
        default=None,
        help='Number of jets for third output file (default: same as --n-jets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    split_h5_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        n_jets=args.n_jets,
        n_jets_2=args.n_jets_2,
        n_jets_3=args.n_jets_3,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
