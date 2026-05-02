"""Cache CASE HDF5 files to disk as per-file PyTorch tensor pickle files.

This script processes one HDF5 file at a time and saves a per-file cache so
that evaluate_CASE.py can load them quickly without re-reading HDF5.

MULTI-NODE SUPPORT
------------------
Multiple allocations (CPU nodes) can run this script simultaneously pointing at
the same --cache_dir.  Before processing a file each worker atomically creates
a lock sentinel (<cache_file>.lock) using O_CREAT|O_EXCL.  If the lock already
exists the worker skips that file.  On completion the .pkl is written and the
.lock is removed.  Workers that start after caching is complete will simply
find all .pkl files present and exit immediately.

USAGE
-----
# Cache all files (run on one or more nodes simultaneously):
python -m scripts.cache_case_data \
    --dataset_path /.automount/net_rw/net__data_ttk/soshaw/CASE \
    --cache_dir .cache/case_per_file \
    --jet_name both

# Cache only a subset of files (useful for slicing across node array jobs):
python -m src.eval.cache_case_data \
    --file_indices 0 1 2 3 4

# Cache a single file by index (zero-based, sorted glob order):
python -m src.eval.cache_case_data \
    --dataset_path /.automount/net_rw/net__data_ttk/soshaw/CASE \
    --cache_dir .cache/case_per_file \
    --jet_name both \
    --file_index 3

# Limit events loaded per file:
python -m src.eval.cache_case_data \
    --dataset_path /.automount/net_rw/net__data_ttk/soshaw/CASE \
    --cache_dir .cache/case_per_file \
    --jet_name both \
    --n_jets 100000
"""

import argparse
import errno
import glob
import hashlib
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import awkward as ak
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Add project root to sys.path so gabbro is importable when run as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gabbro.data.loading import load_case_jets_from_h5
from gabbro.utils.arrays import ak_pad

# ---------------------------------------------------------------------------
# Feature configuration (must match evaluate_CASE.py)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_DICT = {
    "part_pt":     {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
    "part_etarel": {"multiply_by": 3},
    "part_phirel": {"multiply_by": 3},
}
MAX_SEQUENCE_LEN = 128


# ---------------------------------------------------------------------------
# Cache key helpers (must match the logic in evaluate_CASE.py)
# ---------------------------------------------------------------------------

def _feature_hash(feature_dict: dict) -> str:
    """Short MD5 of the feature_dict repr for cache naming."""
    return hashlib.md5(repr(sorted(feature_dict.items())).encode()).hexdigest()[:8]


def per_file_cache_path(h5_path: str, cache_dir: Path, jet_name: str,
                         n_jets, feature_dict: dict) -> Path:
    """Return the expected .pkl path for a given HDF5 file."""
    stem = Path(h5_path).stem
    n_jets_str = "all" if n_jets is None else str(n_jets)
    feat_hash = _feature_hash(feature_dict)
    filename = f"{stem}__{jet_name}__n{n_jets_str}__{feat_hash}.pkl"
    return cache_dir / filename


# ---------------------------------------------------------------------------
# Atomic lock helpers
# ---------------------------------------------------------------------------

def _lock_path(cache_pkl: Path) -> Path:
    return cache_pkl.with_suffix(".lock")


def _try_acquire_lock(cache_pkl: Path) -> bool:
    """Attempt to atomically create a .lock file.  Returns True on success."""
    lock = _lock_path(cache_pkl)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise


def _release_lock(cache_pkl: Path):
    lock = _lock_path(cache_pkl)
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Core: process a single file
# ---------------------------------------------------------------------------

def _to_tensors(features_ak, labels_np):
    """Pad awkward array → stack features → torch tensors."""
    padded, mask = ak_pad(
        features_ak, maxlen=MAX_SEQUENCE_LEN, axis=1, fill_value=0.0, return_mask=True
    )
    feature_names = features_ak.fields
    stacked = ak.concatenate(
        [padded[f][..., np.newaxis] for f in feature_names], axis=-1
    )
    x = torch.from_numpy(ak.to_numpy(stacked)).float()
    m = torch.from_numpy(ak.to_numpy(mask)).bool()
    y = torch.from_numpy(labels_np).long()
    return x, m, y


def cache_single_file(h5_path: str, cache_pkl: Path, jet_name: str,
                       n_jets, feature_dict: dict):
    """Load one HDF5 file, convert to tensors, and save as pickle.

    Returns
    -------
    bool
        True if the file was cached successfully, False if skipped/failed.
    """
    if cache_pkl.exists():
        print(f"[SKIP] Already cached: {cache_pkl.name}")
        return False

    if not _try_acquire_lock(cache_pkl):
        # Another worker is processing this file
        lock_path = _lock_path(cache_pkl)
        try:
            pid = lock_path.read_text().strip()
            print(f"[SKIP] Lock held by PID {pid}: {cache_pkl.name}")
        except Exception:
            print(f"[SKIP] Locked: {cache_pkl.name}")
        return False

    print(f"\n[PROCESS] {Path(h5_path).name}  →  {cache_pkl.name}")
    t0 = time.time()
    try:
        if jet_name == "both":
            j1_ak, j2_ak, labels = load_case_jets_from_h5(
                h5_path, feature_dict, n_jets=n_jets, jet_name="both"
            )
            x1, m1, y = _to_tensors(j1_ak, labels)
            x2, m2, _ = _to_tensors(j2_ak, labels)
            data = {
                "features":      x1,
                "masks":         m1,
                "features_jet2": x2,
                "masks_jet2":    m2,
                "labels":        y,
                "jet_name":      jet_name,
                "source_file":   str(h5_path),
                "n_jets":        n_jets,
                "feature_dict":  feature_dict,
            }
            print(f"  jet1: {x1.shape}  jet2: {x2.shape}  "
                  f"bg={(y==0).sum().item()}  sig={(y==1).sum().item()}")
        else:
            feats_ak, labels = load_case_jets_from_h5(
                h5_path, feature_dict, n_jets=n_jets, jet_name=jet_name
            )
            x, m, y = _to_tensors(feats_ak, labels)
            data = {
                "features":    x,
                "masks":       m,
                "labels":      y,
                "jet_name":    jet_name,
                "source_file": str(h5_path),
                "n_jets":      n_jets,
                "feature_dict": feature_dict,
            }
            print(f"  features: {x.shape}  "
                  f"bg={(y==0).sum().item()}  sig={(y==1).sum().item()}")

        with open(cache_pkl, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = cache_pkl.stat().st_size / (1024 ** 2)
        elapsed = time.time() - t0
        print(f"  Saved {cache_pkl.name}  ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return True

    except Exception:
        traceback.print_exc()
        # Remove partial output on failure
        if cache_pkl.exists():
            cache_pkl.unlink()
        return False
    finally:
        _release_lock(cache_pkl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cache CASE HDF5 files as per-file PyTorch tensor pickles."
    )
    parser.add_argument("--dataset_path", default="/.automount/net_rw/net__data_ttk/soshaw/CASE",
                        help="Directory containing CASE .h5 files")
    parser.add_argument("--cache_dir", default="/.automount/net_rw/net__data_ttk/soshaw/CASE/.cache/evaluation_case/",
                        help="Directory where .pkl caches are stored")
    parser.add_argument("--jet_name", default="both",
                        choices=["jet1", "jet2", "both"],
                        help="Which jet(s) to load (default: both)")
    parser.add_argument("--n_jets", type=int, default=None,
                        help="Max events to load per file (default: all)")
    # File selection: choose one of these
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file_index", type=int, default=None,
                       help="Zero-based index of a single file to process "
                            "(sorted glob order)")
    group.add_argument("--file_indices", type=int, nargs="+", default=None,
                       help="Zero-based indices of specific files to process")
    group.add_argument("--file_range", type=int, nargs=2, default=None,
                       metavar=("START", "STOP"),
                       help="Half-open range [START, STOP) of file indices "
                            "to process")
    parser.add_argument("--background_only", action="store_true",
                        help="Process only background_*.h5 files")
    parser.add_argument("--signal_only", action="store_true",
                        help="Process only non-background .h5 files")
    parser.add_argument("--list_files", action="store_true",
                        help="Print the list of files (with indices) and exit")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_dict = DEFAULT_FEATURE_DICT

    # Collect all h5 files (sorted for stable indexing)
    all_h5 = sorted(glob.glob(str(dataset_path / "*.h5")))
    if not all_h5:
        print(f"ERROR: No .h5 files found in {dataset_path}")
        sys.exit(1)

    # Apply background/signal filter
    if args.background_only:
        all_h5 = [f for f in all_h5 if Path(f).name.startswith("background")]
    elif args.signal_only:
        all_h5 = [f for f in all_h5 if not Path(f).name.startswith("background")]

    if args.list_files:
        print(f"{'Index':>5}  File")
        print("-" * 70)
        for idx, f in enumerate(all_h5):
            cache_pkl = per_file_cache_path(f, cache_dir, args.jet_name,
                                             args.n_jets, feature_dict)
            status = "[cached]" if cache_pkl.exists() else (
                "[locked]" if _lock_path(cache_pkl).exists() else "")
            print(f"{idx:>5}  {Path(f).name}  {status}")
        sys.exit(0)

    # Resolve which files to process
    if args.file_index is not None:
        indices = [args.file_index]
    elif args.file_indices is not None:
        indices = args.file_indices
    elif args.file_range is not None:
        indices = list(range(args.file_range[0], args.file_range[1]))
    else:
        indices = list(range(len(all_h5)))

    files_to_process = []
    for idx in indices:
        if idx < 0 or idx >= len(all_h5):
            print(f"WARNING: file_index {idx} out of range (0–{len(all_h5)-1}), skipping")
            continue
        files_to_process.append(all_h5[idx])

    if not files_to_process:
        print("No files to process.")
        sys.exit(0)

    print(f"Dataset path : {dataset_path}")
    print(f"Cache dir    : {cache_dir}")
    print(f"jet_name     : {args.jet_name}")
    print(f"n_jets       : {args.n_jets if args.n_jets else 'all'}")
    print(f"Files selected: {len(files_to_process)} / {len(all_h5)}")
    print()

    n_cached = 0
    n_skipped = 0
    n_failed = 0

    for h5_path in files_to_process:
        cache_pkl = per_file_cache_path(h5_path, cache_dir, args.jet_name,
                                         args.n_jets, feature_dict)
        result = cache_single_file(h5_path, cache_pkl, args.jet_name,
                                    args.n_jets, feature_dict)
        if result is True:
            n_cached += 1
        elif result is False and cache_pkl.exists():
            n_skipped += 1
        else:
            # cache_pkl doesn't exist → lock skip or failure
            if _lock_path(cache_pkl).exists():
                n_skipped += 1
            else:
                n_failed += 1

    print("\n" + "=" * 60)
    print(f"Done.  Cached={n_cached}  Skipped={n_skipped}  Failed={n_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
