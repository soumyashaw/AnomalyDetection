#!/usr/bin/env python3
""" Script to prepare KitchenSink signal samples (https://zenodo.org/records/18983506) as signal-region HDF5 files.

PROJECT STRUCTURE:
AnomalyDetection/
├── src/
│   ├── train/              # Training scripts
│   ├── eval/               # Evaluation scripts
│   ├── data/               # Data processing (e.g., prepare_kitchensink_signal.py)
│   ├── viz/                # Visualization
│   └── poc_expts/          # Proof-of-concept experiments
├── scripts/                # Job launchers
└── [output directories: data/]

PIPELINE:
├── 1. Clustering (replicates KitchenSink feature/jet_clustering.py and feature/hlf.py, commit 6ed1f07)
│   ├── Read events.h5 (pt, eta, phi of up to 700 particles per event)
│   ├── Cluster all particles as massless PseudoJets with anti-kt R = 1.0
│   ├── Keep the two leading-pT jets and their pT-sorted constituents (pt, eta, phi in [0, 2pi), m)
│   ├── mjj from the full jets, computed as in hlf.py
│   └── Cached as <events dir>/clustered_<n_constituents>.h5
├── 2. Signal region: keep events with mjj_min <= mjj <= mjj_max
├── 3. Random train/test split (seeded), events stored in shuffled order
└── 4. Write sn_<MODEL>_<N>_SR_{train,test}.h5 in the schema of sn_XtoYY_*_SR_*.h5
       (pt, eta, phi in [-pi, pi), m), plus jet_coords: full-jet (pt, eta, phi, m)

The outputs still have to be converted to the background schema afterwards:
  python -m src.data.convert_kitchensink_signal --input_files <outputs>

USAGE: python -m src.data.prepare_kitchensink_signal --model XtoWRto3W --events_file raw/XtoWRto3W/events.h5 [options]
"""

import h5py
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

N_PARTICLES_PER_EVENT = 700
KITCHENSINK_COMMIT = "6ed1f07de4df14c935d3d1b70680b0f9275672bf"
CHUNK_EVENTS = 512  # same chunking as the existing KitchenSink signal files


def cluster_events_chunk(events_file, start, stop, n_constituents):
    """Cluster events [start, stop) of a KitchenSink events.h5 file.

    Returns:
        dict with "constituents" (n, 2, n_constituents, 4) as (pt, eta, phi in [0, 2pi), m),
        "n_constituents" (n, 2), "jets_ptyphim" (n, 2, 4), "jets_ptetaphim" (n, 2, 4), "label" (n,)
    """
    import fastjet as fj  # imported here so worker processes build their own fastjet objects

    df = pd.read_hdf(events_file, start=start, stop=stop)
    momenta_columns = [f"{c}{i}" for i in range(N_PARTICLES_PER_EVENT) for c in ("pt", "eta", "phi")]
    if list(df.columns) != momenta_columns + ["signal"]:
        raise ValueError(f"{events_file} does not have the expected columns pt0, eta0, phi0, ..., signal")
    events = df[momenta_columns].to_numpy(dtype=np.double).reshape(len(df), N_PARTICLES_PER_EVENT, 3)

    jet_def = fj.JetDefinition(fj.antikt_algorithm, 1.0)
    n = len(events)
    out = {
        "constituents": np.zeros((n, 2, n_constituents, 4)),
        "n_constituents": np.zeros((n, 2), dtype=np.int64),
        "jets_ptyphim": np.zeros((n, 2, 4)),
        "jets_ptetaphim": np.zeros((n, 2, 4)),
        "label": df["signal"].to_numpy(),
    }
    for i_event, event in enumerate(events):
        # Remove zero padding and build massless PseudoJets, as in data_prep.to_pseudo_jets
        particles = event[~np.all(event == 0, axis=1)]
        pseudo_jets = []
        for pt, eta, phi in particles:
            pseudo_jet = fj.PseudoJet()
            pseudo_jet.reset_PtYPhiM(pt, eta, phi, np.double(0))
            pseudo_jets.append(pseudo_jet)
        jets = jet_def(pseudo_jets)
        # Two leading-pT inclusive jets, as in jet_clustering.event_to_jet_cluster
        index = np.argsort([jet.pt() for jet in jets])[::-1]
        for i_jet, jet in enumerate([jets[index[0]], jets[index[1]]]):
            constituents = fj.sorted_by_pt(jet.constituents())
            kept = np.array([[c.pt(), c.eta(), c.phi(), c.m()] for c in constituents[:n_constituents]])
            out["constituents"][i_event, i_jet, : len(kept)] = kept
            out["n_constituents"][i_event, i_jet] = len(constituents)
            out["jets_ptyphim"][i_event, i_jet] = [jet.pt(), jet.rap(), jet.phi(), jet.m()]
            out["jets_ptetaphim"][i_event, i_jet] = [jet.pt(), jet.eta(), jet.phi(), jet.m()]
    return out


def cluster_events(events_file, cache_file, n_constituents=100, chunk_size=1000, n_jobs=4):
    """Cluster all events of events_file, reusing cache_file if it exists."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        print(f"Using cached clustering: {cache_file}")
    else:
        with pd.HDFStore(events_file, mode="r") as store:
            n_events = store.get_storer(store.keys()[0]).nrows
        print(f"Clustering {n_events} events from {events_file} with {n_jobs} processes...")
        chunks = [(str(events_file), s, min(s + chunk_size, n_events), n_constituents) for s in range(0, n_events, chunk_size)]
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(cluster_events_chunk, chunks)
        tmp_file = cache_file.with_name(cache_file.name + ".tmp")
        with h5py.File(tmp_file, "w") as f:
            for key in results[0]:
                f.create_dataset(key, data=np.concatenate([r[key] for r in results]), compression="gzip")
            f.attrs["events_file"] = str(events_file)
            f.attrs["jet_algorithm"] = "antikt"
            f.attrs["jet_radius"] = 1.0
            f.attrs["n_constituents_per_jet"] = n_constituents
            f.attrs["constituent_order"] = "pt,eta,phi,mass; phi in [0, 2pi)"
            f.attrs["kitchensink_commit"] = KITCHENSINK_COMMIT
        tmp_file.replace(cache_file)
        print(f"  ✓ Saved clustering to: {cache_file}")

    with h5py.File(cache_file, "r") as f:
        if f.attrs["n_constituents_per_jet"] != n_constituents:
            raise ValueError(f"{cache_file} was made with {f.attrs['n_constituents_per_jet']} constituents per jet")
        return {key: f[key][:] for key in f}


def mjj_from_jets(jets_ptyphim):
    """Dijet invariant mass from (pt, y, phi, m) of the two jets, as in KitchenSink hlf.py."""
    pt, y, phi, m = (jets_ptyphim[..., i] for i in range(4))
    mt = np.sqrt(pt**2 + m**2)
    energy, px, py, pz = mt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)
    return np.sqrt(energy.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2)


def wrap_phi(phi):
    """Map phi from [0, 2pi) to [-pi, pi)."""
    return np.where(phi >= np.pi, phi - 2 * np.pi, phi)


def split_signal_region(sr_indices, n_train, n_test, seed):
    """Shuffle signal-region event indices and split them into train and test (remaining events unused)."""
    shuffled = np.random.default_rng(seed).permutation(sr_indices)
    return shuffled[:n_train], shuffled[n_train : n_train + n_test]


def write_split(output_file, clustered, indices, mjj, attrs):
    """Write the selected events in the schema of the existing sn_XtoYY_*_SR_*.h5 files."""
    n_events = len(indices)
    n_constituents = clustered["constituents"].shape[2]
    tmp_file = output_file.with_name(output_file.name + ".tmp")
    with h5py.File(tmp_file, "w") as f:
        for key, value in attrs.items():
            f.attrs[key] = value
        create = lambda name, data: f.create_dataset(
            name, data=data, chunks=(min(CHUNK_EVENTS, n_events),) + data.shape[1:], compression="gzip"
        )
        for i_jet, jet_name in enumerate(["jet1", "jet2"]):
            constituents = clustered["constituents"][indices, i_jet].copy()
            mask = (np.arange(n_constituents)[None, :] < clustered["n_constituents"][indices, i_jet, None])
            constituents[..., 2] = np.where(mask, wrap_phi(constituents[..., 2]), 0.0)
            create(f"{jet_name}/4mom", constituents)
            create(f"{jet_name}/mask", mask[..., None].astype(np.int8))
        jet_coords = clustered["jets_ptetaphim"][indices].copy()
        jet_coords[..., 2] = wrap_phi(jet_coords[..., 2])
        create("jet_coords", jet_coords)
        create("mjj", mjj[indices])
        create("signal", clustered["label"][indices].astype(np.int8))
        create("source_event_index", indices.astype(np.int64))
    tmp_file.replace(output_file)
    print(f"  ✓ Saved {n_events} events to: {output_file}")


def prepare_kitchensink_signal(
    model, events_file, output_dir, n_train=10000, n_test=20000, n_constituents=100,
    mjj_min=3300.0, mjj_max=3700.0, seed=42, n_jobs=4, overwrite=False,
):
    """
    Cluster a KitchenSink signal sample, select the signal region and write train/test files.

    Args:
        model: KitchenSink model name (e.g. XtoWRto3W), used in output names and attrs
        events_file: Path to the model's events.h5 from its Zenodo tarball
        output_dir: Directory for sn_<model>_<N>_SR_{train,test}.h5
        n_train, n_test: Number of signal-region events per split
        n_constituents: Number of leading-pT constituents stored per jet
        mjj_min, mjj_max: Signal-region window in GeV
        seed: Seed of the train/test shuffle
        n_jobs: Number of clustering processes
        overwrite: Replace existing output files
    """
    events_file, output_dir = Path(events_file), Path(output_dir)
    clustered = cluster_events(events_file, events_file.parent / f"clustered_{n_constituents}.h5", n_constituents, n_jobs=n_jobs)
    if not np.all(clustered["label"] == 1):
        raise ValueError(f"{events_file} contains events with signal != 1")

    mjj = mjj_from_jets(clustered["jets_ptyphim"])
    sr_indices = np.flatnonzero((mjj >= mjj_min) & (mjj <= mjj_max))
    print(f"Signal region {mjj_min:.0f} <= mjj <= {mjj_max:.0f} GeV: {len(sr_indices)} of {len(mjj)} events")
    if n_train + n_test > len(sr_indices):
        raise ValueError(f"Requested {n_train} train + {n_test} test events, but only {len(sr_indices)} are in the signal region")
    train_indices, test_indices = split_signal_region(sr_indices, n_train, n_test, seed)

    fmt = lambda n: f"{n // 1000}k" if n % 1000 == 0 else str(n)
    for split, indices in [("train", train_indices), ("test", test_indices)]:
        output_file = output_dir / f"sn_{model}_{fmt(len(indices))}_SR_{split}.h5"
        if output_file.exists() and not overwrite:
            raise FileExistsError(f"{output_file} exists. Pass --overwrite to replace it.")
        attrs = {
            "coordinate_order": "pt,eta,phi,mass",
            "mom4_format": "ptetaphim",
            "phi_range": "[-pi, pi)",
            "jet_algorithm": "antikt",
            "jet_radius": 1.0,
            "n_constituents_per_jet": n_constituents,
            "jet_coords": "full-jet (pt, eta, phi in [-pi, pi), m) before constituent truncation",
            "mass_window_applied": True,
            "mjj_min_gev": mjj_min,
            "mjj_max_gev": mjj_max,
            "sample_label": 1,
            "source_project": model,
            "source_events_file": str(events_file),
            "kitchensink_repository": "https://github.com/lang-lukas/KitchenSink",
            "kitchensink_commit": KITCHENSINK_COMMIT,
            "zenodo_record": "https://zenodo.org/records/18983506",
            "split": split,
            "split_seed": seed,
            "split_strategy": "random",
            "unused_sr_events": len(sr_indices) - n_train - n_test,
            "preparation_script": "src/data/prepare_kitchensink_signal.py",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_split(output_file, clustered, indices, mjj, attrs)


def main():
    kitchensink_dir = Path("/.automount/net_rw/net__data_ttk/soshaw/KitchenSink")
    parser = argparse.ArgumentParser(description="Prepare a KitchenSink signal sample as signal-region train/test HDF5 files")
    parser.add_argument("--model", type=str, required=True, help="KitchenSink model name, e.g. XtoWRto3W")
    parser.add_argument("--events_file", type=str, default=None, help="Path to events.h5 (default: KitchenSink/raw/<model>/events.h5)")
    parser.add_argument("--output_dir", type=str, default=str(kitchensink_dir), help="Output directory")
    parser.add_argument("--n_train", type=int, default=10000, help="Signal-region events in the train file (default: 10000)")
    parser.add_argument("--n_test", type=int, default=20000, help="Signal-region events in the test file (default: 20000)")
    parser.add_argument("--n_constituents", type=int, default=100, help="Constituents stored per jet (default: 100)")
    parser.add_argument("--mjj_min", type=float, default=3300.0, help="Lower signal-region edge in GeV (default: 3300)")
    parser.add_argument("--mjj_max", type=float, default=3700.0, help="Upper signal-region edge in GeV (default: 3700)")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the train/test shuffle (default: 42)")
    parser.add_argument("--jobs", type=int, default=4, help="Number of clustering processes (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    events_file = args.events_file or str(kitchensink_dir / "raw" / args.model / "events.h5")
    prepare_kitchensink_signal(
        model=args.model, events_file=events_file, output_dir=args.output_dir, n_train=args.n_train,
        n_test=args.n_test, n_constituents=args.n_constituents, mjj_min=args.mjj_min, mjj_max=args.mjj_max,
        seed=args.seed, n_jobs=args.jobs, overwrite=args.overwrite,
    )
    print(f"\n✅ Successfully prepared {args.model}!")


if __name__ == "__main__":
    main()
