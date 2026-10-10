#!/usr/bin/env python3
""" Script to convert KitchenSink signal files to the HDF5 schema of the KitchenSink/LHCO background files.

PROJECT STRUCTURE:
AnomalyDetection/
├── src/
│   ├── train/              # Training scripts
│   ├── eval/               # Evaluation scripts
│   ├── data/               # Data processing (e.g., convert_kitchensink_signal.py)
│   ├── viz/                # Visualization
│   └── poc_expts/          # Proof-of-concept experiments
├── scripts/                # Job launchers
└── [output directories: data/]

WHY:
The KitchenSink signal files store constituents as (pt, eta, phi, mass) in float64 and have no
precomputed features. The background files store (E, px, py, pz) in float32 plus precomputed
features. load_lhco_jets_from_h5 therefore parsed the signal as (E, px, py, pz), which gave every
signal constituent eta_rel = 0 and made signal trivially separable from background.

OUTPUT SCHEMA (identical to KitchenSink/bg_*.h5):
├── jet{1,2}/4mom        (N, 100, 4) float32   E, px, py, pz
├── jet{1,2}/coords      (N, 100, 2) float32   absolute eta, phi
├── jet{1,2}/features    (N, 100, 9) float32   eta_rel, phi_rel, pt, E, pt/pt_jet, E/E_jet,
│                                              log(pt/pt_jet), log(E/E_jet), deltaR
├── jet{1,2}/mask        (N, 100, 1) int8
├── jet_coords           (N, 2, 4)   float32   jet pt, eta, phi in [0, 2pi), m
├── jet_features         (N, 7)      float32   tau1-3 of jet1 and jet2 (NaN, not in source), mjj
├── signal               (N,)        int8
└── source_event_index   (N,)        int64

JET AXIS:
The background's jet_coords is the full jet before truncation to 100 constituents. The signal file
only contains the stored constituents, so the jet axis here is their 4-momentum sum. This is exact
for untruncated jets (~96% of signal jets) and approximate for jets that hit the 100-constituent cap.

USAGE: python -m src.data.convert_kitchensink_signal [--input_files a.h5 b.h5] [--suffix _epxpypz]
Writes <input stem><suffix>.h5 next to each input file. Input files are not modified.
"""

import h5py
import numpy as np
import argparse
from datetime import datetime, timezone
from pathlib import Path

PARTICLE_FEATURE_ORDER = "eta_rel,phi_rel,pt,E,pt/pt_jet,E/E_jet,log(pt/pt_jet),log(E/E_jet),deltaR"
CHUNK_EVENTS = 256  # same chunking as the background files


def wrap_phi(dphi):
    """Wrap angle differences to [-pi, pi)."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def constituents_to_lhco_schema(pt, eta, phi, mass, mask):
    """Build background-style per-jet arrays from (pt, eta, phi, mass) constituents.

    Args:
        pt, eta, phi, mass: (N, n_particles) arrays, padding entries are ignored
        mask: (N, n_particles) boolean array, True for real constituents

    Returns:
        dict with "4mom" (N, P, 4), "coords" (N, P, 2), "features" (N, P, 9) as float64
        (padding set to 0) and "jet_coords" (N, 4) as (pt, eta, phi in [0, 2pi), m).
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    px, py, pz, energy = (np.where(mask, a, 0.0) for a in (px, py, pz, energy))

    # Jet 4-vector = sum of stored constituents
    jet_e, jet_px, jet_py, jet_pz = (a.sum(axis=1) for a in (energy, px, py, pz))
    jet_pt = np.hypot(jet_px, jet_py)
    jet_eta = np.arcsinh(jet_pz / jet_pt)
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_m = np.sqrt(np.maximum(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0.0))

    eta_rel = eta - jet_eta[:, None]
    phi_rel = wrap_phi(phi - jet_phi[:, None])
    pt_rel = pt / jet_pt[:, None]
    e_rel = energy / jet_e[:, None]
    features = np.stack(
        [
            eta_rel,
            phi_rel,
            pt,
            energy,
            pt_rel,
            e_rel,
            np.log(np.where(mask, pt_rel, 1.0)),
            np.log(np.where(mask, e_rel, 1.0)),
            np.hypot(eta_rel, phi_rel),
        ],
        axis=-1,
    )

    return {
        "4mom": np.stack([energy, px, py, pz], axis=-1),
        "coords": np.where(mask[..., None], np.stack([eta, phi], axis=-1), 0.0),
        "features": np.where(mask[..., None], features, 0.0),
        "jet_coords": np.stack([jet_pt, jet_eta, np.mod(jet_phi, 2 * np.pi), jet_m], axis=-1),
    }


def convert_kitchensink_signal(input_file, output_file, overwrite=False):
    """
    Convert one KitchenSink signal file from (pt, eta, phi, mass) to the background schema.

    Args:
        input_file: Path to KitchenSink signal HDF5 file (mom4_format 'ptetaphim')
        output_file: Path of the converted HDF5 file
        overwrite: Replace output_file if it already exists
    """
    input_file, output_file = Path(input_file), Path(output_file)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists. Pass --overwrite to replace it.")

    print(f"Reading from: {input_file}")
    with h5py.File(input_file, "r") as f_in:
        source_format = f_in.attrs.get("mom4_format")
        if source_format != "ptetaphim":
            raise ValueError(
                f"{input_file} has mom4_format={source_format!r}, expected 'ptetaphim'. "
                "This script only converts KitchenSink signal files."
            )

        n_events = f_in["signal"].shape[0]
        jets = {}
        masks = {}
        for jet_name in ["jet1", "jet2"]:
            mom4 = f_in[f"{jet_name}/4mom"][:].astype(np.float64)
            masks[jet_name] = f_in[f"{jet_name}/mask"][:]
            jets[jet_name] = constituents_to_lhco_schema(
                pt=mom4[..., 0],
                eta=mom4[..., 1],
                phi=mom4[..., 2],
                mass=mom4[..., 3],
                mask=masks[jet_name][..., 0].astype(bool),
            )

        # tau1-3 are not stored in the signal file; mjj is the full-jet value from the source
        jet_features = np.full((n_events, 7), np.nan)
        jet_features[:, 6] = f_in["mjj"][:]

        datasets = {
            "jet_coords": np.stack([jets["jet1"]["jet_coords"], jets["jet2"]["jet_coords"]], axis=1).astype(np.float32),
            "jet_features": jet_features.astype(np.float32),
            "signal": f_in["signal"][:].astype(np.int8),
            "source_event_index": f_in["source_event_index"][:].astype(np.int64),
        }
        for jet_name in ["jet1", "jet2"]:
            for key in ["4mom", "coords", "features"]:
                datasets[f"{jet_name}/{key}"] = jets[jet_name][key].astype(np.float32)
            datasets[f"{jet_name}/mask"] = masks[jet_name].astype(np.int8)

        attrs = dict(f_in.attrs)

    attrs.update(
        {
            "coordinate_order": "E,px,py,pz",
            "mom4_format": "epxpypz",
            "particle_feature_order": PARTICLE_FEATURE_ORDER,
            "pt_energy_features": "raw",
            "jet_axis": "sum of stored constituents (full-jet 4-vector not available in source file)",
            "jet_coords_phi_range": "[0, 2pi)",
            "jet_features_note": "tau1-3 not available in source file, stored as NaN; mjj copied from source 'mjj'",
            "converted_from": str(input_file),
            "converted_from_mom4_format": "ptetaphim",
            "conversion_script": "src/data/convert_kitchensink_signal.py",
            "converted_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    tmp_file = output_file.with_name(output_file.name + ".tmp")
    with h5py.File(tmp_file, "w") as f_out:
        for key, value in attrs.items():
            f_out.attrs[key] = value
        for name, data in datasets.items():
            f_out.create_dataset(
                name,
                data=data,
                chunks=(min(CHUNK_EVENTS, n_events),) + data.shape[1:],
                compression="gzip",
            )
    tmp_file.replace(output_file)
    print(f"  ✓ Saved {n_events} events to: {output_file}")


def main():
    kitchensink_dir = Path("/.automount/net_rw/net__data_ttk/soshaw/KitchenSink")
    parser = argparse.ArgumentParser(
        description="Convert KitchenSink signal files to the background (E, px, py, pz + features) schema"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=[
            str(kitchensink_dir / "sn_XtoYY_10k_SR_train.h5"),
            str(kitchensink_dir / "sn_XtoYY_20k_SR_test.h5"),
        ],
        help="KitchenSink signal HDF5 files to convert",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_epxpypz",
        help="Suffix appended to each input file stem for the output file (default: _epxpypz)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    for input_file in args.input_files:
        input_path = Path(input_file)
        output_file = input_path.with_name(f"{input_path.stem}{args.suffix}{input_path.suffix}")
        convert_kitchensink_signal(input_file, output_file, overwrite=args.overwrite)

    print(f"\n✅ Successfully converted {len(args.input_files)} file(s)!")


if __name__ == "__main__":
    main()
