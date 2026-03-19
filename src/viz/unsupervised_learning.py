""" Unsupervised learning & dimensionality reduction analysis for signal vs background jets.

PROJECT STRUCTURE:
AnomalyDetection/
├── src/
│   ├── train/              # Training scripts
│   ├── eval/               # Evaluation scripts
│   ├── data/               # Data processing
│   ├── viz/                # Visualization (e.g., unsupervised_learning.py)
│   └── poc_expts/          # Proof-of-concept experiments
├── scripts/                # Job launchers
├── gabbro/                 # Core library (data utilities)
└── [output directories: plots/]

FEATURES:
├── Load signal and background data
├── Apply t-SNE for dimensionality reduction
├── Apply UMAP for comparison
├── Compute silhouette scores
├── Visualize clustering patterns
└── Pre-training analysis before model training

USAGE: python -m src.viz.unsupervised_learning --checkpoint path/to/checkpoint.ckpt [args]
Applies multiple dimensionality reduction techniques to check for emergent clustering patterns.
"""

import os
import argparse

import awkward as ak
import matplotlib
import numpy as np
import torch
from gabbro.data.loading import load_multiple_h5_files
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inspect_loaded_data(all_features, all_labels):
    print("Features container type:", type(all_features))
    print("Labels container type:", type(all_labels))

    if isinstance(all_features, ak.Array):
        print("Number of jets:", len(all_features))
        print("Feature fields:", list(all_features.fields))
        for field in all_features.fields:
            lengths = ak.to_numpy(ak.num(all_features[field]))
            print(
                f"  {field}: min_len={int(lengths.min())}, "
                f"max_len={int(lengths.max())}, mean_len={float(lengths.mean()):.2f}"
            )
    else:
        features_np = np.asarray(all_features)
        print("Features shape:", features_np.shape, "dtype:", features_np.dtype)

    labels_np = np.asarray(all_labels)
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    print("Labels shape:", labels_np.shape, "dtype:", labels_np.dtype)
    print("Label counts:", dict(zip(unique_labels.tolist(), counts.tolist())))


def parse_loader_output(loader_output):
    if not isinstance(loader_output, tuple):
        raise TypeError(f"Unexpected loader output type: {type(loader_output)}")

    if len(loader_output) == 2:
        all_features, all_labels = loader_output
        return all_features, None, all_labels

    if len(loader_output) == 3:
        all_features_jet1, all_features_jet2, all_labels = loader_output
        return all_features_jet1, all_features_jet2, all_labels

    raise ValueError(f"Unexpected number of outputs from loader: {len(loader_output)}")


def awkward_features_to_matrix(features_ak, max_particles=128):
    if not isinstance(features_ak, ak.Array):
        features_np = np.asarray(features_ak)
        if features_np.ndim > 2:
            return features_np.reshape(features_np.shape[0], -1)
        return features_np

    field_mats = []
    for field in features_ak.fields:
        padded = ak.pad_none(features_ak[field], max_particles, clip=True)
        filled = ak.fill_none(padded, 0.0)
        field_mats.append(ak.to_numpy(filled).astype(np.float32))

    return np.concatenate(field_mats, axis=1)


def stratified_sample_indices(labels, max_points, seed=42):
    n_total = len(labels)
    if n_total <= max_points:
        return np.arange(n_total)

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    sampled = []

    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        n_cls = max(1, int(round(max_points * len(cls_idx) / n_total)))
        n_cls = min(n_cls, len(cls_idx))
        sampled.append(rng.choice(cls_idx, size=n_cls, replace=False))

    sampled = np.concatenate(sampled)
    if len(sampled) > max_points:
        sampled = rng.choice(sampled, size=max_points, replace=False)
    elif len(sampled) < max_points:
        remaining = np.setdiff1d(np.arange(n_total), sampled, assume_unique=False)
        extra = rng.choice(remaining, size=max_points - len(sampled), replace=False)
        sampled = np.concatenate([sampled, extra])

    return np.sort(sampled)


def plot_embeddings(emb_tsne, emb_umap, labels, out_path):
    unique_labels = np.unique(labels)
    cmap = matplotlib.colormaps["tab10"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    plots = [(axes[0], emb_tsne, "t-SNE"), (axes[1], emb_umap, "UMAP")]

    for ax, emb, title in plots:
        for i, cls in enumerate(unique_labels):
            mask = labels == cls
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=8,
                alpha=0.7,
                c=[cmap(i % cmap.N)],
                label=f"label {cls}",
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.legend(loc="best", markerscale=2)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved embedding figure to: {out_path}")


def safe_silhouette_score(x, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan
    if len(labels) <= len(unique_labels):
        return np.nan

    try:
        return float(silhouette_score(x, labels))
    except Exception:
        return np.nan

def main():
    parser = argparse.ArgumentParser(description="LHCO unsupervised embedding visualization")
    parser.add_argument("--n-signal", type=int, default=25000)
    parser.add_argument("--n-background", type=int, default=200000)
    parser.add_argument("--jet-name", type=str, default="jet1")
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--max-embed-points", type=int, default=20000)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_features_dict = {
        "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "signed_log", "inv_func": "signed_exp"},
        "part_etarel": {"multiply_by": 3},
        "part_phirel": {"multiply_by": 3}
    }

    signal_path = os.path.join("/.automount/net_rw/net__data_ttk/soshaw", "sn_25k_SR_train.h5")
    background_path = os.path.join("/.automount/net_rw/net__data_ttk/soshaw", "bg_200k_SR_train.h5")
    
    h5_files_train = [signal_path, background_path]

    n_jets_train = [args.n_signal, args.n_background]  # [signal, background]
    jet_name = args.jet_name

    print("n_jets_train:", n_jets_train)
    print("Using Jet:", jet_name)

    loader_output = load_multiple_h5_files(
                h5_files_train,
                input_features_dict,
                n_jets_per_file=n_jets_train,
                mom4_format="epxpypz",
                jet_name=jet_name,
            )

    all_features_jet1, all_features_jet2, all_labels = parse_loader_output(loader_output)

    if jet_name == "both":
        print("Detected dijet mode (jet_name='both').")
        print("Jet1 feature summary:")
        inspect_loaded_data(all_features_jet1, all_labels)
        print("Jet2 feature summary:")
        inspect_loaded_data(all_features_jet2, all_labels)
    else:
        inspect_loaded_data(all_features_jet1, all_labels)

    labels_np = np.asarray(all_labels)
    jet1_mat = awkward_features_to_matrix(all_features_jet1, max_particles=args.max_particles)
    if all_features_jet2 is not None:
        jet2_mat = awkward_features_to_matrix(all_features_jet2, max_particles=args.max_particles)
        features_2d = np.concatenate([jet1_mat, jet2_mat], axis=1)
        print("Combined dijet feature matrix from jet1+jet2")
    else:
        features_2d = jet1_mat
    print("Flattened feature matrix shape:", features_2d.shape)

    sample_idx = stratified_sample_indices(labels_np, max_points=args.max_embed_points, seed=42)
    x_plot = features_2d[sample_idx]
    y_plot = labels_np[sample_idx]
    print(f"Using {len(sample_idx)} jets for embeddings")

    tsne_perplexity = max(5, min(30, (len(x_plot) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=tsne_perplexity,
    )
    emb_tsne = tsne.fit_transform(x_plot)

    umap_model = UMAP(n_components=2, random_state=42)
    emb_umap = umap_model.fit_transform(x_plot)

    sil_input = safe_silhouette_score(x_plot, y_plot)
    sil_tsne = safe_silhouette_score(emb_tsne, y_plot)
    sil_umap = safe_silhouette_score(emb_umap, y_plot)

    print(f"Silhouette (input feature space): {sil_input:.4f}" if not np.isnan(sil_input) else "Silhouette (input feature space): n/a")
    print(f"Silhouette (t-SNE embedding): {sil_tsne:.4f}" if not np.isnan(sil_tsne) else "Silhouette (t-SNE embedding): n/a")
    print(f"Silhouette (UMAP embedding): {sil_umap:.4f}" if not np.isnan(sil_umap) else "Silhouette (UMAP embedding): n/a")

    output_path = args.output
    if not output_path:
        output_path = f"plots/lhco_tsne_umap_labels_{jet_name}.png"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_embeddings(emb_tsne, emb_umap, y_plot, output_path)

if __name__ == "__main__":
    main()