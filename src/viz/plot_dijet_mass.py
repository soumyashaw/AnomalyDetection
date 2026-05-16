"""
Visualization functions for dijet invariant mass distributions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Optional, Tuple, Union


def load_dijet_mass_from_h5(
    h5_filename: str,
    n_jets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dijet invariant mass directly from LHCO HDF5 file.
    
    Parameters
    ----------
    h5_filename : str
        Path to HDF5 file containing dijet data.
    n_jets : int, optional
        Number of jets to load. If None, load all jets.
        
    Returns
    -------
    mjj : np.ndarray
        Array of dijet invariant masses in GeV.
    labels : np.ndarray
        Array of labels (0=background, 1=signal).
    """
    with h5py.File(h5_filename, "r") as f:
        # Load dijet invariant mass from jet_features dataset
        # jet_features: (n_events, 7) - (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)
        if n_jets is None:
            mjj = f["jet_features"][:, 6]  # mjj is at index 6
            labels = f["signal"][:]
        else:
            mjj = f["jet_features"][:n_jets, 6]
            labels = f["signal"][:n_jets]
    
    return mjj, labels


def plot_dijet_mass_distribution(
    h5_filename: Optional[str] = None,
    background_file: Optional[str] = None,
    signal_file: Optional[str] = None,
    n_jets: Optional[int] = None,
    bandwidth: float = 50.0,
    xlim: Tuple[float, float] = (1500, 7000),
    ylim: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    show_signal_region: bool = False,
    signal_region: Tuple[float, float] = (3300, 3700),
    show_background_region: bool = False,
    background_region: Tuple[float, float] = (2900, 4100),
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot dijet invariant mass distribution with KDE smoothing.
    
    Parameters
    ----------
    h5_filename : str, optional
        Path to single HDF5 file containing both background and signal data 
        (distinguished by labels). If provided, background_file and signal_file 
        are ignored.
    background_file : str, optional
        Path to HDF5 file containing background data only.
    signal_file : str, optional
        Path to HDF5 file containing signal data only.
    n_jets : int, optional
        Number of jets to load from each file. If None, load all jets.
    bandwidth : float, optional
        Bandwidth for KDE smoothing in GeV. Default: 50.0.
    xlim : tuple, optional
        X-axis limits (min_mass, max_mass) in GeV. Default: (1500, 7000).
    ylim : tuple, optional
        Y-axis limits (min_density, max_density). If None, auto-scales to fit data.
        Use this to prevent signal peaks from dominating the scale.
        Example: ylim=(0, 0.001) to limit maximum density display.
    normalize : bool, optional
        If True, KDE is normalized so area under curve = 1 (probability density).
        If False, scale by number of data points to show approximate counts.
        Default: True.
    show_signal_region : bool, optional
        If True, draw vertical lines marking the signal region. Default: False.
    signal_region : tuple, optional
        Signal region boundaries (min_mass, max_mass) in GeV. Default: (3300, 3700).
    show_background_region : bool, optional
        If True, draw vertical lines marking the background region. Default: False.
    background_region : tuple, optional
        Background region boundaries (min_mass, max_mass) in GeV. Default: (2900, 4100).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (10, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, optional
        Whether to display the figure. Default: True.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
        
    Notes
    -----
    Either h5_filename OR at least one of (background_file, signal_file) must be provided.
    If h5_filename is provided, it takes precedence and the file should contain both
    background (label=0) and signal (label=1) data.
    """
    if h5_filename is None and background_file is None and signal_file is None:
        raise ValueError("Either h5_filename or at least one of (background_file, signal_file) must be provided")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create evaluation points for KDE
    x_eval = np.linspace(xlim[0], xlim[1], 1000)
    
    # Case 1: Single file with both background and signal (distinguished by labels)
    if h5_filename is not None:
        mjj, labels = load_dijet_mass_from_h5(h5_filename, n_jets=n_jets)
        mjj_background = mjj[labels == 0]
        # For signal, use all data (labels == 0 OR labels == 1)
        mjj_signal = mjj  # All data regardless of label
        
        # Plot background
        if len(mjj_background) > 0:
            kde_bg = gaussian_kde(mjj_background, bw_method=bandwidth / mjj_background.std())
            density_bg = kde_bg(x_eval)
            if not normalize:
                density_bg = density_bg * len(mjj_background)
            ax.fill_between(x_eval, density_bg, alpha=0.7, color='cyan', 
                            label='background', linewidth=2, edgecolor='darkcyan')
        
        # Plot signal (all data)
        if len(mjj_signal) > 0:
            kde_sig = gaussian_kde(mjj_signal, bw_method=bandwidth / mjj_signal.std())
            density_sig = kde_sig(x_eval)
            if not normalize:
                density_sig = density_sig * len(mjj_signal)
            ax.fill_between(x_eval, density_sig, alpha=0.7, color='darkred',
                            label='signal', linewidth=2, edgecolor='maroon')
    
    # Case 2: Separate files for background and signal
    else:
        # Plot background distribution
        if background_file is not None:
            mjj_background, _ = load_dijet_mass_from_h5(background_file, n_jets=n_jets)
            if len(mjj_background) > 0:
                kde_bg = gaussian_kde(mjj_background, bw_method=bandwidth / mjj_background.std())
                density_bg = kde_bg(x_eval)
                if not normalize:
                    density_bg = density_bg * len(mjj_background)
                ax.fill_between(x_eval, density_bg, alpha=0.7, color='cyan', 
                                label='background', linewidth=2, edgecolor='darkcyan')
        
        # Plot signal distribution
        if signal_file is not None:
            mjj_signal, _ = load_dijet_mass_from_h5(signal_file, n_jets=n_jets)
            if len(mjj_signal) > 0:
                kde_sig = gaussian_kde(mjj_signal, bw_method=bandwidth / mjj_signal.std())
                density_sig = kde_sig(x_eval)
                if not normalize:
                    density_sig = density_sig * len(mjj_signal)
                ax.fill_between(x_eval, density_sig, alpha=0.7, color='darkred',
                                label='signal', linewidth=2, edgecolor='maroon')
    
    # Formatting
    ax.set_xlabel(r'$M_{JJ}$ (GeV)', fontsize=14)
    ylabel = 'Counts' if not normalize else 'Density'
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title('Dijet invariant mass', fontsize=16)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add background region markers if requested (plot first so signal region is on top)
    if show_background_region:
        ymin, ymax = ax.get_ylim()
        ax.axvline(background_region[0], color='blue', linestyle='--', linewidth=2, alpha=0.5, zorder=2)
        ax.axvline(background_region[1], color='blue', linestyle='--', linewidth=2, alpha=0.5, zorder=2)
        # Add shaded regions for side bands
        ax.axvspan(xlim[0], background_region[0], alpha=0.03, color='blue', zorder=1)
        ax.axvspan(background_region[1], xlim[1], alpha=0.03, color='blue', zorder=1)
    
    # Add signal region markers if requested (plot on top)
    if show_signal_region:
        ymin, ymax = ax.get_ylim()
        ax.axvline(signal_region[0], color='blue', linestyle='--', linewidth=2, alpha=0.9, zorder=3)
        ax.axvline(signal_region[1], color='blue', linestyle='--', linewidth=2, alpha=0.9, zorder=3)
        # Optionally add shaded region
        ax.axvspan(signal_region[0], signal_region[1], alpha=0.15, color='blue', label='signal region', zorder=2)
    
    if show_background_region:
        # Add background region to legend
        from matplotlib.patches import Patch
        legend_elements = ax.get_legend_handles_labels()
        if legend_elements[0]:  # If there are existing legend items
            handles, labels = legend_elements
            handles.append(Patch(facecolor='blue', alpha=0.03, label='side bands'))
            labels.append('side bands')
            ax.legend(handles=handles, labels=labels, fontsize=12, loc='upper right')
        else:
            ax.legend(fontsize=12, loc='upper right')
    else:
        ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def plot_dijet_mass_comparison(
    h5_filenames: list,
    labels_list: list,
    n_jets: Optional[int] = None,
    bandwidth: float = 50.0,
    xlim: Tuple[float, float] = (1500, 7000),
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    colors: Optional[list] = None,
) -> plt.Figure:
    """Plot and compare dijet invariant mass distributions from multiple files.
    
    Parameters
    ----------
    h5_filenames : list of str
        Paths to HDF5 files containing dijet data.
    labels_list : list of str
        Labels for each file (for legend).
    n_jets : int, optional
        Number of jets to load from each file. If None, load all jets.
    bandwidth : float, optional
        Bandwidth for KDE smoothing in GeV. Default: 50.0.
    xlim : tuple, optional
        X-axis limits (min_mass, max_mass) in GeV. Default: (1500, 7000).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (10, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, optional
        Whether to display the figure. Default: True.
    colors : list, optional
        List of colors for each distribution. If None, use default colors.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(h5_filenames)))
    
    # Create evaluation points for KDE
    x_eval = np.linspace(xlim[0], xlim[1], 1000)
    
    # Plot each distribution
    for h5_filename, label, color in zip(h5_filenames, labels_list, colors):
        # Load data
        mjj, _ = load_dijet_mass_from_h5(h5_filename, n_jets=n_jets)
        
        # Compute KDE
        if len(mjj) > 0:
            kde = gaussian_kde(mjj, bw_method=bandwidth / mjj.std())
            density = kde(x_eval)
            ax.plot(x_eval, density, label=label, linewidth=2, color=color, alpha=0.8)
    
    # Formatting
    ax.set_xlabel(r'$M_{JJ}$ (GeV)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Dijet invariant mass comparison', fontsize=16)
    ax.set_xlim(xlim)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot dijet invariant mass distributions",
        epilog="Example: python plot_dijet_mass.py --file combined.h5  OR  "
               "python plot_dijet_mass.py --background bg.h5 --signal sig.h5"
    )
    parser.add_argument("--file", type=str, default=None, 
                       help="Path to single HDF5 file with both background and signal (labeled)")
    parser.add_argument("--background", type=str, default=None, 
                       help="Path to background HDF5 file")
    parser.add_argument("--signal", type=str, default=None, 
                       help="Path to signal HDF5 file")
    parser.add_argument("--n_jets", type=int, default=None, 
                       help="Number of jets to load from each file")
    parser.add_argument("--bandwidth", type=float, default=50.0, 
                       help="KDE bandwidth in GeV")
    parser.add_argument("--xlim", type=float, nargs=2, default=(1500, 7000),
                       help="X-axis limits (min max) in GeV")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, 
                       help="Y-axis limits (min max) for density. Use to prevent signal peak from dominating scale")
    parser.add_argument("--no-normalize", action="store_true", 
                       help="Don't normalize to unit area (show counts instead of density)")
    parser.add_argument("--show-signal-region", action="store_true",
                       help="Show vertical lines marking the signal region (3300-3700 GeV)")
    parser.add_argument("--signal-region", type=float, nargs=2, default=[3300, 3700],
                       help="Signal region boundaries (min max) in GeV")
    parser.add_argument("--show-background-region", action="store_true",
                       help="Show vertical lines marking the background region (2900-4100 GeV)")
    parser.add_argument("--background-region", type=float, nargs=2, default=[2900, 4100],
                       help="Background region boundaries (min max) in GeV")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure")
    parser.add_argument("--no-show", action="store_true", help="Don't display the figure")
    
    args = parser.parse_args()
    
    if args.file is None and args.background is None and args.signal is None:
        parser.error("Either --file or at least one of --background/--signal must be provided")
    
    plot_dijet_mass_distribution(
        h5_filename=args.file,
        background_file=args.background,
        signal_file=args.signal,
        n_jets=args.n_jets,
        bandwidth=args.bandwidth,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        normalize=not args.no_normalize,
        show_signal_region=args.show_signal_region,
        signal_region=tuple(args.signal_region),
        show_background_region=args.show_background_region,
        background_region=tuple(args.background_region),
        save_path=args.save,
        show=not args.no_show,
    )
