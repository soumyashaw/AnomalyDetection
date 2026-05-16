"""
Visualization script for CASE dataset dijet invariant mass distributions.

This script creates plots similar to the CASE benchmark paper, showing:
- Background and signal dijet invariant mass distributions
- Signal region and sideband markers
- Normalized histograms or KDE smoothed distributions

Usage:
    # Single signal
    python -m src.viz.plot_case_dijet_mass --background /path/to/background.h5 --signal /path/to/signal.h5
    
    # Multiple signals
    python -m src.viz.plot_case_dijet_mass --background /path/to/background.h5 \
        --signal /path/to/signal1.h5 /path/to/signal2.h5 /path/to/signal3.h5 \
        --signal_labels "Q* 2 TeV" "Q* 3 TeV" "Q* 4 TeV" \
        --signal_colors "#8B0000" "#DC143C" "#FF6347"
    
    # Custom regions
    python -m src.viz.plot_case_dijet_mass --background /path/to/background.h5 \
        --signal /path/to/signal.h5 \
        --signal_region 2900 3100 \
        --sideband_region 2725 3331
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Optional, Tuple, List, Union
import argparse


def load_case_dijet_mass(
    h5_filename: str,
    n_jets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dijet invariant mass from CASE HDF5 file.
    
    The CASE dataset structure:
    - jet_kinematics: (N, 14) containing [m_jj, delta_eta, j1_pt, j1_eta, j1_phi, 
                      j1_mass, j2_pt, j2_eta, j2_phi, j2_mass, j3_pt, j3_eta, 
                      j3_phi, j3_mass]
    - truth_label: (N, 1) with 0=background, 1=signal
    
    Parameters
    ----------
    h5_filename : str
        Path to CASE HDF5 file.
    n_jets : int, optional
        Number of events to load. If None, load all events.
        
    Returns
    -------
    mjj : np.ndarray
        Array of dijet invariant masses in GeV.
    labels : np.ndarray
        Array of labels (0=background, 1=signal).
    """
    with h5py.File(h5_filename, "r") as f:
        # jet_kinematics: first column (index 0) is m_jj
        if n_jets is None:
            mjj = f["jet_kinematics"][:, 0]  # m_jj is at index 0
            labels = f["truth_label"][:].squeeze()  # Remove extra dimension
        else:
            mjj = f["jet_kinematics"][:n_jets, 0]
            labels = f["truth_label"][:n_jets].squeeze()
    
    # Handle negative labels (convert to 0)
    if np.any(labels < 0):
        n_negative = np.sum(labels < 0)
        print(f"Warning: Found {n_negative} negative labels in {os.path.basename(h5_filename)}, converting to 0 (background)")
        labels = np.clip(labels, 0, None)
    
    return mjj, labels.astype(int)


def plot_case_dijet_mass_distribution(
    background_file: Optional[str] = None,
    signal_file: Optional[Union[str, List[str]]] = None,
    signal_labels: Optional[List[str]] = None,
    signal_colors: Optional[List[str]] = None,
    n_jets: Optional[int] = None,
    use_kde: bool = True,
    bandwidth: float = 50.0,
    n_bins: int = 100,
    xlim: Tuple[float, float] = (1500, 7000),
    ylim: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    show_signal_region: bool = True,
    signal_region: Tuple[float, float] = (3300, 3700),
    show_sidebands: bool = True,
    sideband_region: Tuple[float, float] = (2725, 3331),
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """Plot CASE dataset dijet invariant mass distribution.
    
    Creates a publication-quality plot showing background and signal distributions
    with signal region and sideband markers, similar to CASE benchmark plots.
    
    Parameters
    ----------
    background_file : str, optional
        Path to HDF5 file containing background events.
    signal_file : str or list of str, optional
        Path(s) to HDF5 file(s) containing signal events. Can be a single string
        or a list of strings for multiple signal files.
    signal_labels : list of str, optional
        Labels for each signal file (for legend). If None, uses default labels
        like 'signal', 'signal 2', etc.
    signal_colors : list of str, optional
        Colors for each signal file. If None, uses default color palette.
    n_jets : int, optional
        Number of events to load from each file. If None, load all.
    use_kde : bool, optional
        If True, use KDE smoothing. If False, use histograms. Default: True.
    bandwidth : float, optional
        Bandwidth for KDE smoothing in GeV. Default: 50.0.
    n_bins : int, optional
        Number of bins for histogram (if not using KDE). Default: 100.
    xlim : tuple, optional
        X-axis limits (min_mass, max_mass) in GeV. Default: (1500, 7000).
    ylim : tuple, optional
        Y-axis limits. If None, auto-scales to fit data.
    normalize : bool, optional
        If True, normalize to show relative frequency. If False, show counts. Default: False.
    show_signal_region : bool, optional
        If True, mark the signal region. Default: True.
    signal_region : tuple, optional
        Signal region boundaries (min, max) in GeV. Default: (3300, 3700).
    show_sidebands : bool, optional
        If True, mark the sideband region. Default: True.
    sideband_region : tuple, optional
        Sideband region boundaries (lower, upper) in GeV. Default: (2725, 3331).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 6).
    save_path : str, optional
        Path to save the figure. If None, not saved.
    show : bool, optional
        Whether to display the figure. Default: True.
    dpi : int, optional
        DPI for saved figure. Default: 300.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    if background_file is None and signal_file is None:
        raise ValueError("At least one of (background_file, signal_file) must be provided")
    
    # Convert signal_file to list if it's a single string
    if signal_file is not None and isinstance(signal_file, str):
        signal_file = [signal_file]
    
    # Set up default labels and colors for signals
    if signal_file is not None:
        n_signals = len(signal_file)
        if signal_labels is None:
            if n_signals == 1:
                signal_labels = ['signal']
            else:
                signal_labels = [f'signal {i+1}' for i in range(n_signals)]
        
        if signal_colors is None:
            # Default color palette for multiple signals
            default_colors = ['#8B0000', '#DC143C', '#FF6347', '#FF4500', '#FF8C00']
            signal_colors = default_colors[:n_signals]
        
        # Validate lengths
        if len(signal_labels) != n_signals:
            raise ValueError(f"signal_labels length ({len(signal_labels)}) must match number of signal files ({n_signals})")
        if len(signal_colors) != n_signals:
            raise ValueError(f"signal_colors length ({len(signal_colors)}) must match number of signal files ({n_signals})")
    
    # Create figure with publication-quality settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load and plot data
    if use_kde:
        # KDE smoothing approach
        x_eval = np.linspace(xlim[0], xlim[1], 1000)
        
        # Plot background
        if background_file is not None:
            mjj_bg, labels_bg = load_case_dijet_mass(background_file, n_jets=n_jets)
            # Filter to only background events (label=0)
            mjj_bg = mjj_bg[labels_bg == 0]
            
            if len(mjj_bg) > 0:
                # Filter to xlim range
                mjj_bg = mjj_bg[(mjj_bg >= xlim[0]) & (mjj_bg <= xlim[1])]
                
                kde_bg = gaussian_kde(mjj_bg, bw_method=bandwidth / mjj_bg.std())
                density_bg = kde_bg(x_eval)
                
                if normalize:
                    # Normalize so integral over range equals 1
                    dx = x_eval[1] - x_eval[0]
                    density_bg = density_bg / (density_bg.sum() * dx)
                else:
                    density_bg = density_bg * len(mjj_bg)
                
                ax.fill_between(x_eval, density_bg, alpha=0.8, color='#20B2AA', 
                               label='background', linewidth=0, zorder=2)
                ax.plot(x_eval, density_bg, color='#008080', linewidth=1.5, alpha=0.9, zorder=3)
        
        # Plot signal(s)
        if signal_file is not None:
            for idx, (sig_file, sig_label, sig_color) in enumerate(zip(signal_file, signal_labels, signal_colors)):
                mjj_sig, labels_sig = load_case_dijet_mass(sig_file, n_jets=n_jets)
                # Filter to only signal events (label=1)
                mjj_sig = mjj_sig[labels_sig == 1]
                
                if len(mjj_sig) > 0:
                    # Filter to xlim range
                    mjj_sig = mjj_sig[(mjj_sig >= xlim[0]) & (mjj_sig <= xlim[1])]
                    
                    kde_sig = gaussian_kde(mjj_sig, bw_method=bandwidth / mjj_sig.std())
                    density_sig = kde_sig(x_eval)
                    
                    if normalize:
                        dx = x_eval[1] - x_eval[0]
                        density_sig = density_sig / (density_sig.sum() * dx)
                    else:
                        density_sig = density_sig * len(mjj_sig)
                    
                    # Lighter edge color for better visibility with multiple signals
                    edge_color = sig_color if idx == 0 else None
                    ax.fill_between(x_eval, density_sig, alpha=0.7, color=sig_color, 
                                   label=sig_label, linewidth=0, zorder=2 + idx)
                    if edge_color:
                        # Darken the edge color slightly
                        import matplotlib.colors as mcolors
                        rgb = mcolors.to_rgb(edge_color)
                        dark_rgb = tuple(max(0, c * 0.7) for c in rgb)
                        ax.plot(x_eval, density_sig, color=dark_rgb, linewidth=1.5, alpha=0.9, zorder=3 + idx)
    
    else:
        # Histogram approach
        bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
        
        # Plot background
        if background_file is not None:
            mjj_bg, labels_bg = load_case_dijet_mass(background_file, n_jets=n_jets)
            mjj_bg = mjj_bg[labels_bg == 0]
            
            if len(mjj_bg) > 0:
                weights_bg = np.ones(len(mjj_bg)) / len(mjj_bg) if normalize else None
                ax.hist(mjj_bg, bins=bins, weights=weights_bg, alpha=0.8, 
                       color='#20B2AA', label='background', edgecolor='#008080',
                       linewidth=1.0, zorder=2)
        
        # Plot signal(s)
        if signal_file is not None:
            for idx, (sig_file, sig_label, sig_color) in enumerate(zip(signal_file, signal_labels, signal_colors)):
                mjj_sig, labels_sig = load_case_dijet_mass(sig_file, n_jets=n_jets)
                mjj_sig = mjj_sig[labels_sig == 1]
                
                if len(mjj_sig) > 0:
                    weights_sig = np.ones(len(mjj_sig)) / len(mjj_sig) if normalize else None
                    # Darken edge color slightly
                    import matplotlib.colors as mcolors
                    rgb = mcolors.to_rgb(sig_color)
                    dark_rgb = tuple(max(0, c * 0.7) for c in rgb)
                    ax.hist(mjj_sig, bins=bins, weights=weights_sig, alpha=0.7,
                           color=sig_color, label=sig_label, 
                           edgecolor=mcolors.to_hex(dark_rgb),
                           linewidth=1.0, zorder=2 + idx)
    
    # Add sideband markers (plot first, lower z-order)
    if show_sidebands:
        ymin, ymax = ax.get_ylim()
        # Sideband region boundaries
        ax.axvline(sideband_region[0], color='#4B0082', linestyle='--', 
                  linewidth=2.5, alpha=0.9, zorder=4)
        ax.axvline(sideband_region[1], color='#4B0082', linestyle='--', 
                  linewidth=2.5, alpha=0.9, zorder=4, label='sideband region')
    
    # Add signal region markers (plot on top, higher z-order)
    if show_signal_region:
        ymin, ymax = ax.get_ylim()
        ax.axvline(signal_region[0], color='#1E90FF', linestyle='--', 
                  linewidth=2.5, alpha=0.9, zorder=5)
        ax.axvline(signal_region[1], color='#1E90FF', linestyle='--', 
                  linewidth=2.5, alpha=0.9, zorder=5, label='signal region')
    
    # Formatting
    ax.set_xlabel(r'$M_{JJ}$ (GeV)', fontsize=18)
    ylabel = 'Rel. Freq.' if normalize else 'Counts'
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title('Dijet invariant mass', fontsize=20, pad=15)
    ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def plot_case_signal_significance(
    background_file: str,
    signal_file: str,
    n_jets: Optional[int] = None,
    signal_region: Tuple[float, float] = (3300, 3700),
    scan_range: Tuple[float, float] = (2500, 4500),
    window_sizes: List[int] = [200, 400, 600, 800],
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot signal significance as a function of mass window position.
    
    This creates a scan plot showing how signal significance varies with
    the choice of signal region window.
    
    Parameters
    ----------
    background_file : str
        Path to background HDF5 file.
    signal_file : str
        Path to signal HDF5 file.
    n_jets : int, optional
        Number of events to load.
    signal_region : tuple
        Current signal region for reference line.
    scan_range : tuple
        Range over which to scan window positions.
    window_sizes : list of int
        Window widths to scan (in GeV).
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Load data
    mjj_bg, labels_bg = load_case_dijet_mass(background_file, n_jets=n_jets)
    mjj_bg = mjj_bg[labels_bg == 0]
    
    mjj_sig, labels_sig = load_case_dijet_mass(signal_file, n_jets=n_jets)
    mjj_sig = mjj_sig[labels_sig == 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scan over window positions
    for window_size in window_sizes:
        centers = np.arange(scan_range[0] + window_size/2, 
                           scan_range[1] - window_size/2, 10)
        significances = []
        
        for center in centers:
            window_low = center - window_size / 2
            window_high = center + window_size / 2
            
            # Count events in window
            n_sig = np.sum((mjj_sig >= window_low) & (mjj_sig <= window_high))
            n_bg = np.sum((mjj_bg >= window_low) & (mjj_bg <= window_high))
            
            # Calculate significance (S/sqrt(B))
            if n_bg > 0:
                significance = n_sig / np.sqrt(n_bg)
            else:
                significance = 0
            
            significances.append(significance)
        
        ax.plot(centers, significances, label=f'Window = {window_size} GeV', linewidth=2)
    
    # Mark current signal region
    sr_center = (signal_region[0] + signal_region[1]) / 2
    ax.axvline(sr_center, color='red', linestyle='--', linewidth=2, 
              label=f'Signal region center ({sr_center:.0f} GeV)')
    
    ax.set_xlabel(r'Window Center $M_{JJ}$ (GeV)', fontsize=14)
    ax.set_ylabel(r'Significance ($S/\sqrt{B}$)', fontsize=14)
    ax.set_title('Signal Significance vs Mass Window', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    """Command-line interface for plotting CASE dijet mass distributions."""
    parser = argparse.ArgumentParser(
        description="Plot CASE dataset dijet invariant mass distributions"
    )
    
    # Input files
    parser.add_argument("--background", type=str, help="Path to background HDF5 file")
    parser.add_argument("--signal", type=str, nargs='+', 
                       help="Path(s) to signal HDF5 file(s). Can specify multiple files.")
    parser.add_argument("--signal_labels", type=str, nargs='+',
                       help="Labels for each signal file (for legend). Must match number of signal files.")
    parser.add_argument("--signal_colors", type=str, nargs='+',
                       help="Colors for each signal file (hex or named colors). Must match number of signal files.")
    parser.add_argument("--n_jets", type=int, default=None, 
                       help="Number of events to load from each file")
    
    # Plotting options
    parser.add_argument("--use_kde", action="store_true", default=True,
                       help="Use KDE smoothing (default: True)")
    parser.add_argument("--use_hist", action="store_true",
                       help="Use histograms instead of KDE")
    parser.add_argument("--bandwidth", type=float, default=50.0,
                       help="KDE bandwidth in GeV (default: 50)")
    parser.add_argument("--n_bins", type=int, default=100,
                       help="Number of histogram bins (default: 100)")
    
    # Regions
    parser.add_argument("--signal_region", type=float, nargs=2, default=[2900, 3100],
                       help="Signal region boundaries [min max] in GeV")
    parser.add_argument("--sideband_region", type=float, nargs=2, default=[2725, 3331],
                       help="Sideband region boundaries [lower upper] in GeV")
    parser.add_argument("--no_signal_region", action="store_true",
                       help="Don't show signal region markers")
    parser.add_argument("--no_sidebands", action="store_true",
                       help="Don't show sideband region markers")
    
    # Display options
    parser.add_argument("--xlim", type=float, nargs=2, default=[1500, 7000],
                       help="X-axis limits [min max] in GeV")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                       help="Y-axis limits [min max]")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 6],
                       help="Figure size [width height] in inches")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize distributions to show relative frequency (default: show counts)")
    
    # Output
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save the figure")
    parser.add_argument("--no_show", action="store_true",
                       help="Don't display the figure")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figure (default: 300)")
    
    # Additional plots
    parser.add_argument("--significance_scan", action="store_true",
                       help="Also create a significance scan plot")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.background is None and args.signal is None:
        parser.error("At least one of --background or --signal must be provided")
    
    # Create main plot
    print("Creating dijet mass distribution plot...")
    fig = plot_case_dijet_mass_distribution(
        background_file=args.background,
        signal_file=args.signal,
        signal_labels=args.signal_labels,
        signal_colors=args.signal_colors,
        n_jets=args.n_jets,
        use_kde=not args.use_hist,
        bandwidth=args.bandwidth,
        n_bins=args.n_bins,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim) if args.ylim else None,
        normalize=args.normalize,
        show_signal_region=not args.no_signal_region,
        signal_region=tuple(args.signal_region),
        show_sidebands=not args.no_sidebands,
        sideband_region=tuple(args.sideband_region),
        figsize=tuple(args.figsize),
        save_path=args.save,
        show=not args.no_show,
        dpi=args.dpi,
    )
    
    # Create significance scan if requested
    if args.significance_scan and args.background and args.signal:
        print("\nCreating significance scan plot...")
        # Use first signal file for significance scan
        signal_for_scan = args.signal[0] if isinstance(args.signal, list) else args.signal
        print(f"Using {os.path.basename(signal_for_scan)} for significance scan")
        
        save_path_sig = None
        if args.save:
            base = Path(args.save)
            save_path_sig = str(base.parent / f"{base.stem}_significance{base.suffix}")
        
        fig_sig = plot_case_signal_significance(
            background_file=args.background,
            signal_file=signal_for_scan,
            n_jets=args.n_jets,
            signal_region=tuple(args.signal_region),
            save_path=save_path_sig,
            show=not args.no_show,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
