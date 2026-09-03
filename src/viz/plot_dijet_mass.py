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
        
        # Plot signal first (all data)
        if len(mjj_signal) > 0:
            kde_sig = gaussian_kde(mjj_signal, bw_method=bandwidth / mjj_signal.std())
            density_sig = kde_sig(x_eval)
            if not normalize:
                density_sig = density_sig * len(mjj_signal)
            ax.fill_between(x_eval, density_sig, alpha=1, color='#fbb4ae',
                            label='signal', linewidth=1, edgecolor='#D6938A')
        
        # Plot background on top
        if len(mjj_background) > 0:
            kde_bg = gaussian_kde(mjj_background, bw_method=bandwidth / mjj_background.std())
            density_bg = kde_bg(x_eval)
            if not normalize:
                density_bg = density_bg * len(mjj_background)
            ax.fill_between(x_eval, density_bg, alpha=1, color='#fed9a6', 
                            label='background', linewidth=1, edgecolor='#e6ab02')
    
    # Case 2: Separate files for background and signal
    else:
        # Plot signal distribution first
        if signal_file is not None:
            mjj_signal, _ = load_dijet_mass_from_h5(signal_file, n_jets=n_jets)
            if len(mjj_signal) > 0:
                kde_sig = gaussian_kde(mjj_signal, bw_method=bandwidth / mjj_signal.std())
                density_sig = kde_sig(x_eval)
                if not normalize:
                    density_sig = density_sig * len(mjj_signal)
                ax.fill_between(x_eval, density_sig, alpha=1, color='#fbb4ae',
                                label='signal', linewidth=1, edgecolor='#D6938A')
        
        # Plot background distribution on top
        if background_file is not None:
            mjj_background, _ = load_dijet_mass_from_h5(background_file, n_jets=n_jets)
            if len(mjj_background) > 0:
                kde_bg = gaussian_kde(mjj_background, bw_method=bandwidth / mjj_background.std())
                density_bg = kde_bg(x_eval)
                if not normalize:
                    density_bg = density_bg * len(mjj_background)
                ax.fill_between(x_eval, density_bg, alpha=1, color='#fed9a6', 
                                label='background', linewidth=1, edgecolor='#e6ab02')
    
    # Formatting
    ax.set_xlabel(r'$M_{JJ}$ (GeV)', fontsize=15)
    ylabel = 'Counts' if not normalize else 'Density'
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Ensure y-axis starts from 0
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, current_ylim[1])
    
    # Add background region markers if requested (plot first so signal region is on top)
    if show_background_region:
        ymin, ymax = ax.get_ylim()
        ax.axvline(background_region[0], color='#1f78b4', linestyle='--', linewidth=1, alpha=1, zorder=2)
        ax.axvline(background_region[1], color='#1f78b4', linestyle='--', linewidth=1, alpha=1, zorder=2)
        # Add shaded regions for side bands: [2900, 3300] and [3700, 4100]
        ax.axvspan(2900, 3300, alpha=0.1, color='#1f78b4', zorder=1)
        ax.axvspan(3700, 4100, alpha=0.1, color='#1f78b4', zorder=1)
    
    # Add signal region markers if requested (plot on top)
    if show_signal_region:
        ymin, ymax = ax.get_ylim()
        ax.axvline(signal_region[0], color='#1f78b4', linestyle='--', linewidth=1, alpha=1, zorder=3)
        ax.axvline(signal_region[1], color='#1f78b4', linestyle='--', linewidth=1, alpha=1, zorder=3)
        # Optionally add shaded region
        ax.axvspan(signal_region[0], signal_region[1], alpha=0.15, color='#1f78b4', label='signal region', zorder=2)
    
    if show_background_region:
        # Add background region to legend
        from matplotlib.patches import Patch
        legend_elements = ax.get_legend_handles_labels()
        if legend_elements[0]:  # If there are existing legend items
            handles, labels = legend_elements
            handles.append(Patch(facecolor='#1f78b4', alpha=0.1, label='side bands'))
            labels.append('side bands')
            ax.legend(handles=handles, labels=labels, fontsize=15, loc='upper right', framealpha=0.9)
        else:
            ax.legend(fontsize=15, loc='upper right', framealpha=0.9)
    else:
        ax.legend(fontsize=15, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        # Save both PNG and PDF formats
        save_path_obj = Path(save_path)
        if save_path_obj.suffix:  # Has extension
            # Save with provided extension
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            # Also save in other format
            if save_path_obj.suffix.lower() == '.png':
                pdf_path = save_path_obj.with_suffix('.pdf')
                fig.savefig(pdf_path, bbox_inches='tight')
                print(f"Figure saved to {pdf_path}")
            elif save_path_obj.suffix.lower() == '.pdf':
                png_path = save_path_obj.with_suffix('.png')
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {png_path}")
        else:  # No extension, save both
            png_path = str(save_path_obj) + '.png'
            pdf_path = str(save_path_obj) + '.pdf'
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f"Figure saved to {png_path}")
            print(f"Figure saved to {pdf_path}")
    
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
            ax.plot(x_eval, density, label=label, linewidth=1, color=color, alpha=0.8)
    
    # Formatting
    ax.set_xlabel(r'$M_{JJ}$ (GeV)', fontsize=15)
    ax.set_ylabel('Density', fontsize=15)
    ax.set_xlim(xlim)
    # Ensure y-axis starts from 0
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1])
    ax.legend(fontsize=15, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        # Save both PNG and PDF formats
        save_path_obj = Path(save_path)
        if save_path_obj.suffix:  # Has extension
            # Save with provided extension
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            # Also save in other format
            if save_path_obj.suffix.lower() == '.png':
                pdf_path = save_path_obj.with_suffix('.pdf')
                fig.savefig(pdf_path, bbox_inches='tight')
                print(f"Figure saved to {pdf_path}")
            elif save_path_obj.suffix.lower() == '.pdf':
                png_path = save_path_obj.with_suffix('.png')
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {png_path}")
        else:  # No extension, save both
            png_path = str(save_path_obj) + '.png'
            pdf_path = str(save_path_obj) + '.pdf'
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f"Figure saved to {png_path}")
            print(f"Figure saved to {pdf_path}")
    
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
