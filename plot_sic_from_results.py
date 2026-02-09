"""
Load evaluation results and plot SIC curves.

This script loads the predictions saved by evaluate.py and generates SIC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import roc_curve, auc


def calculate_sic(y_true: np.ndarray, y_scores: np.ndarray, 
                  signal_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Significance Improvement Characteristic curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 for background, 1 for signal)
    y_scores : np.ndarray
        Predicted scores/probabilities for the positive class
    signal_ratio : float
        Ratio of signal to background in the dataset (default: 1.0 for balanced)
    
    Returns
    -------
    tpr : np.ndarray
        True Positive Rate values
    sic : np.ndarray
        SIC values (TPR / sqrt(FPR))
    """
    # Sort by scores in descending order
    sort_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[sort_idx]
    
    # Calculate cumulative true positives and false positives
    n_signal = np.sum(y_true == 1)
    n_background = np.sum(y_true == 0)
    
    tp_cumsum = np.cumsum(y_true_sorted)
    fp_cumsum = np.cumsum(1 - y_true_sorted)
    
    # Calculate TPR and FPR
    tpr = tp_cumsum / n_signal
    fpr = fp_cumsum / n_background
    
    # Only calculate SIC where FPR > 0 to avoid division issues
    # For very small FPR, SIC becomes unreliable
    valid_idx = fpr > 0
    tpr_valid = tpr[valid_idx]
    fpr_valid = fpr[valid_idx]
    
    # Calculate SIC: TPR / sqrt(FPR)
    sic_valid = tpr_valid / np.sqrt(fpr_valid)
    
    # Add point at origin
    tpr = np.concatenate([[0], tpr_valid])
    sic = np.concatenate([[0], sic_valid])
    
    return tpr, sic


def plot_sic_curve(y_true: np.ndarray, y_scores: np.ndarray,
                   signal_ratios: Optional[List[float]] = None,
                   figsize: Tuple[int, int] = (8, 6),
                   save_path: Optional[str] = None,
                   show_plot: bool = True) -> plt.Figure:
    """
    Plot SIC curves for multiple signal ratios.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 for background, 1 for signal)
    y_scores : np.ndarray
        Predicted scores/probabilities for the positive class
    signal_ratios : list of float, optional
        Signal ratios to plot (default: [0.1, 0.3, 0.5, 1.0])
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    if signal_ratios is None:
        signal_ratios = [0.1, 0.3, 0.5, 1.0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for different signal ratios
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot SIC curves for each signal ratio
    for idx, ratio in enumerate(signal_ratios):
        tpr, sic = calculate_sic(y_true, y_scores, signal_ratio=ratio)
        
        label = f"{ratio * 100:.1f}%"
        color = colors[idx % len(colors)]
        ax.plot(tpr, sic, linewidth=2.0, label=label, color=color)
    
    # Plot random classifier baseline (SIC = sqrt(TPR))
    tpr_random = np.linspace(0, 1, 1000)
    sic_random = np.sqrt(tpr_random)
    ax.plot(tpr_random, sic_random, 'k--', linewidth=2, label='random', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('True Positive Rate', fontsize=14)
    ax.set_ylabel('SIC (TPR / $\\sqrt{\\mathrm{FPR}}$)', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    # ax.set_ylim(0, 20.0)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(title='Signal ratio', loc='upper right', fontsize=12, title_fontsize=12)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SIC curve saved to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


def plot_sic_curve_multiple_models(
    predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    signal_ratio: float = 0.1,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True) -> plt.Figure:
    """
    Plot SIC curves for multiple models at a fixed signal ratio.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to (y_true, y_scores) tuples
    signal_ratio : float
        Signal ratio to use for all models
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot SIC curve for each model
    for model_name, (y_true, y_scores) in predictions_dict.items():
        tpr, sic = calculate_sic(y_true, y_scores, signal_ratio=signal_ratio)
        ax.plot(tpr, sic, linewidth=2.0, label=model_name)
    
    # Plot random classifier baseline
    tpr_random = np.linspace(0, 1, 1000)
    sic_random = np.sqrt(tpr_random)
    ax.plot(tpr_random, sic_random, 'k--', linewidth=2, label='random', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('True Positive Rate', fontsize=14)
    ax.set_ylabel('SIC (TPR / $\\sqrt{\\mathrm{FPR}}$)', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 25.0])
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(title='Signal ratio', loc='upper right', fontsize=12, title_fontsize=12)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SIC curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_roc_curve_multiple_models(
    predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to (y_true, y_scores) tuples
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve for each model (TPR on x-axis, 1/FPR on y-axis)
    for model_name, (y_true, y_scores) in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Filter out FPR = 0 to avoid division by zero
        valid_idx = fpr > 0
        fpr_valid = fpr[valid_idx]
        tpr_valid = tpr[valid_idx]
        inv_fpr = 1.0 / fpr_valid
        
        ax.plot(tpr_valid, inv_fpr, linewidth=2.0, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot random classifier baseline (TPR = FPR => 1/FPR = 1/TPR)
    tpr_random = np.linspace(0.001, 1, 1000)  # TPR from 0.001 to 1
    inv_fpr_random = 1.0 / tpr_random
    ax.plot(tpr_random, inv_fpr_random, 'k--', linewidth=2, label='random', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('True Positive Rate', fontsize=14)
    ax.set_ylabel('1 / False Positive Rate', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.legend(title='Signal ratio', loc='upper right', fontsize=12, title_fontsize=12)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def calculate_r30(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate R30 metric: background rejection at 30% signal efficiency.
    
    R30 = 1/FPR at TPR=0.3
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 for background, 1 for signal)
    y_scores : np.ndarray
        Predicted scores/probabilities for the positive class
    
    Returns
    -------
    r30 : float
        Background rejection factor at 30% signal efficiency
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find the FPR at TPR closest to 0.3
    target_tpr = 0.3
    idx = np.argmin(np.abs(tpr - target_tpr))
    
    # R30 = 1/FPR at TPR=0.3
    if fpr[idx] > 0:
        r30 = 1.0 / fpr[idx]
    else:
        r30 = np.inf
    
    return r30


def plot_r30_vs_signal_ratio(
    predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True) -> plt.Figure:
    """
    Plot R30 values vs signal percentage for multiple models.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping signal ratio labels to (y_true, y_scores) tuples
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract signal ratios and calculate R30 for each
    signal_ratios = []
    r30_values = []
    
    for label, (y_true, y_scores) in predictions_dict.items():
        # Extract percentage from label (e.g., "0.60%" -> 0.60)
        try:
            ratio = float(label.rstrip('%'))
            signal_ratios.append(ratio)
            r30 = calculate_r30(y_true, y_scores)
            r30_values.append(r30)
        except ValueError:
            print(f"Warning: Could not parse signal ratio from label '{label}'")
            continue
    
    # Sort by signal ratio
    sorted_pairs = sorted(zip(signal_ratios, r30_values))
    signal_ratios, r30_values = zip(*sorted_pairs)
    
    # Plot
    ax.plot(signal_ratios, r30_values, marker='o', markersize=8, linewidth=2.0, color='#1f77b4')
    
    # Formatting
    ax.set_xlabel('Signal fraction [%]', fontsize=14)
    ax.set_ylabel('$R_{30}$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"R30 plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def load_predictions(eval_dir):
    """
    Load predictions from evaluation results directory.
    
    Parameters
    ----------
    eval_dir : str or Path
        Path to evaluation results directory (e.g., 'results/eval_run_xxx_20251203_113952/')
    
    Returns
    -------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Prediction scores
    """
    eval_dir = Path(eval_dir)
    predictions_path = eval_dir / 'predictions.npz'
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
    
    # Load the npz file
    data = np.load(predictions_path)
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    print(f"Loaded predictions from {predictions_path}")
    print(f"  - Total samples: {len(y_true)}")
    print(f"  - Signal samples: {np.sum(y_true == 1)}")
    print(f"  - Background samples: {np.sum(y_true == 0)}")
    print(f"  - Score range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="Plot SIC curves from evaluation results"
    )
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        required=True,
        help="Path to evaluation results directory (e.g., results/eval_run_xxx_20251203_113952/)"
    )
    parser.add_argument(
        "--signal_ratios", 
        type=float, 
        nargs='+',
        default=[0.1, 0.3, 0.5, 1.0],
        help="Signal ratios to plot (e.g., 0.1 0.3 0.5 1.0)"
    )
    parser.add_argument(
        "--title", 
        type=str,
        default="Pre-trained on QCD",
        help="Plot title"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output path for the SIC curve plot (default: saves in eval_dir)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='+',
        help="Additional eval directories to compare (optional)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        help="Labels for comparison plot (must match number of directories)"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    y_true, y_pred = load_predictions(args.eval_dir)
    
    # Determine output path
    if args.output is None:
        output_path = Path('plots') / 'sic_curve.png'
    else:
        output_path = Path(args.output)
    
    if args.compare:
        # Comparison mode: plot multiple models
        print("\nComparison mode: plotting multiple models...")
        
        # Load all predictions
        predictions_dict = {}
        
        # First model (from --eval_dir)
        label_main = args.labels[0] if args.labels else Path(args.eval_dir).name
        predictions_dict[label_main] = (y_true, y_pred)
        
        # Additional models (from --compare)
        for idx, compare_dir in enumerate(args.compare):
            y_true_comp, y_pred_comp = load_predictions(compare_dir)
            label = args.labels[idx + 1] if args.labels and len(args.labels) > idx + 1 else Path(compare_dir).name
            predictions_dict[label] = (y_true_comp, y_pred_comp)
        
        # Use first signal ratio for comparison
        signal_ratio = args.signal_ratios[0]
        
        # Plot SIC curve
        fig = plot_sic_curve_multiple_models(
            predictions_dict=predictions_dict,
            signal_ratio=signal_ratio,
            save_path=str(output_path),
            show_plot=False
        )
        
        # Plot ROC curve
        roc_output_path = output_path.parent / output_path.name.replace('sic_curve', 'roc_curve')
        fig_roc = plot_roc_curve_multiple_models(
            predictions_dict=predictions_dict,
            save_path=str(roc_output_path),
            show_plot=False
        )
        
        # Plot R30 vs Signal Ratio
        r30_output_path = output_path.parent / output_path.name.replace('sic_curve', 'r30_vs_signal')
        fig_r30 = plot_r30_vs_signal_ratio(
            predictions_dict=predictions_dict,
            save_path=str(r30_output_path),
            show_plot=False
        )
        
    else:
        # Single model mode
        print("\nPlotting SIC curve...")
        fig = plot_sic_curve(
            y_true=y_true,
            y_scores=y_pred,
            signal_ratios=args.signal_ratios,
            title=args.title,
            save_path=str(output_path),
            show_plot=False
        )
    
    print(f"\nSIC curve saved to: {output_path}")
    if args.compare:
        print("SIC, ROC, and R30 curves generated.")
    print("Done!")


if __name__ == "__main__":
    main()
