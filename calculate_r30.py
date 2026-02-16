"""
Calculate R30 metric from evaluation results.

R30 = Background rejection at 30% signal efficiency
    = 1/FPR at TPR=0.3

This script loads predictions and outputs only the R30 value.
"""

import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_curve


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
    # Check if predictions are essentially constant (no variation)
    if np.std(y_scores) < 1e-9:
        return 1.0, 0.0, "no_discrimination"
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find the FPR at TPR closest to 0.3
    target_tpr = 0.3
    idx = np.argmin(np.abs(tpr - target_tpr))
    
    actual_tpr = tpr[idx]
    
    # If TPR is essentially zero, model has no discrimination
    if actual_tpr < 1e-6:
        return 1.0, actual_tpr, "no_discrimination"
    
    # R30 is only meaningful if we can get reasonably close to 30% TPR
    # If we're far from target, the metric is misleading - return baseline
    if actual_tpr < 0.2:  # Less than 20% signal efficiency
        return 1.0, actual_tpr, "insufficient_efficiency"
    
    # Check if we're somewhat far from target TPR
    if np.abs(actual_tpr - target_tpr) > 0.1:
        status = "far_from_target"
    else:
        status = "ok"
    
    # R30 = 1/FPR at TPR=0.3 (or closest achievable)
    if fpr[idx] > 0:
        r30 = 1.0 / fpr[idx]
    else:
        r30 = np.inf
        status = "fpr_zero"
    
    return r30, actual_tpr, status


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
    
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="Calculate R30 metric from evaluation results"
    )
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        required=True,
        help="Path to evaluation results directory (e.g., results/eval_run_xxx_20251203_113952/)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print additional information"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    y_true, y_pred = load_predictions(args.eval_dir)
    
    # Calculate R30
    r30, actual_tpr, status = calculate_r30(y_true, y_pred)
    
    if args.verbose:
        print(f"Evaluation directory: {args.eval_dir}")
        print(f"Total samples: {len(y_true)}")
        print(f"Signal samples: {np.sum(y_true == 1)}")
        print(f"Background samples: {np.sum(y_true == 0)}")
        print(f"Prediction score range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
        print(f"Prediction std: {y_pred.std():.10f}")
        print(f"Actual TPR at measurement: {actual_tpr:.4f} (target: 0.30)")
        print(f"Status: {status}")
        print(f"R30: {r30:.2f}")
        
        if status == "no_discrimination":
            print("\nWARNING: Model has no discriminative power (all predictions are constant)")
        elif status == "insufficient_efficiency":
            print(f"\nWARNING: Model cannot reach sufficient signal efficiency")
            print(f"         Maximum achievable TPR near target: {actual_tpr:.4f} (need ~0.30)")
            print(f"         R30 is not meaningful for this model")
        elif status == "far_from_target":
            print(f"\nWARNING: Actual TPR ({actual_tpr:.4f}) is somewhat far from target (0.30)")
        elif status == "fpr_zero":
            print("\nWARNING: FPR is zero at measurement point (returning inf)")
    else:
        if status in ["no_discrimination", "insufficient_efficiency"]:
            print(f"1.00 ({status})")
        else:
            print(f"{r30:.2f}")


if __name__ == "__main__":
    main()
