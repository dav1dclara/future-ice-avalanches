"""Training utilities for model training."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

# Chart color: CMYK 100/57/0/0 converted to RGB (0, 110, 255) normalized to (0, 110/255, 1.0)
CHART_COLOR = "#006ab6"
BACKGROUND_COLOR = "#ededee"

A3_TO_A0 = 2.828
MM_TO_INCH = 1 / 25.4


def figsize_from_mm(
    w_mm: float, h_mm: float, scale: float = A3_TO_A0
) -> Tuple[float, float]:
    """Convert dimensions from millimeters to inches for matplotlib figure size.

    Args:
        w_mm: Width in millimeters
        h_mm: Height in millimeters
        scale: Scaling factor (default A3_TO_A0 for poster scaling)

    Returns:
        Tuple of (width, height) in inches
    """
    return (w_mm * MM_TO_INCH * scale, h_mm * MM_TO_INCH * scale)


def compute_success_rate_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute Success Rate Curve for a spatial risk-ranking problem.

    The Success Rate Curve shows: if we inspect areas in order of risk (highest first),
    what percentage of known positives do we find as a function of inspected area percentage.

    Args:
        y_true: Array of true labels (1 for positives, 0 for unlabeled).
        y_pred_proba: Array of predicted risk probabilities (same shape as y_true).

    Returns:
        Tuple of:
        - inspected_area_pct: Array of inspected area percentages (0 to 100)
        - detected_positives_pct: Array of detected positive percentages (0 to 100)
        - metrics: Dict with keys:
            - 'pct_found_top2': Percentage of positives found in top 2% area
            - 'pct_found_top5': Percentage of positives found in top 5% area
            - 'pct_found_top10': Percentage of positives found in top 10% area
            - 'pct_found_top20': Percentage of positives found in top 20% area
            - 'auc': Area under the curve (0 to 1)
    """
    if y_true.shape != y_pred_proba.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} != y_pred_proba {y_pred_proba.shape}"
        )

    # Keep only valid entries (finite predictions)
    valid_mask = np.isfinite(y_pred_proba)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_proba[valid_mask]

    if len(y_pred_valid) == 0:
        raise ValueError("No valid predictions found")

    # Sort by risk (highest to lowest)
    sort_indices = np.argsort(y_pred_valid)[::-1]
    y_pred_sorted = y_pred_valid[sort_indices]
    y_true_sorted = y_true_valid[sort_indices]

    # Total area and total positives
    total_area = len(y_pred_sorted)
    total_positives = np.sum(y_true_sorted == 1)

    if total_positives == 0:
        # No positives found - return zero curve
        inspected_area_pct = np.linspace(0, 100, 101)
        detected_positives_pct = np.zeros_like(inspected_area_pct)
        metrics = {
            "pct_found_top2": 0.0,
            "pct_found_top5": 0.0,
            "pct_found_top10": 0.0,
            "pct_found_top20": 0.0,
            "auc": 0.0,
        }
        return inspected_area_pct, detected_positives_pct, metrics

    # Compute cumulative statistics
    positive_sorted = (y_true_sorted == 1).astype(int)
    cumulative_positives = np.cumsum(positive_sorted)
    cumulative_area = np.arange(1, total_area + 1)

    # Convert to percentages
    inspected_area_pct = (cumulative_area / total_area) * 100
    detected_positives_pct = (cumulative_positives / total_positives) * 100

    # Compute metrics for top 2%, 5%, 10%, 20%
    top2_idx = int(np.ceil(total_area * 0.02))
    top5_idx = int(np.ceil(total_area * 0.05))
    top10_idx = int(np.ceil(total_area * 0.10))
    top20_idx = int(np.ceil(total_area * 0.20))

    pct_found_top2 = (
        (cumulative_positives[min(top2_idx - 1, total_area - 1)] / total_positives)
        * 100
        if top2_idx > 0
        else 0.0
    )
    pct_found_top5 = (
        (cumulative_positives[min(top5_idx - 1, total_area - 1)] / total_positives)
        * 100
        if top5_idx > 0
        else 0.0
    )
    pct_found_top10 = (
        (cumulative_positives[min(top10_idx - 1, total_area - 1)] / total_positives)
        * 100
        if top10_idx > 0
        else 0.0
    )
    pct_found_top20 = (
        (cumulative_positives[min(top20_idx - 1, total_area - 1)] / total_positives)
        * 100
        if top20_idx > 0
        else 0.0
    )

    # Compute AUC (area under curve, normalized to 0-1)
    # The curve goes from (0, 0) to (100, 100) for perfect model
    # AUC is computed using trapezoidal rule, then normalized by dividing by 10000 (max possible)
    auc = np.trapz(detected_positives_pct, inspected_area_pct) / 10000.0

    metrics = {
        "pct_found_top2": pct_found_top2,
        "pct_found_top5": pct_found_top5,
        "pct_found_top10": pct_found_top10,
        "pct_found_top20": pct_found_top20,
        "auc": auc,
    }

    return inspected_area_pct, detected_positives_pct, metrics


def _plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
    verbose: bool = True,
) -> None:
    """Plot feature importance as a horizontal bar chart.

    Args:
        importance_df: DataFrame with columns 'feature', 'importance_mean', 'importance_std'
        output_dir: Directory to save the plot
        top_n: Number of top features to display (default 20)
        verbose: Whether to print progress
    """
    if importance_df is None or len(importance_df) == 0:
        if verbose:
            print("  -> Skipping feature importance plot (no data)")
        return

    # Get top N features
    top_features = importance_df.head(top_n).copy()

    # Save original font settings
    original_font_family = plt.rcParams["font.family"]
    original_font_serif = plt.rcParams["font.serif"]
    original_mathtext = plt.rcParams["mathtext.fontset"]

    # Set scientific font (serif font family)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Computer Modern Roman",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"  # Use STIX fonts for math

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize_from_mm(88, 45))

    # No need to reverse the order -- highest mean will be at the top

    # Plot horizontal bars
    y_pos = np.arange(len(top_features))
    bars = ax.barh(
        y_pos,
        top_features["importance_mean"] * 100,
        xerr=top_features["importance_std"] * 100,
        capsize=3,
        color=CHART_COLOR,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        zorder=100,
    )

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"], fontsize=16)

    # Labels and title
    ax.set_xlabel("Feature Importance (%)", fontsize=16)
    ax.set_title(
        f"c) Feature Importance (Top {min(top_n, len(importance_df))} features)",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, alpha=0.3, axis="x")

    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "feature_importance.png"
    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        facecolor=BACKGROUND_COLOR,
    )
    plt.close()

    # Restore original font settings
    plt.rcParams["font.family"] = original_font_family
    plt.rcParams["font.serif"] = original_font_serif
    plt.rcParams["mathtext.fontset"] = original_mathtext

    if verbose:
        print(f"  -> Saved feature importance plot to '{plot_path}'")


def _plot_success_rate_curve(
    success_rate_curves: List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Plot Success Rate Curve with mean and std bands across folds.

    Success Rate Curve: Shows what percentage of known positives are found
    as a function of inspected area percentage (spatial risk-ranking).

    Args:
        success_rate_curves: List of (inspected_area_pct, detected_positives_pct, metrics) tuples
        output_dir: Directory to save the plot
        verbose: Whether to print progress
    """
    # Interpolate all curves to common x-axis for computing mean/std
    n_points = 100
    area_pct_common = np.linspace(0, 100, n_points)

    # Interpolate Success Rate curves
    detected_pct_interp = []
    success_rate_aucs = []
    for inspected_area_pct, detected_positives_pct, metrics in success_rate_curves:
        # Interpolate to common area_pct
        detected_pct_interp.append(
            np.interp(area_pct_common, inspected_area_pct, detected_positives_pct)
        )
        success_rate_aucs.append(metrics["auc"])

    # Compute mean and std
    detected_pct_mean = np.mean(detected_pct_interp, axis=0)
    detected_pct_std = np.std(detected_pct_interp, axis=0)
    success_rate_auc_mean = np.mean(success_rate_aucs)
    success_rate_auc_std = np.std(success_rate_aucs)

    # Save original font settings
    original_font_family = plt.rcParams["font.family"]
    original_font_serif = plt.rcParams["font.serif"]
    original_mathtext = plt.rcParams["mathtext.fontset"]

    # Set scientific font (serif font family)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Computer Modern Roman",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"  # Use STIX fonts for math

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize_from_mm(45, 45))

    # Plot Success Rate curve
    ax.plot(
        area_pct_common,
        detected_pct_mean,
        color=CHART_COLOR,
        linestyle="-",
        label=f"Mean (AUC = {success_rate_auc_mean:.2f} ± {success_rate_auc_std:.2f})",
        linewidth=2,
        zorder=200,
    )
    ax.fill_between(
        area_pct_common,
        detected_pct_mean - detected_pct_std,
        detected_pct_mean + detected_pct_std,
        alpha=0.2,
        color=CHART_COLOR,
        zorder=100,
        edgecolor="none",  # Remove outline
    )
    # Diagonal reference line (random model)
    ax.plot([0, 100], [0, 100], "k--", label="Random model", linewidth=1, zorder=1)
    ax.set_xlabel("Cumulative area, ranked by risk (%)", fontsize=16)
    ax.set_ylabel("Cumulative percentage of positives (%)", fontsize=16)
    ax.set_title("a) Success Rate Curve", fontsize=18, fontweight="bold", pad=16)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect("equal", adjustable="box")  # Explicitly ensure equal aspect
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "success_rate_curve.png"
    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        facecolor=BACKGROUND_COLOR,
    )
    plt.close()

    # Restore original font settings
    plt.rcParams["font.family"] = original_font_family
    plt.rcParams["font.serif"] = original_font_serif
    plt.rcParams["mathtext.fontset"] = original_mathtext

    if verbose:
        print(f"  -> Saved Success Rate curve to '{plot_path}'")


def _compute_positives_per_bin(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    bin_size_pct: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute percentage of positives in each inspected area bin.

    Args:
        y_true: Array of true labels (1 for positives, 0 for unlabeled).
        y_pred_proba: Array of predicted risk probabilities (same shape as y_true).
        bin_size_pct: Size of each bin in percentage (default 10%).

    Returns:
        Tuple of:
        - bin_centers: Array of bin center percentages (e.g., [5, 15, 25, ...])
        - pct_positives_per_bin: Array of percentage of total positives in each bin
    """
    # Keep only valid entries (finite predictions)
    valid_mask = np.isfinite(y_pred_proba)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_proba[valid_mask]

    if len(y_pred_valid) == 0:
        return np.array([]), np.array([])

    # Sort by risk (highest to lowest)
    sort_indices = np.argsort(y_pred_valid)[::-1]
    y_true_sorted = y_true_valid[sort_indices]

    # Total area and total positives
    total_area = len(y_true_sorted)
    total_positives = np.sum(y_true_sorted == 1)

    if total_positives == 0:
        return np.array([]), np.array([])

    # Create bins (0-10%, 10-20%, etc.)
    n_bins = int(100 / bin_size_pct)
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert area percentages to indices
    area_indices = np.arange(total_area)
    area_pct = (area_indices / total_area) * 100

    # Bin the area percentages and count positives in each bin
    pct_positives_per_bin = []
    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        # Get indices in this bin (use < bin_end for last bin to include 100%)
        if i == n_bins - 1:
            bin_mask = (area_pct >= bin_start) & (area_pct <= bin_end)
        else:
            bin_mask = (area_pct >= bin_start) & (area_pct < bin_end)

        # Count positives in this bin
        positives_in_bin = np.sum(y_true_sorted[bin_mask] == 1)
        # Percentage of total positives
        pct_positives = (positives_in_bin / total_positives) * 100
        pct_positives_per_bin.append(pct_positives)

    return bin_centers, np.array(pct_positives_per_bin)


def _plot_positives_histogram(
    y_test_list: List[np.ndarray],
    y_pred_proba_list: List[np.ndarray],
    output_dir: Path,
    bin_size_pct: float = 10.0,
    verbose: bool = True,
) -> None:
    """Plot histogram showing percentage of positives in each inspected area bin.

    Args:
        y_test_list: List of y_test arrays for each fold
        y_pred_proba_list: List of y_pred_proba arrays for each fold
        output_dir: Directory to save the plot
        bin_size_pct: Size of each bin in percentage (default 10%)
        verbose: Whether to print progress
    """
    # Compute histogram data for each fold
    all_bin_centers = []
    all_pct_positives = []

    for y_test, y_pred_proba in zip(y_test_list, y_pred_proba_list):
        bin_centers, pct_positives = _compute_positives_per_bin(
            y_true=y_test,
            y_pred_proba=y_pred_proba,
            bin_size_pct=bin_size_pct,
        )
        if len(bin_centers) > 0:
            all_bin_centers.append(bin_centers)
            all_pct_positives.append(pct_positives)

    if len(all_bin_centers) == 0:
        if verbose:
            print("  -> Skipping positives histogram (no valid data)")
        return

    # Compute mean and std across folds
    # All folds should have the same bin centers, so we can just use the first one
    bin_centers = all_bin_centers[0]
    pct_positives_array = np.array(all_pct_positives)
    pct_positives_mean = np.mean(pct_positives_array, axis=0)
    pct_positives_std = np.std(pct_positives_array, axis=0)

    # Save original font settings
    original_font_family = plt.rcParams["font.family"]
    original_font_serif = plt.rcParams["font.serif"]
    original_mathtext = plt.rcParams["mathtext.fontset"]

    # Set scientific font (serif font family)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Computer Modern Roman",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"  # Use STIX fonts for math

    # Create figure with a square Axes object
    fig, ax = plt.subplots(1, 1, figsize=figsize_from_mm(45, 45))

    # --- Force the subplot (axes area) to be square, even after tight_layout ---
    # Use set_aspect('equal', adjustable='box') on the axis.
    # Also, after tight_layout, use fig.subplots_adjust to force square plotting region.

    # Plot histogram with error bars
    bin_width = bin_size_pct * 0.85  # Make bars slightly narrower than bin width
    bars = ax.bar(
        bin_centers,
        pct_positives_mean,
        width=bin_width,
        yerr=pct_positives_std,
        capsize=3,
        color=CHART_COLOR,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Mean ± Std",
        zorder=100,
    )

    ax.set_xlabel("Cumulative area, ranked by risk (%)", fontsize=16)
    ax.set_ylabel("Cumulative percentage of positives (%)", fontsize=16)
    ax.set_title(
        "b) Distribution of Positives",
        fontsize=18,
        fontweight="bold",
        pad=16,  # Add a little bit of spacing after the title
    )
    ax.grid(True, alpha=0.3, axis="y")
    # ax.set_xlim([bin_centers[0] - bin_size_pct, bin_centers[-1] + bin_size_pct])
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)

    # --- Force aspect ratio to be exactly square on content area ---
    ax.set_box_aspect(1)
    # ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Optionally further ensure squareness by tweaking subplot parameters
    # fig.subplots_adjust(left=..., right=..., top=..., bottom=...) as needed
    # This is usually not necessary, but if you want the axis limits to also be equal, you can use:
    # ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val)

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "positives_histogram.png"
    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        facecolor=BACKGROUND_COLOR,
    )
    plt.close()

    # Restore original font settings
    plt.rcParams["font.family"] = original_font_family
    plt.rcParams["font.serif"] = original_font_serif
    plt.rcParams["mathtext.fontset"] = original_mathtext

    if verbose:
        print(f"  -> Saved positives histogram to '{plot_path}'")


def _extract_feature_importance(
    model: Any, feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """Extract feature importance from a PU classifier model.

    Handles different PU classifier types:
    - BaggingPuClassifier: Aggregates importances from all base estimators
    - ElkanotoPuClassifier / WeightedElkanotoPuClassifier: Gets importance from single base estimator

    Args:
        model: Trained PU classifier instance
        feature_names: Optional list of feature names (for validation)

    Returns:
        Array of feature importances (normalized to sum to 1)
    """
    # Check if model has base_estimator attribute (ElkanotoPU, WeightedElkanotoPU)
    if hasattr(model, "base_estimator") and hasattr(
        model.base_estimator, "feature_importances_"
    ):
        return model.base_estimator.feature_importances_

    # Check if model has estimator attribute (BaggingPU)
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        # Aggregate importances from all estimators
        importances = []
        for estimator in model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                importances.append(estimator.feature_importances_)

        if importances:
            # Average across all estimators
            return np.mean(importances, axis=0)

    # Fallback: try to get directly from model
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # If no importance found, return None
    return None


def train_cross_validation(
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    fold_samples: np.ndarray,
    model: Any,  # PU classifier (BaggingPuClassifier, ElkanotoPuClassifier, etc.)
    n_folds: int,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    wandb_run: Optional[object] = None,
) -> Dict[str, Any]:
    """Train model using cross-validation across folds.

    For each fold:
    1. Split data into train (all other folds) and test (current fold)
    2. Scale features using StandardScaler fit on training data
    3. Train model on scaled training data
    4. Evaluate on scaled test data
    5. Extract feature importance from trained model
    6. Log metrics to wandb if provided

    Args:
        X_samples: Feature matrix of shape (n_samples, n_features).
        y_samples: Label array of shape (n_samples,). Values: 1=positive, 0=unlabeled.
        fold_samples: Fold assignment array of shape (n_samples,).
        model: PU classifier instance (will be cloned for each fold).
            Supports BaggingPuClassifier, ElkanotoPuClassifier, WeightedElkanotoPuClassifier, etc.
        n_folds: Number of folds for cross-validation.
        feature_names: Optional list of feature names for importance analysis.
        output_dir: Optional directory to save Success Rate curve plots.
        verbose: Whether to print progress and results.
        wandb_run: Optional wandb run object for logging metrics.

    Returns:
        Dictionary with overall metrics and feature importance:
        - 'cv_src_auc_mean': Mean Success Rate Curve AUC across folds
        - 'cv_src_auc_std': Std Success Rate Curve AUC across folds
        - 'cv_mean_rank_percentile_mean': Mean rank percentile of positives across folds (0-100%, lower is better)
        - 'cv_mean_rank_percentile_std': Std rank percentile of positives across folds
        - 'feature_importance_mean': Mean feature importance across folds (array)
        - 'feature_importance_std': Std feature importance across folds (array)
        - 'feature_importance_df': DataFrame with feature names and importance stats
    """
    fold_metrics = {"src_auc": [], "mean_rank_percentile": []}
    fold_importances = []  # Store importances from each fold
    success_rate_curves = []  # Store Success Rate curve data for each fold
    y_test_list = []  # Store y_test for each fold (for histogram)
    y_pred_proba_list = []  # Store y_pred_proba for each fold (for histogram)

    for fold_idx in range(1, n_folds + 1):
        # Split data: train on all other folds, test on current fold
        train_mask = fold_samples != fold_idx
        test_mask = fold_samples == fold_idx

        X_train = X_samples[train_mask]
        y_train = y_samples[train_mask]
        X_test = X_samples[test_mask]
        y_test = y_samples[test_mask]

        if len(X_test) == 0:
            if verbose:
                print(f"  Fold {fold_idx}: No test samples, skipping...")
            continue

        if verbose:
            n_train_pos = np.sum(y_train == 1)
            n_train_unl = np.sum(y_train == 0)
            n_test_pos = np.sum(y_test == 1)
            n_test_unl = np.sum(y_test == 0)
            print(
                f"  Fold {fold_idx}: Train={len(X_train):,} ({n_train_pos:,} pos, {n_train_unl:,} unl), "
                f"Test={len(X_test):,} ({n_test_pos:,} pos, {n_test_unl:,} unl)"
            )

        # Scale features: fit on training data, transform both train and test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clone model for this fold (to avoid reusing state)
        fold_model = deepcopy(model)

        # Train model
        fold_model.fit(X_train_scaled, y_train)

        # Extract feature importance from trained model
        fold_importance = _extract_feature_importance(fold_model)
        if fold_importance is not None:
            fold_importances.append(fold_importance)

        # Predict on test set
        y_pred_proba = fold_model.predict_proba(X_test_scaled)[:, 1]

        # Store test data for histogram
        y_test_list.append(y_test)
        y_pred_proba_list.append(y_pred_proba)

        # MEMORY OPTIMIZATION: Delete fold-specific data after use
        # (X_train, X_test, y_train, y_test, scaler are recreated each fold)
        del fold_model, X_train_scaled, X_test_scaled

        # Compute mean rank percentile of positives
        mean_rank_percentile = np.nan
        try:
            # Keep only valid entries
            valid_mask = np.isfinite(y_pred_proba)
            y_test_valid = y_test[valid_mask]
            y_pred_valid = y_pred_proba[valid_mask]

            if len(y_pred_valid) > 0:
                # Sort by risk (highest to lowest)
                sort_indices = np.argsort(y_pred_valid)[::-1]
                y_test_sorted = y_test_valid[sort_indices]

                # Find ranks of positives (0-indexed, so rank 0 = highest risk)
                positive_mask = y_test_sorted == 1
                positive_ranks = np.where(positive_mask)[0]

                if len(positive_ranks) > 0:
                    # Convert ranks to percentiles (0-100%)
                    # Rank 0 (highest risk) = 0%, rank (n-1) (lowest risk) = 100%
                    total_samples = len(y_test_sorted)
                    rank_percentiles = (
                        (positive_ranks / (total_samples - 1)) * 100
                        if total_samples > 1
                        else np.zeros_like(positive_ranks)
                    )
                    mean_rank_percentile = np.mean(rank_percentiles)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not compute mean rank percentile: {e}")

        # Compute Success Rate curve
        src_auc = np.nan
        try:
            inspected_area_pct, detected_positives_pct, metrics = (
                compute_success_rate_curve(
                    y_true=y_test,
                    y_pred_proba=y_pred_proba,
                )
            )
            success_rate_curves.append(
                (inspected_area_pct, detected_positives_pct, metrics)
            )
            src_auc = metrics["auc"]

            if verbose:
                print(
                    f"    Success Rate - Top 2%: {metrics['pct_found_top2']:.1f}%, "
                    f"Top 5%: {metrics['pct_found_top5']:.1f}%, "
                    f"Top 10%: {metrics['pct_found_top10']:.1f}%, "
                    f"Top 20%: {metrics['pct_found_top20']:.1f}%, "
                    f"AUC: {src_auc:.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not compute success rate curve: {e}")

        fold_metrics["src_auc"].append(src_auc)
        fold_metrics["mean_rank_percentile"].append(mean_rank_percentile)

        if verbose:
            print(
                f"    SRC-AUC: {src_auc:.4f}, Mean Rank Percentile: {mean_rank_percentile:.2f}%"
            )

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"fold_{fold_idx}/src_auc": src_auc,
                    f"fold_{fold_idx}/mean_rank_percentile": mean_rank_percentile,
                }
            )

    # Calculate overall metrics
    cv_src_auc_mean = np.nanmean(fold_metrics["src_auc"])
    cv_src_auc_std = np.nanstd(fold_metrics["src_auc"])
    cv_mean_rank_percentile_mean = np.nanmean(fold_metrics["mean_rank_percentile"])
    cv_mean_rank_percentile_std = np.nanstd(fold_metrics["mean_rank_percentile"])

    if verbose:
        print()
        print("-" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("-" * 60)
        print(f"SRC-AUC: {cv_src_auc_mean:.4f} (+/- {cv_src_auc_std:.4f})")
        print(
            f"Mean Rank Percentile: {cv_mean_rank_percentile_mean:.2f}% (+/- {cv_mean_rank_percentile_std:.2f}%)"
        )
        print("=" * 60)

    # Calculate feature importance statistics
    importance_mean = None
    importance_std = None
    importance_df = None

    if fold_importances:
        fold_importances_array = np.array(fold_importances)
        importance_mean = np.mean(fold_importances_array, axis=0)
        importance_std = np.std(fold_importances_array, axis=0)

        # Create DataFrame with feature importance
        if feature_names is not None and len(feature_names) == len(importance_mean):
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": importance_mean,
                    "importance_std": importance_std,
                }
            ).sort_values("importance_mean", ascending=False)

            if verbose:
                print()
                print("-" * 60)
                print("FEATURE IMPORTANCE (Top 10)")
                print("-" * 60)
                print(f"{'Rank':<6} {'Feature':<40} {'Mean':<12} {'Std':<12}")
                print("-" * 70)
                for i, row in importance_df.head(10).iterrows():
                    print(
                        f"{row.name + 1:<6} {row['feature']:<40} {row['importance_mean']:<12.6f} {row['importance_std']:<12.6f}"
                    )

            # Plot feature importance as horizontal bar chart and save data
            if output_dir is not None:
                try:
                    # Create subdirectory for feature importance
                    feature_importance_dir = output_dir / "feature_importance"
                    feature_importance_dir.mkdir(parents=True, exist_ok=True)

                    # Save data as CSV
                    data_path = feature_importance_dir / "feature_importance.csv"
                    importance_df.to_csv(data_path, index=False)
                    if verbose:
                        print(f"  -> Saved feature importance data to '{data_path}'")

                    # Plot
                    _plot_feature_importance(
                        importance_df=importance_df,
                        output_dir=feature_importance_dir,
                        top_n=15,
                        verbose=verbose,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  -> Error plotting feature importance: {e}")
                        import traceback

                        traceback.print_exc()

    # Plot and save Success Rate curve
    if output_dir is None:
        if verbose:
            print("  -> Skipping Success Rate curve plot (output_dir not provided)")
    elif not success_rate_curves:
        if verbose:
            print(
                f"  -> Skipping Success Rate curve plot (no curves collected: {len(success_rate_curves)})"
            )
    else:
        try:
            # Create subdirectory for success rate curve
            success_rate_dir = output_dir / "success_rate_curve"
            success_rate_dir.mkdir(parents=True, exist_ok=True)

            # Save per-fold data
            for fold_idx, (
                inspected_area_pct,
                detected_positives_pct,
                metrics,
            ) in enumerate(success_rate_curves, 1):
                fold_data = pd.DataFrame(
                    {
                        "inspected_area_pct": inspected_area_pct,
                        "detected_positives_pct": detected_positives_pct,
                    }
                )
                fold_data_path = success_rate_dir / f"fold_{fold_idx}.csv"
                fold_data.to_csv(fold_data_path, index=False)

                # Save metrics as JSON
                metrics_path = success_rate_dir / f"fold_{fold_idx}_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

            if verbose:
                print(
                    f"  -> Saved per-fold success rate curve data to '{success_rate_dir}'"
                )

            # Compute and save mean/std data
            n_points = 100
            area_pct_common = np.linspace(0, 100, n_points)
            detected_pct_interp = []
            success_rate_aucs = []
            for (
                inspected_area_pct,
                detected_positives_pct,
                metrics,
            ) in success_rate_curves:
                detected_pct_interp.append(
                    np.interp(
                        area_pct_common, inspected_area_pct, detected_positives_pct
                    )
                )
                success_rate_aucs.append(metrics["auc"])

            detected_pct_mean = np.mean(detected_pct_interp, axis=0)
            detected_pct_std = np.std(detected_pct_interp, axis=0)
            success_rate_auc_mean = np.mean(success_rate_aucs)
            success_rate_auc_std = np.std(success_rate_aucs)

            mean_std_data = pd.DataFrame(
                {
                    "inspected_area_pct": area_pct_common,
                    "detected_positives_pct_mean": detected_pct_mean,
                    "detected_positives_pct_std": detected_pct_std,
                }
            )
            mean_std_data_path = success_rate_dir / "mean_std.csv"
            mean_std_data.to_csv(mean_std_data_path, index=False)

            summary_metrics = {
                "auc_mean": float(success_rate_auc_mean),
                "auc_std": float(success_rate_auc_std),
                "n_folds": len(success_rate_curves),
            }
            summary_metrics_path = success_rate_dir / "summary_metrics.json"
            with open(summary_metrics_path, "w") as f:
                json.dump(summary_metrics, f, indent=2)

            if verbose:
                print(
                    f"  -> Saved mean/std success rate curve data to '{mean_std_data_path}'"
                )

            # Plot
            _plot_success_rate_curve(
                success_rate_curves=success_rate_curves,
                output_dir=success_rate_dir,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  -> Error plotting Success Rate curve: {e}")
                import traceback

                traceback.print_exc()

    # Plot positives histogram (independent of success rate curve)
    if output_dir is None:
        if verbose:
            print("  -> Skipping positives histogram (output_dir not provided)")
    elif not y_test_list or not y_pred_proba_list:
        if verbose:
            print(
                f"  -> Skipping positives histogram (no data collected: y_test={len(y_test_list)}, y_pred={len(y_pred_proba_list)})"
            )
    else:
        try:
            # Create subdirectory for positives histogram
            positives_histogram_dir = output_dir / "positives_histogram"
            positives_histogram_dir.mkdir(parents=True, exist_ok=True)

            # Save per-fold raw data
            for fold_idx, (y_test, y_pred_proba) in enumerate(
                zip(y_test_list, y_pred_proba_list), 1
            ):
                fold_data = pd.DataFrame(
                    {
                        "y_test": y_test,
                        "y_pred_proba": y_pred_proba,
                    }
                )
                fold_data_path = positives_histogram_dir / f"fold_{fold_idx}_raw.csv"
                fold_data.to_csv(fold_data_path, index=False)

                # Compute and save bin data for this fold
                bin_centers, pct_positives = _compute_positives_per_bin(
                    y_true=y_test,
                    y_pred_proba=y_pred_proba,
                    bin_size_pct=10.0,
                )
                if len(bin_centers) > 0:
                    fold_bin_data = pd.DataFrame(
                        {
                            "bin_center": bin_centers,
                            "pct_positives": pct_positives,
                        }
                    )
                    fold_bin_data_path = (
                        positives_histogram_dir / f"fold_{fold_idx}_bins.csv"
                    )
                    fold_bin_data.to_csv(fold_bin_data_path, index=False)

            if verbose:
                print(
                    f"  -> Saved per-fold positives histogram data to '{positives_histogram_dir}'"
                )

            # Compute mean/std across folds
            all_bin_centers = []
            all_pct_positives = []
            for y_test, y_pred_proba in zip(y_test_list, y_pred_proba_list):
                bin_centers, pct_positives = _compute_positives_per_bin(
                    y_true=y_test,
                    y_pred_proba=y_pred_proba,
                    bin_size_pct=10.0,
                )
                if len(bin_centers) > 0:
                    all_bin_centers.append(bin_centers)
                    all_pct_positives.append(pct_positives)

            if len(all_bin_centers) > 0:
                bin_centers = all_bin_centers[0]
                pct_positives_array = np.array(all_pct_positives)
                pct_positives_mean = np.mean(pct_positives_array, axis=0)
                pct_positives_std = np.std(pct_positives_array, axis=0)

                mean_std_data = pd.DataFrame(
                    {
                        "bin_center": bin_centers,
                        "pct_positives_mean": pct_positives_mean,
                        "pct_positives_std": pct_positives_std,
                    }
                )
                mean_std_data_path = positives_histogram_dir / "mean_std.csv"
                mean_std_data.to_csv(mean_std_data_path, index=False)

                if verbose:
                    print(
                        f"  -> Saved mean/std positives histogram data to '{mean_std_data_path}'"
                    )

            # Plot
            _plot_positives_histogram(
                y_test_list=y_test_list,
                y_pred_proba_list=y_pred_proba_list,
                output_dir=positives_histogram_dir,
                bin_size_pct=10.0,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  -> Error plotting positives histogram: {e}")
                import traceback

                traceback.print_exc()

    # Log overall metrics to wandb
    if wandb_run is not None:
        log_dict = {
            "cv/src_auc_mean": cv_src_auc_mean,
            "cv/src_auc_std": cv_src_auc_std,
            "cv/mean_rank_percentile_mean": cv_mean_rank_percentile_mean,
            "cv/mean_rank_percentile_std": cv_mean_rank_percentile_std,
        }

        # # Log top feature importances to wandb
        # if importance_df is not None:
        #     for i, row in importance_df.head(10).iterrows():
        #         feature_name = row['feature'].replace('/', '_').replace(' ', '_')
        #         log_dict[f"feature_importance/{feature_name}_mean"] = row['importance_mean']
        #         log_dict[f"feature_importance/{feature_name}_std"] = row['importance_std']

        wandb_run.log(log_dict)

    result = {
        "cv_src_auc_mean": cv_src_auc_mean,
        "cv_src_auc_std": cv_src_auc_std,
        "cv_mean_rank_percentile_mean": cv_mean_rank_percentile_mean,
        "cv_mean_rank_percentile_std": cv_mean_rank_percentile_std,
    }

    # Add feature importance to results
    if importance_mean is not None:
        result["feature_importance_mean"] = importance_mean
        result["feature_importance_std"] = importance_std
        if importance_df is not None:
            result["feature_importance_df"] = importance_df

    return result


def train_and_save_final_model(
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    model: Any,
    cfg: DictConfig,
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Any, StandardScaler, Optional[pd.DataFrame]]:
    """Train final model on all data and save to disk.

    Args:
        X_samples: Feature matrix of shape (n_samples, n_features).
        y_samples: Label array of shape (n_samples,). Values: 1=positive, 0=unlabeled.
        model: PU classifier instance to train.
        cfg: Configuration dict to save alongside the model.
        output_dir: Directory to save model, scaler, config, and feature importance.
        feature_names: Optional list of feature names for importance analysis.
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_model, scaler, feature_importance_df).
        feature_importance_df is None if feature_names not provided or importance unavailable.
    """
    # Scale features
    scaler = StandardScaler()
    X_samples_scaled = scaler.fit_transform(X_samples)

    # Train model
    model.fit(X_samples_scaled, y_samples)

    if verbose:
        print("  -> Final model trained on all samples")
        print(f"    - Training samples: {len(X_samples):,}")
        print(f"    - Positive samples: {np.sum(y_samples == 1):,}")
        print(f"    - Unlabeled samples: {np.sum(y_samples == 0):,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    scaler_path = output_dir / "scaler.joblib"
    config_path = output_dir / "config.yaml"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    OmegaConf.save(cfg, config_path)

    if verbose:
        print(f"  -> Saved model to '{model_path}'")
        print(f"  -> Saved scaler to '{scaler_path}'")
        print(f"  -> Saved config to '{config_path}'")

    return model, scaler
