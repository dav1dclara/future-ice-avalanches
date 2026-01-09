"""Data loading utilities for model training."""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio as rio
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from tqdm import tqdm


def create_fold_samples(
    folds_img: np.ndarray,
    labels_img: np.ndarray,
    n_folds: int = 5,
    buffer_pixels: int = 10,
    unlabeled_ratio: float = 5.0,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
    save_outputs: bool = False,
    output_dir: Optional[Path] = None,
    labels_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, int]]]:
    """Sample positive and unlabeled pixels for each fold.

    For each fold, this function:
    1. Extracts labels for the current fold
    2. Applies a buffer around positive pixels to exclude nearby unlabeled pixels
    3. Samples all positive pixels (label=1)
    4. Samples random unlabeled pixels (label=0) with 20% oversampling to account for invalid features

    Args:
        folds_img: Fold assignment array (NaN for pixels not in any fold).
        labels_img: Label array (1=positive, 0=unlabeled, NaN=invalid).
        n_folds: Number of folds to process.
        buffer_pixels: Number of pixels to buffer around positive pixels.
        unlabeled_ratio: Ratio of unlabeled pixels to sample per positive pixel.
        seed: Random seed for sampling.
        n_jobs: Number of parallel jobs. Use -1 for all cores, 1 for sequential.
        verbose: Whether to print progress and summary.
        save_outputs: Whether to save fold_labels_samples rasters to disk.
        output_dir: Directory to save output rasters. Required if save_outputs=True.
        labels_profile: Rasterio profile dict from labels raster. Required if save_outputs=True.

    Returns:
        Tuple of:
        - samples: Dict mapping fold_idx to fold_labels_samples array.
            Values: 1=positive sample, 0=unlabeled sample, NaN=not sampled.
        - sample_counts: Dict mapping fold_idx to counts dict with 'positive' and 'unlabeled' keys.
    """
    if verbose:
        print(f"- n_folds: {n_folds}")
        print(f"- buffer_pixels: {buffer_pixels}")
        print(f"- unlabeled_ratio: {unlabeled_ratio}")
        print(f"- seed: {seed}")
        print()

    # Process folds in parallel or sequentially
    if n_jobs == 1:
        # Sequential processing with progress bar
        results = []
        for fold_idx in tqdm(range(1, n_folds + 1), desc="Processing folds"):
            result = _process_single_fold(
                fold_idx=fold_idx,
                folds_img=folds_img,
                labels_img=labels_img,
                buffer_pixels=buffer_pixels,
                unlabeled_ratio=unlabeled_ratio,
                seed=seed,
            )
            results.append(result)
    else:
        # Parallel processing
        start_time = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_single_fold)(
                fold_idx=fold_idx,
                folds_img=folds_img,
                labels_img=labels_img,
                buffer_pixels=buffer_pixels,
                unlabeled_ratio=unlabeled_ratio,
                seed=seed,
            )
            for fold_idx in range(1, n_folds + 1)
        )
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"-> Completed in {elapsed_time:.2f} seconds")

    # Organize results
    samples = {}
    sample_counts = {}
    for fold_idx, fold_labels_samples, counts in results:
        samples[fold_idx] = fold_labels_samples
        sample_counts[fold_idx] = counts

    # Print sample statistics
    if verbose:
        total_positive = sum(counts["positive"] for counts in sample_counts.values())
        total_unlabeled = sum(counts["unlabeled"] for counts in sample_counts.values())
        total_samples = total_positive + total_unlabeled

        print()
        print("-> Sample statistics:")
        for fold_idx in sorted(sample_counts.keys()):
            fold_counts = sample_counts[fold_idx]
            fold_positive = fold_counts["positive"]
            fold_unlabeled = fold_counts["unlabeled"]
            fold_total = fold_positive + fold_unlabeled
            print(
                f"    Fold {fold_idx}: {fold_total:,} ({fold_positive:,} positive, {fold_unlabeled:,} unlabeled)"
            )
        print(
            f"    Total: {total_samples:,} ({total_positive:,} positive, {total_unlabeled:,} unlabeled)"
        )

    # Save fold_labels_samples for all folds to output directory
    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_outputs=True")
        if labels_profile is None:
            raise ValueError("labels_profile must be provided when save_outputs=True")

        if verbose:
            print()
            print("Saving fold_labels_samples...")
        output_dir.mkdir(parents=True, exist_ok=True)
        nodata_value = -9999.0

        for fold_idx, fold_labels_samples in samples.items():
            fold_labels_samples_path = (
                output_dir / f"fold_labels_samples_fold{fold_idx}.tif"
            )

            # Convert NaN to nodata value for saving (rasterio doesn't support NaN as nodata)
            fold_labels_samples_to_save = np.where(
                np.isnan(fold_labels_samples), nodata_value, fold_labels_samples
            )

            with rio.open(
                fold_labels_samples_path,
                "w",
                driver="GTiff",
                height=fold_labels_samples.shape[0],
                width=fold_labels_samples.shape[1],
                count=1,
                dtype=rio.float32,
                crs=labels_profile["crs"],
                transform=labels_profile["transform"],
                nodata=nodata_value,
                compress="deflate",
                tiled=True,
                blockxsize=512,
                blockysize=512,
            ) as dst:
                dst.write(fold_labels_samples_to_save, 1)
                dst.set_band_description(1, "fold_labels_samples")

            if verbose:
                print(f"  -> Saved fold {fold_idx} to '{fold_labels_samples_path}'")

        if verbose:
            print(f"Saved {len(samples)} fold(s) to '{output_dir}'")

    return samples, sample_counts


def load_features(
    features_dir: Path,
    features: list,
    expected_shape: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load and stack feature rasters.

    Args:
        features_dir: Directory containing feature raster files.
        features: List of feature configs. Each config must have:
            - 'name': feature name (e.g., 'slope', 'elevation')
            - 'sigma': sigma value (e.g., 0, 1, 3, 5)
            - Optional 'stat' and 'window': for zonal statistics (e.g., stat='std', window=5)
              If both are provided, loads: {name}_sigma{sigma}_{stat}_w{window}.tif
              Otherwise loads: {name}_sigma{sigma}.tif
        expected_shape: Optional (height, width) tuple to validate feature shape against.
        verbose: Whether to print loading progress.

    Returns:
        Tuple of:
        - X_all: Stacked feature array of shape (height, width, n_features).
        - valid_features_mask: Boolean mask of pixels where all features are valid (finite).
        - feature_names: List of feature names.

    Raises:
        FileNotFoundError: If features directory or any feature file is not found.
        ValueError: If feature shape doesn't match expected_shape.
    """
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Build feature file paths from config
    feature_files = []
    feature_names = []
    for feat_cfg in features:
        feat_name = feat_cfg.name
        feat_sigma = feat_cfg.sigma

        # Support zonal statistics: check for optional 'stat' and 'window' attributes
        stat = getattr(feat_cfg, "stat", None)
        window = getattr(feat_cfg, "window", None)

        if stat is not None and window is not None:
            # Zonal statistics feature: {name}_sigma{sigma}_{stat}_w{window}.tif
            feat_file = (
                features_dir / f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}.tif"
            )
            feat_display_name = f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}"
        else:
            # Regular feature: {name}_sigma{sigma}.tif
            feat_file = features_dir / f"{feat_name}_sigma{feat_sigma}.tif"
            feat_display_name = f"{feat_name}_sigma{feat_sigma}"

        if not feat_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_file}")

        feature_files.append(feat_file)
        feature_names.append(feat_display_name)

    if verbose:
        print(f"  -> Found {len(feature_files)} feature(s):")
        for name, fpath in zip(feature_names, feature_files):
            print(f"    - {name}: {fpath.name}")

    feature_arrays = []
    for feat_file in feature_files:
        with rio.open(feat_file) as src:
            arr = src.read(1, masked=True).astype(np.float32)
            # Convert masked values to NaN
            arr = np.where(arr.mask, np.nan, arr.data)
            feature_arrays.append(arr)
        # MEMORY OPTIMIZATION: Clear rasterio cache after each file
        # This helps free memory during loading

    # Load all feature rasters
    if verbose:
        print(f"  -> Loaded features from '{features_dir}'")

    # Stack features: shape (height, width, n_features)
    # MEMORY NOTE: This creates a large array in memory
    # For a 20k x 20k raster with 8 features: ~12.8 GB (float32)
    X_all = np.stack(feature_arrays, axis=-1)

    # Clear individual feature arrays to free memory (X_all has the data now)
    del feature_arrays
    import gc

    gc.collect()
    if verbose:
        print(f"    - Stacked features shape: {X_all.shape}")

    # Verify feature shape matches expected shape if provided
    if expected_shape is not None:
        if X_all.shape[:2] != expected_shape:
            raise ValueError(
                f"Feature shape {X_all.shape[:2]} does not match expected shape {expected_shape}"
            )

    # Create mask where all features are valid (not NaN)
    # A pixel is valid if all features at that pixel are finite (not NaN, not inf)
    valid_features_mask = np.all(np.isfinite(X_all), axis=-1)
    n_valid_pixels = np.sum(valid_features_mask)
    if verbose:
        percentage = (n_valid_pixels / valid_features_mask.size) * 100
        print(
            f"    - Valid feature pixels: {n_valid_pixels:,} out of {valid_features_mask.size:,} total ({percentage:.1f}%)"
        )

    return X_all, valid_features_mask, feature_names


def compute_valid_features_mask(
    features_dir: Path,
    features: list,
    expected_shape: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, list]:
    """Compute valid_features_mask without loading all features into memory.

    Checks each feature file individually and combines masks. This is memory-efficient
    compared to load_features() which loads all features into memory.

    Args:
        features_dir: Directory containing feature raster files.
        features: List of feature configs. Each config must have:
            - 'name': feature name (e.g., 'slope', 'elevation')
            - 'sigma': sigma value (e.g., 0, 1, 3, 5)
            - Optional 'stat' and 'window': for zonal statistics (e.g., stat='std', window=5)
              If both are provided, loads: {name}_sigma{sigma}_{stat}_w{window}.tif
              Otherwise loads: {name}_sigma{sigma}.tif
        expected_shape: Optional (height, width) tuple to validate feature shape against.
        verbose: Whether to print loading progress.

    Returns:
        Tuple of:
        - valid_features_mask: Boolean mask of pixels where all features are valid (finite).
        - feature_names: List of feature names.

    Raises:
        FileNotFoundError: If features directory or any feature file is not found.
        ValueError: If feature shape doesn't match expected_shape.
    """
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Build feature file paths from config
    feature_files = []
    feature_names = []
    for feat_cfg in features:
        feat_name = feat_cfg.name
        feat_sigma = feat_cfg.sigma

        # Support zonal statistics: check for optional 'stat' and 'window' attributes
        stat = getattr(feat_cfg, "stat", None)
        window = getattr(feat_cfg, "window", None)

        if stat is not None and window is not None:
            # Zonal statistics feature: {name}_sigma{sigma}_{stat}_w{window}.tif
            feat_file = (
                features_dir / f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}.tif"
            )
            feat_display_name = f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}"
        else:
            # Regular feature: {name}_sigma{sigma}.tif
            feat_file = features_dir / f"{feat_name}_sigma{feat_sigma}.tif"
            feat_display_name = f"{feat_name}_sigma{feat_sigma}"

        if not feat_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_file}")

        feature_files.append(feat_file)
        feature_names.append(feat_display_name)

    if verbose:
        print(f"  -> Found {len(feature_files)} feature(s):")
        for name, fpath in zip(feature_names, feature_files):
            print(f"    - {name}: {fpath.name}")

    # Load each feature individually, compute valid mask, then discard the array
    # This avoids keeping all features in memory at once
    valid_masks = []
    first_shape = None

    for feat_file in feature_files:
        with rio.open(feat_file) as src:
            arr = src.read(1, masked=True).astype(np.float32)
            # Convert masked values to NaN
            arr = np.where(arr.mask, np.nan, arr.data)

            # Track first shape for validation
            if first_shape is None:
                first_shape = arr.shape

            # Check shape consistency
            if arr.shape != first_shape:
                raise ValueError(
                    f"Feature shape mismatch: {feat_file.name} has shape {arr.shape}, "
                    f"expected {first_shape}"
                )

            # Create boolean mask: True where feature is valid (finite)
            valid_mask = np.isfinite(arr)
            valid_masks.append(valid_mask)

        # Memory optimization: arr is automatically freed after context manager

    if verbose:
        print(f"  -> Computed validity masks for {len(feature_files)} feature(s)")

    # Combine all masks: a pixel is valid only if ALL features are valid
    # Stack masks temporarily to use np.all along the feature axis
    valid_masks_stack = np.stack(
        valid_masks, axis=0
    )  # Shape: (n_features, height, width)
    valid_features_mask = np.all(valid_masks_stack, axis=0)  # Shape: (height, width)

    # Clear individual masks and stack to free memory
    del valid_masks, valid_masks_stack
    import gc

    gc.collect()

    # Verify feature shape matches expected shape if provided
    if expected_shape is not None:
        if valid_features_mask.shape != expected_shape:
            raise ValueError(
                f"Feature shape {valid_features_mask.shape} doesn't match expected shape {expected_shape}"
            )

    n_valid_pixels = np.sum(valid_features_mask)
    if verbose:
        percentage = (n_valid_pixels / valid_features_mask.size) * 100
        print(
            f"    - Valid feature pixels: {n_valid_pixels:,} out of {valid_features_mask.size:,} total ({percentage:.1f}%)"
        )

    return valid_features_mask, feature_names


def _process_single_fold(
    fold_idx: int,
    folds_img: np.ndarray,
    labels_img: np.ndarray,
    buffer_pixels: int,
    unlabeled_ratio: float,
    seed: int,
) -> Tuple[int, np.ndarray, Dict[str, int]]:
    """Process a single fold (helper function for parallelization).

    Args:
        fold_idx: Fold index to process.
        folds_img: Fold assignment array.
        labels_img: Label array.
        buffer_pixels: Buffer size around positive pixels.
        unlabeled_ratio: Ratio of unlabeled pixels to sample per positive pixel.
        seed: Random seed.

    Returns:
        Tuple of (fold_idx, fold_labels_samples, sample_counts_dict).
    """
    # Create fold array with NaN outside current fold (avoid full copy)
    fold_mask = folds_img == fold_idx
    fold_labels = np.where(fold_mask, labels_img, np.nan)

    # Create mask of positive pixels
    positive_mask = (fold_labels == 1) & ~np.isnan(fold_labels)

    if buffer_pixels > 0:
        # Use 8-connected structuring element (3x3) for direct neighbors
        struct = np.ones((3, 3), dtype=bool)

        # Dilate positive pixels by buffer_pixels iterations
        buffer_mask = binary_dilation(
            positive_mask, structure=struct, iterations=buffer_pixels
        )

        # Find unlabeled pixels in buffer zone and exclude them
        unlabeled_in_buffer = (fold_labels == 0) & buffer_mask & ~np.isnan(fold_labels)
        fold_labels[unlabeled_in_buffer] = np.nan

    # Get positive pixel count
    n_positive_pixels = np.sum(positive_mask)

    # Get valid unlabeled pixels (excluding buffer zone)
    valid_unlabeled_mask = (fold_labels == 0) & ~np.isnan(fold_labels)

    # Calculate target number of unlabeled pixels with 20% oversampling
    # This accounts for potential invalid features that will be filtered later
    n_unlabeled_pixels_target = int(n_positive_pixels * unlabeled_ratio * 1.2)

    # Get all valid unlabeled pixel coordinates
    valid_unlabeled_coords = np.column_stack(np.where(valid_unlabeled_mask))
    n_valid_unlabeled = len(valid_unlabeled_coords)

    # Sample random pixels (with oversampling)
    n_unlabeled_pixels_sample = min(n_unlabeled_pixels_target, n_valid_unlabeled)

    rng = np.random.default_rng(seed + fold_idx)  # Different seed per fold
    if n_unlabeled_pixels_sample > 0:
        sampled_indices = rng.choice(
            n_valid_unlabeled, size=n_unlabeled_pixels_sample, replace=False
        )
        sampled_coords = valid_unlabeled_coords[sampled_indices]

        # Mark sampled pixels
        unlabeled_patch_mask = np.zeros_like(fold_labels, dtype=bool)
        unlabeled_patch_mask[sampled_coords[:, 0], sampled_coords[:, 1]] = True
    else:
        unlabeled_patch_mask = np.zeros_like(fold_labels, dtype=bool)

    n_unlabeled_pixels_sampled = np.sum(unlabeled_patch_mask)

    # Create fold_labels_samples
    fold_labels_samples = np.full_like(fold_labels, np.nan)
    fold_labels_samples[positive_mask] = 1
    fold_labels_samples[unlabeled_patch_mask] = 0

    sample_counts = {
        "positive": n_positive_pixels,
        "unlabeled": n_unlabeled_pixels_sampled,
    }

    return fold_idx, fold_labels_samples, sample_counts


def extract_sample_features_lazy(
    features_dir: Path,
    features: list,
    samples: Dict[int, np.ndarray],
    feature_names: list,
    unlabeled_ratio: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for samples by loading features lazily (one at a time).

    This is memory-efficient: only loads one feature at a time and extracts values
    for sampled pixels, rather than loading all features into memory.

    After filtering valid samples, keeps only the required number of unlabeled samples
    to maintain the specified unlabeled_ratio per fold.

    Args:
        features_dir: Directory containing feature raster files.
        features: List of feature configs (same format as load_features).
        samples: Dict mapping fold_idx to fold_labels_samples array.
            Values: 1=positive sample, 0=unlabeled sample, NaN=not sampled.
        feature_names: List of feature names (for display).
        unlabeled_ratio: Ratio of unlabeled samples to keep per positive sample.
        seed: Random seed for selecting unlabeled samples.
        verbose: Whether to print progress.

    Returns:
        Tuple of:
        - X_samples: Feature matrix of shape (n_final_samples, n_features).
        - y_samples: Label array of shape (n_final_samples,). Values: 1=positive, 0=unlabeled.
        - fold_samples: Fold assignment array of shape (n_final_samples,).
    """
    # First, collect all sample pixel coordinates and labels per fold
    sample_coords_per_fold = {}
    sample_labels_per_fold = {}
    sample_fold_numbers_per_fold = {}

    for fold_idx in sorted(samples.keys()):
        fold_labels_samples = samples[fold_idx]
        sample_mask = ~np.isnan(fold_labels_samples)
        n_samples = np.sum(sample_mask)

        if n_samples == 0:
            continue

        # Get pixel coordinates for all samples in this fold
        coords = np.column_stack(np.where(sample_mask))  # Shape: (n_samples, 2)
        labels = fold_labels_samples[sample_mask].astype(int)  # Shape: (n_samples,)
        fold_numbers = np.full(n_samples, fold_idx, dtype=int)  # Shape: (n_samples,)

        sample_coords_per_fold[fold_idx] = coords
        sample_labels_per_fold[fold_idx] = labels
        sample_fold_numbers_per_fold[fold_idx] = fold_numbers

    if not sample_coords_per_fold:
        raise ValueError("No samples found across all folds!")

    # Get total number of samples (before filtering)
    total_samples_before = sum(
        len(coords) for coords in sample_coords_per_fold.values()
    )

    if verbose:
        print(f"- Total samples before filtering: {total_samples_before:,}")

    # Build feature file paths (same logic as load_features)
    feature_files = []
    for feat_cfg in features:
        feat_name = feat_cfg.name
        feat_sigma = feat_cfg.sigma

        stat = getattr(feat_cfg, "stat", None)
        window = getattr(feat_cfg, "window", None)

        if stat is not None and window is not None:
            feat_file = (
                features_dir / f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}.tif"
            )
        else:
            feat_file = features_dir / f"{feat_name}_sigma{feat_sigma}.tif"

        if not feat_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_file}")

        feature_files.append(feat_file)

    print()

    # Load features one at a time and extract values for sampled pixels
    all_feature_values = []  # List of arrays, each shape (total_samples,)

    for feat_idx, feat_file in enumerate(feature_files):
        if verbose:
            print(
                f"-> Loading feature {feat_idx + 1}/{len(feature_files)}: {feature_names[feat_idx]}"
            )

        # Load feature raster
        with rio.open(feat_file) as src:
            arr = src.read(1, masked=True).astype(np.float32)
            arr = np.where(arr.mask, np.nan, arr.data)

        # Extract values for all sampled pixels across all folds
        feature_values = []
        for fold_idx in sorted(sample_coords_per_fold.keys()):
            coords = sample_coords_per_fold[fold_idx]
            # Extract values at these coordinates
            values = arr[coords[:, 0], coords[:, 1]]  # Shape: (n_samples_in_fold,)
            feature_values.append(values)

        # Concatenate across all folds
        all_feature_values.append(np.hstack(feature_values))  # Shape: (total_samples,)

    # Stack all features: shape (total_samples, n_features)
    X_samples_all = np.column_stack(all_feature_values)

    # Concatenate labels and fold numbers
    y_samples_all = np.hstack(
        [
            sample_labels_per_fold[fold_idx]
            for fold_idx in sorted(sample_labels_per_fold.keys())
        ]
    )
    fold_samples_all = np.hstack(
        [
            sample_fold_numbers_per_fold[fold_idx]
            for fold_idx in sorted(sample_fold_numbers_per_fold.keys())
        ]
    )

    print()

    # Filter out samples with any NaN values
    valid_mask = np.all(np.isfinite(X_samples_all), axis=1)  # Shape: (total_samples,)
    n_valid_samples = np.sum(valid_mask)
    n_invalid_samples = total_samples_before - n_valid_samples

    if verbose:
        print(f"-> Valid samples (all features finite): {n_valid_samples:,}")
        print(
            f"-> Invalid samples (some features NaN): {n_invalid_samples:,} ({n_invalid_samples / total_samples_before * 100:.1f}%)"
        )

    print()

    # Filter to only valid samples
    X_samples = X_samples_all[valid_mask]  # Shape: (n_valid_samples, n_features)
    y_samples = y_samples_all[valid_mask]  # Shape: (n_valid_samples,)
    fold_samples = fold_samples_all[valid_mask]  # Shape: (n_valid_samples,)

    # For each fold, keep all positive samples and randomly select unlabeled samples to match ratio
    final_X_samples = []
    final_y_samples = []
    final_fold_samples = []

    rng = np.random.default_rng(seed)

    for fold_idx in sorted(sample_coords_per_fold.keys()):
        fold_mask = fold_samples == fold_idx
        fold_indices = np.where(fold_mask)[0]

        # Get positive and unlabeled samples for this fold
        fold_positive_mask = y_samples[fold_indices] == 1
        fold_unlabeled_mask = y_samples[fold_indices] == 0

        positive_indices = fold_indices[fold_positive_mask]
        unlabeled_indices = fold_indices[fold_unlabeled_mask]

        n_positive = len(positive_indices)
        n_unlabeled_target = int(n_positive * unlabeled_ratio)
        n_unlabeled_available = len(unlabeled_indices)

        # Keep all positive samples
        selected_positive = positive_indices

        # Randomly select unlabeled samples to match ratio
        if n_unlabeled_target > 0 and n_unlabeled_available > 0:
            n_unlabeled_to_keep = min(n_unlabeled_target, n_unlabeled_available)
            # Use fold-specific seed for reproducibility
            rng_fold = np.random.default_rng(seed + fold_idx)
            selected_unlabeled_idx = rng_fold.choice(
                n_unlabeled_available, size=n_unlabeled_to_keep, replace=False
            )
            selected_unlabeled = unlabeled_indices[selected_unlabeled_idx]
        else:
            selected_unlabeled = np.array([], dtype=int)

        # Combine positive and selected unlabeled samples
        selected_indices = np.concatenate([selected_positive, selected_unlabeled])

        final_X_samples.append(X_samples[selected_indices])
        final_y_samples.append(y_samples[selected_indices])
        final_fold_samples.append(fold_samples[selected_indices])

    # Combine all folds
    X_samples_final = np.vstack(final_X_samples)
    y_samples_final = np.hstack(final_y_samples)
    fold_samples_final = np.hstack(final_fold_samples)

    # Print final per-fold statistics
    if verbose:
        print("-> Per-fold statistics (after ratio adjustment):")
        for fold_idx in sorted(sample_coords_per_fold.keys()):
            fold_mask = fold_samples_final == fold_idx
            n_fold_final = np.sum(fold_mask)
            n_fold_positive = np.sum((fold_mask) & (y_samples_final == 1))
            n_fold_unlabeled = np.sum((fold_mask) & (y_samples_final == 0))
            actual_ratio = (
                n_fold_unlabeled / n_fold_positive if n_fold_positive > 0 else 0
            )
            print(
                f"    Fold {fold_idx}: {n_fold_final:,} samples ({n_fold_positive:,} positive, {n_fold_unlabeled:,} unlabeled, ratio={actual_ratio:.2f})"
            )
        print(
            f"    Total: {n_valid_samples:,} samples ({np.sum(y_samples_final == 1):,} positive, {np.sum(y_samples_final == 0):,} unlabeled, ratio={np.sum(y_samples_final == 0) / np.sum(y_samples_final == 1):.2f})"
        )

        print()

    return X_samples_final, y_samples_final, fold_samples_final


def extract_sample_features(
    X_all: np.ndarray,
    samples: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all samples across all folds.

    Args:
        X_all: Feature array of shape (height, width, n_features).
        samples: Dict mapping fold_idx to fold_labels_samples array.
            Values: 1=positive sample, 0=unlabeled sample, NaN=not sampled.

    Returns:
        Tuple of:
        - X_samples: Feature matrix of shape (n_total_samples, n_features).
        - y_samples: Label array of shape (n_total_samples,). Values: 1=positive, 0=unlabeled.
        - fold_samples: Fold assignment array of shape (n_total_samples,).

    Raises:
        ValueError: If no samples are found across all folds.
    """
    all_samples_features = []
    all_samples_labels = []
    all_samples_folds = []

    for fold_idx in sorted(samples.keys()):
        fold_labels_samples = samples[fold_idx]

        # Find all pixels with samples (not NaN)
        sample_mask = ~np.isnan(fold_labels_samples)
        n_samples = np.sum(sample_mask)

        if n_samples == 0:
            continue

        # Extract features for these pixels
        # X_all[sample_mask] gives shape (n_samples, n_features)
        fold_features = X_all[sample_mask]

        # Extract labels (1 for positive, 0 for unlabeled)
        fold_labels = fold_labels_samples[sample_mask].astype(int)

        # Store fold number
        fold_numbers = np.full(n_samples, fold_idx, dtype=int)

        all_samples_features.append(fold_features)
        all_samples_labels.append(fold_labels)
        all_samples_folds.append(fold_numbers)

    # Combine all folds
    if not all_samples_features:
        raise ValueError("No samples found across all folds!")

    X_samples = np.vstack(all_samples_features)  # Shape: (n_total_samples, n_features)
    y_samples = np.hstack(all_samples_labels)  # Shape: (n_total_samples,)
    fold_samples = np.hstack(all_samples_folds)  # Shape: (n_total_samples,)

    return X_samples, y_samples, fold_samples


def filter_features_by_set(features: list, feature_set: int) -> list:
    """Filter features based on feature_set parameter.

    Feature set encoding:
    - Sets 1-12: Standard combinations
      - Elevation: sets 1-6 = no elevation, sets 7-12 = with elevation
      - Zonal stats: sets 1-3,7-9 = no zonal, sets 4-6,10-12 = with zonal
      - Sigmas: sets 1,4,7,10 = sigma 1 only; sets 2,5,8,11 = sigma 1 and 3; sets 3,6,9,12 = all sigmas (1,3,5)
    - Sets 13-18: Custom feature name combinations (sigma 1 for terrain, sigma 0 for elevation, no zonal)
      - 13: slope
      - 14: slope, slope_break_index
      - 15: profile_curvature, planform_curvature, mean_curvature
      - 16: slope, profile_curvature, planform_curvature, mean_curvature
      - 17: slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature
      - 18: elevation, slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature
    - Sets 19-24: Fixed feature set with varying sigma combinations (no elevation, no zonal)
      - Features: slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature
      - 19: sigma [1]
      - 20: sigma [3]
      - 21: sigma [5]
      - 22: sigma [1,3]
      - 23: sigma [3,5]
      - 24: sigma [1,3,5]
    - Sets 25-30: Feature set 24 (sigma [1,3,5]) with varying zonal statistics
      - Base features: slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [1,3,5])
      - 25: zonal std, window [3]
      - 26: zonal range, window [3]
      - 27: zonal std and range, window [3]
      - 28: zonal std, window [3,5,7]
      - 29: zonal range, window [3,5,7]
      - 30: zonal std and range, window [3,5,7]

    Args:
        features: List of feature configs from config file
        feature_set: Integer from 1-30 indicating which feature combination to use

    Returns:
        Filtered list of feature configs
    """
    # Handle zonal statistics feature sets (25-30) - feature set 24 + zonal stats
    if feature_set in [25, 26, 27, 28, 29, 30]:
        # Base features from set 24: slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [1,3,5])
        allowed_feature_names = [
            "slope",
            "slope_break_index",
            "profile_curvature",
            "planform_curvature",
            "mean_curvature",
        ]
        base_sigmas = [1, 3, 5]

        # Define zonal statistics configuration for each set
        zonal_configs = {
            25: {"stats": ["std"], "windows": [3]},
            26: {"stats": ["range"], "windows": [3]},
            27: {"stats": ["std", "range"], "windows": [3]},
            28: {"stats": ["std"], "windows": [3, 5, 7]},
            29: {"stats": ["range"], "windows": [3, 5, 7]},
            30: {"stats": ["std", "range"], "windows": [3, 5, 7]},
        }
        zonal_config = zonal_configs[feature_set]
        allowed_zonal_stats = zonal_config["stats"]
        allowed_zonal_windows = zonal_config["windows"]

        filtered = []
        for feat in features:
            feat_name = feat.name
            feat_sigma = feat.sigma
            has_stat = hasattr(feat, "stat") and feat.stat is not None
            has_window = hasattr(feat, "window") and feat.window is not None
            is_zonal = has_stat and has_window

            # Exclude elevation (regular or zonal)
            if feat_name == "elevation":
                continue

            # Handle zonal statistics
            if is_zonal:
                # Check if feature name is in allowed list (zonal stats are computed on sigma 1 features)
                if feat_name not in allowed_feature_names:
                    continue
                # Zonal stats use sigma 1 for terrain features
                if feat_sigma != 1:
                    continue
                # Check if stat and window match the configuration
                feat_stat = feat.stat
                feat_window = feat.window
                if (
                    feat_stat in allowed_zonal_stats
                    and feat_window in allowed_zonal_windows
                ):
                    filtered.append(feat)
                continue

            # Handle regular (non-zonal) features - include all from set 24
            if feat_name not in allowed_feature_names:
                continue

            # Check if sigma is in allowed list (sigma [1,3,5] for set 24)
            if feat_sigma in base_sigmas:
                filtered.append(feat)

        return filtered

    # Handle sigma variation feature sets (19-24)
    if feature_set in [19, 20, 21, 22, 23, 24]:
        # Fixed feature names: slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature
        allowed_feature_names = [
            "slope",
            "slope_break_index",
            "profile_curvature",
            "planform_curvature",
            "mean_curvature",
        ]

        # Define allowed sigmas for each set
        sigma_sets = {
            19: [1],
            20: [3],
            21: [5],
            22: [1, 3],
            23: [3, 5],
            24: [1, 3, 5],
        }
        allowed_sigmas = sigma_sets[feature_set]

        filtered = []
        for feat in features:
            feat_name = feat.name
            feat_sigma = feat.sigma
            has_stat = hasattr(feat, "stat") and feat.stat is not None
            has_window = hasattr(feat, "window") and feat.window is not None
            is_zonal = has_stat and has_window

            # Exclude all zonal statistics
            if is_zonal:
                continue

            # Exclude elevation
            if feat_name == "elevation":
                continue

            # Check if feature name is in allowed list
            if feat_name not in allowed_feature_names:
                continue

            # Check if sigma is in allowed list
            if feat_sigma in allowed_sigmas:
                filtered.append(feat)

        return filtered

    # Handle custom feature sets (13-18)
    if feature_set in [13, 14, 15, 16, 17, 18]:
        # Define allowed feature names for each custom set
        custom_feature_sets = {
            13: ["slope"],
            14: ["slope", "slope_break_index"],
            15: ["profile_curvature", "planform_curvature", "mean_curvature"],
            16: ["slope", "profile_curvature", "planform_curvature", "mean_curvature"],
            17: [
                "slope",
                "slope_break_index",
                "profile_curvature",
                "planform_curvature",
                "mean_curvature",
            ],
            18: [
                "elevation",
                "slope",
                "slope_break_index",
                "profile_curvature",
                "planform_curvature",
                "mean_curvature",
            ],
        }
        allowed_feature_names = custom_feature_sets[feature_set]

        filtered = []
        for feat in features:
            feat_name = feat.name
            feat_sigma = feat.sigma
            has_stat = hasattr(feat, "stat") and feat.stat is not None
            has_window = hasattr(feat, "window") and feat.window is not None
            is_zonal = has_stat and has_window

            # Exclude all zonal statistics
            if is_zonal:
                continue

            # Check if feature name is in allowed list
            if feat_name not in allowed_feature_names:
                continue

            # Elevation uses sigma 0, all others use sigma 1
            if feat_name == "elevation":
                if feat_sigma == 0:
                    filtered.append(feat)
            else:
                if feat_sigma == 1:
                    filtered.append(feat)

        return filtered

    # Original logic for sets 1-12
    # Determine settings from feature_set
    include_elevation = feature_set >= 7
    include_zonal = feature_set in [4, 5, 6, 10, 11, 12]

    # Determine sigma filter for regular features (not zonal stats)
    if feature_set in [1, 4, 7, 10]:
        allowed_sigmas = [1]  # Sigma 1 only
    elif feature_set in [2, 5, 8, 11]:
        allowed_sigmas = [1, 3]  # Sigma 1 and 3
    else:  # feature_set in [3, 6, 9, 12]
        allowed_sigmas = [1, 3, 5]  # All sigmas

    filtered = []
    for feat in features:
        feat_name = feat.name
        feat_sigma = feat.sigma
        has_stat = hasattr(feat, "stat") and feat.stat is not None
        has_window = hasattr(feat, "window") and feat.window is not None
        is_zonal = has_stat and has_window

        # Handle elevation (sigma 0, regular feature, not zonal)
        if feat_name == "elevation" and feat_sigma == 0 and not is_zonal:
            if not include_elevation:
                continue
            # Elevation is always included if include_elevation is True (no sigma filter for elevation)
            filtered.append(feat)
            continue

        # Filter zonal statistics
        if is_zonal:
            if not include_zonal:
                continue
            # For zonal stats, check if the base feature's sigma is allowed
            # Zonal stats typically use sigma 0 or 1 for the base feature
            # If sigma is 0 (elevation zonal), always include if zonal is enabled
            if feat_sigma == 0:
                filtered.append(feat)
            elif feat_sigma in allowed_sigmas:
                filtered.append(feat)
            continue

        # Filter regular features by sigma
        if feat_sigma in allowed_sigmas:
            filtered.append(feat)

    return filtered
