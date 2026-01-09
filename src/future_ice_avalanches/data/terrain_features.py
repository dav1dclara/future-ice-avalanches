"""Terrain feature extraction functions for computing terrain attributes at multiple scales."""

import inspect
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import rasterio as rio
from scipy.ndimage import (
    convolve,
    gaussian_filter,
    generic_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)
from skimage.filters.rank import maximum as rank_max
from skimage.filters.rank import mean as rank_mean
from skimage.filters.rank import median as rank_median
from skimage.filters.rank import minimum as rank_min
from skimage.morphology import disk
from tqdm import tqdm


def compute_slope(dem, pixel_size_x, pixel_size_y, sigma):
    """Compute slope at a given sigma using gradient."""
    # Smooth DEM at the specified sigma
    smoothed = gaussian_filter(dem, sigma=sigma)

    # Compute gradients
    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)

    # Slope in degrees: atan(sqrt(dx^2 + dy^2))
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180.0 / np.pi

    return slope


def compute_slope_break_index(dem, pixel_size_x, pixel_size_y, sigma):
    # Smooth DEM at the specified sigma
    # Use minimum sigma of 0.5 for very small sigmas to reduce noise
    sigma = max(sigma, 0.5)
    smoothed = gaussian_filter(dem, sigma=sigma)

    # Compute slope first
    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180.0 / np.pi

    # Compute gradient of slope (rate of change of slope)
    # This measures where slope changes rapidly
    dslope_dy, dslope_dx = np.gradient(slope, pixel_size_y, pixel_size_x)

    # Slope break index = magnitude of slope gradient
    # Higher values = more rapid slope changes = terrain breaks
    # Units: degrees per meter
    slope_break = np.sqrt(dslope_dx**2 + dslope_dy**2)

    return slope_break


def compute_profile_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    smoothed = gaussian_filter(dem, sigma=sigma)

    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)
    dyy, dyx = np.gradient(dy, pixel_size_y, pixel_size_x)
    dxy, dxx = np.gradient(dx, pixel_size_y, pixel_size_x)

    g = dx**2 + dy**2
    denominator = g**1.5
    denominator = np.where(denominator == 0, np.nan, denominator)

    num = dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2
    return num / denominator


def compute_planform_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    smoothed = gaussian_filter(dem, sigma=sigma)

    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)
    dyy, dyx = np.gradient(dy, pixel_size_y, pixel_size_x)
    dxy, dxx = np.gradient(dx, pixel_size_y, pixel_size_x)

    g = dx**2 + dy**2
    denominator = np.sqrt(g)
    denominator = np.where(denominator == 0, np.nan, denominator)

    num = dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2
    return num / denominator


def compute_mean_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    smoothed = gaussian_filter(dem, sigma=sigma)

    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)
    dyy, dyx = np.gradient(dy, pixel_size_y, pixel_size_x)
    dxy, dxx = np.gradient(dx, pixel_size_y, pixel_size_x)

    g = dx**2 + dy**2
    denominator = (g + 1) ** 1.5
    denominator = np.where(denominator == 0, np.nan, denominator)

    num = dxx * (1 + dy**2) - 2 * dxy * dx * dy + dyy * (1 + dx**2)
    return num / (2 * denominator)


def compute_gaussian_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    smoothed = gaussian_filter(dem, sigma=sigma)

    dy, dx = np.gradient(smoothed, pixel_size_y, pixel_size_x)
    dyy, dyx = np.gradient(dy, pixel_size_y, pixel_size_x)
    dxy, dxx = np.gradient(dx, pixel_size_y, pixel_size_x)

    g = dx**2 + dy**2
    denominator = (g + 1) ** 2
    denominator = np.where(denominator == 0, np.nan, denominator)

    num = dxx * dyy - dxy**2
    return num / denominator


def compute_max_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    H = compute_mean_curvature(dem, pixel_size_x, pixel_size_y, sigma)
    K = compute_gaussian_curvature(dem, pixel_size_x, pixel_size_y, sigma)

    inside = H**2 - K
    inside = np.where(inside < 0, 0, inside)  # numerical stability

    return H + np.sqrt(inside)


def compute_min_curvature(dem, pixel_size_x, pixel_size_y, sigma):
    H = compute_mean_curvature(dem, pixel_size_x, pixel_size_y, sigma)
    K = compute_gaussian_curvature(dem, pixel_size_x, pixel_size_y, sigma)

    inside = H**2 - K
    inside = np.where(inside < 0, 0, inside)  # numerical stability

    return H - np.sqrt(inside)


def compute_elevation(dem, sigma):
    """Return elevation (DEM) with optional Gaussian smoothing.

    When sigma is 0, returns the raw DEM without processing.
    When sigma > 0, applies Gaussian filtering at the specified sigma.
    Useful for including elevation at different scales as a feature.

    Args:
        dem: Input DEM array
        sigma: Gaussian filter standard deviation. If 0, returns raw DEM.

    Returns:
        The DEM array, optionally smoothed with Gaussian filter
    """
    dem = dem.astype(np.float64)

    if sigma > 0:
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(dem, sigma=sigma)
        return smoothed
    else:
        # Return raw elevation
        return dem


def compute_zonal_statistic(
    raster: np.ndarray,
    window_radius: int,
    statistic: str,
    nodata=np.nan,
):
    """
    FAST, NaN-safe circular-window zonal statistics.
    Correctly rescales float data for skimage rank filters.

    Only computes statistics for interior pixels where:
    1. The circular window is completely within raster bounds (not on edges)
    2. ALL pixels within the circular window have valid data (no nodata)

    Edge pixels and pixels with any nodata in their window are set to nodata.

    Args:
        raster: Input raster array.
        window_radius: Radius of circular window in pixels.
        statistic: Statistic to compute ('mean', 'median', 'min', 'max', 'range', 'std', 'var').
        nodata: Nodata value in input raster (default: np.nan).

    Returns:
        Array with computed statistic, with edge pixels and pixels with incomplete window coverage set to nodata.
    """

    # --- 1. Normalize & mask nodata ---
    arr = raster.astype(np.float32)
    input_valid_mask = ~np.isnan(arr) if np.isnan(nodata) else (arr != nodata)

    # Replace nodata with zero (used for rank filters but ignored by mask)
    arr_filled = np.where(input_valid_mask, arr, 0).astype(np.float32)

    # Circular footprint
    footprint = disk(window_radius)
    expected_count = np.sum(footprint).astype(
        np.float32
    )  # Total pixels in circular window

    # Create interior pixel mask: only pixels where the circular window is completely within bounds
    # A pixel is interior if it's at least window_radius pixels away from all edges
    height, width = raster.shape
    interior_mask = np.ones((height, width), dtype=bool)

    # Mask pixels too close to edges
    interior_mask[:window_radius, :] = False  # Top edge
    interior_mask[-window_radius:, :] = False  # Bottom edge
    interior_mask[:, :window_radius] = False  # Left edge
    interior_mask[:, -window_radius:] = False  # Right edge

    # Count of valid pixels in each window (only count input-valid pixels)
    valid_count = convolve(
        input_valid_mask.astype(np.float32), footprint, mode="constant", cval=0
    )

    # Final mask: pixel must be interior AND all pixels within window must be valid
    # This means valid_count must equal expected_count (100% coverage of valid data)
    full_coverage_mask = (valid_count == expected_count) & interior_mask

    # Utility: rescale to uint16 safely
    # Use fewer bins (256) for better performance while maintaining reasonable precision
    def to_uint16(a, n_bins=256):
        vmin = np.nanmin(a[input_valid_mask])
        vmax = np.nanmax(a[input_valid_mask])
        if vmax == vmin or not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.zeros_like(a, dtype=np.uint16), vmin, vmax
        # Scale to 0-255 range (or n_bins-1) for better performance
        max_val = n_bins - 1
        scaled = (a - vmin) / (vmax - vmin) * max_val
        # Handle NaN and inf values before casting
        scaled = np.where(np.isfinite(scaled), scaled, 0)
        scaled = np.clip(scaled, 0, max_val)
        return scaled.astype(np.uint16), vmin, vmax

    # -----------------------------
    # RANK FILTERS (min/max/median/mean)
    # -----------------------------

    if statistic in ("mean", "median", "min", "max", "range"):
        # Use 256 bins for better performance (can be adjusted if needed)
        n_bins = 256
        arr_uint, vmin, vmax = to_uint16(arr_filled, n_bins=n_bins)
        max_bin_val = n_bins - 1

        if statistic == "mean":
            s = rank_mean(arr_uint, footprint=footprint).astype(np.float32)
            # Convert back to original scale
            mean_vals = vmin + (s / max_bin_val) * (vmax - vmin)
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, mean_vals, np.nan)

        if statistic == "median":
            med = rank_median(arr_uint, footprint=footprint).astype(np.float32)
            med_rescaled = vmin + (med / max_bin_val) * (vmax - vmin)
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, med_rescaled, np.nan)

        if statistic == "min":
            mn = rank_min(arr_uint, footprint=footprint).astype(np.float32)
            mn_rescaled = vmin + (mn / max_bin_val) * (vmax - vmin)
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, mn_rescaled, np.nan)

        if statistic == "max":
            mx = rank_max(arr_uint, footprint=footprint).astype(np.float32)
            mx_rescaled = vmin + (mx / max_bin_val) * (vmax - vmin)
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, mx_rescaled, np.nan)

        if statistic == "range":
            mn = rank_min(arr_uint, footprint=footprint).astype(np.float32)
            mx = rank_max(arr_uint, footprint=footprint).astype(np.float32)

            mn_rescaled = vmin + (mn / max_bin_val) * (vmax - vmin)
            mx_rescaled = vmin + (mx / max_bin_val) * (vmax - vmin)

            rng = mx_rescaled - mn_rescaled
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, rng, np.nan)

    # -----------------------------
    # STD / VAR
    # -----------------------------

    if statistic in ("std", "var"):
        # Sum in window
        sum_vals = convolve(arr_filled, footprint, mode="constant", cval=0)

        # Mean (only valid where we have full coverage)
        mean_vals = sum_vals / expected_count

        # Sum of squares
        sum_vals2 = convolve(
            arr_filled * arr_filled, footprint, mode="constant", cval=0
        )

        # Variance = E[x^2] - E[x]^2 (only valid where we have full coverage)
        var_vals = (sum_vals2 / expected_count) - mean_vals**2
        var_vals = np.maximum(var_vals, 0)

        if statistic == "var":
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, var_vals, np.nan)

        if statistic == "std":
            # Only keep pixels where all data in window is valid
            return np.where(full_coverage_mask, np.sqrt(var_vals), np.nan)

    raise ValueError(f"Unsupported statistic: {statistic}")


def compute_zonal_statistics_for_features(
    features_dir: Path,
    zonal_config: Dict,
    terrain_attrs: List[Dict],
) -> None:
    """
    Compute zonal statistics for specified features over different window sizes.

    Args:
        features_dir: Directory containing feature rasters
        zonal_config: Configuration dict with:
            - 'window_sizes' or 'windows': list of window sizes (in pixels)
            - 'stats': list of statistics to compute (e.g., ['std', 'mean'])
            - 'apply_to': dict mapping feature names to their sigmas, e.g.:
                {'slope': {'sigmas': [1, 3]}, 'elevation': {'sigmas': [0]}}
            OR (legacy):
            - 'include_features': list of feature names (uses all sigmas from terrain_attrs)
        terrain_attrs: List of terrain attribute configs (used for legacy mode)
    """
    # Handle OmegaConf DictConfig - use getattr for safe access
    # Support both "windows" and "window_sizes" for backward compatibility
    windows = list(
        getattr(zonal_config, "window_sizes", getattr(zonal_config, "windows", []))
    )
    stats = list(getattr(zonal_config, "stats", []))

    # Check for new structure: apply_to
    apply_to = getattr(zonal_config, "apply_to", None)

    # Build feature-sigma mapping
    feature_sigmas = {}

    if apply_to is not None:
        # New structure: apply_to dict with feature-specific sigmas
        print("\n=== Zonal Statistics Configuration (apply_to structure) ===")
        for feature_name, feature_config in apply_to.items():
            sigmas = list(getattr(feature_config, "sigmas", []))
            feature_sigmas[feature_name] = sigmas
            print(f"  {feature_name}: sigmas={sigmas}")
    else:
        # Legacy structure: include_features list
        include_features = list(getattr(zonal_config, "include_features", []))
        # Build a map of feature names to their sigmas from terrain_attrs
        for attr_config in terrain_attrs:
            attr_name = getattr(attr_config, "name", None)
            if attr_name and attr_name in include_features:
                sigmas = list(getattr(attr_config, "sigmas", []))
                feature_sigmas[attr_name] = sigmas
        print("\n=== Zonal Statistics Configuration (legacy include_features) ===")
        for feature_name in include_features:
            if feature_name in feature_sigmas:
                print(f"  {feature_name}: sigmas={feature_sigmas[feature_name]}")

    if not feature_sigmas or not windows or not stats:
        print("No zonal statistics to compute (empty config).")
        return

    print(f"\nWindow sizes: {windows}")
    print(f"Statistics: {stats}")
    print(f"\n=== Features that would be computed ===")

    # Count and list all computations
    total_computations = 0
    for feature_name, sigmas in feature_sigmas.items():
        print(f"\n{feature_name}:")
        for sigma in sigmas:
            feature_path = features_dir / f"{feature_name}_sigma{sigma}.tif"
            exists = feature_path.exists()
            status = "✓ exists" if exists else "✗ missing"
            print(f"  sigma={sigma}: {feature_path.name} {status}")

            for window in windows:
                for stat in stats:
                    out_path = (
                        features_dir
                        / f"{feature_name}_sigma{sigma}_{stat}_w{window}.tif"
                    )
                    out_exists = out_path.exists()
                    out_status = "✓ exists" if out_exists else "→ would compute"
                    print(f"    {stat}_w{window}: {out_path.name} {out_status}")
                    total_computations += 1

    print(f"\n=== Summary ===")
    print(f"Total zonal statistics to compute: {total_computations}")
    print(f"Features: {len(feature_sigmas)}")
    print(f"Window sizes: {len(windows)}")
    print(f"Statistics: {len(stats)}")

    # COMMENTED OUT: Actual computation
    # Uncomment below to enable computation

    if total_computations == 0:
        print("No valid zonal statistics to compute.")
        return

    print(f"\nComputing zonal statistics for {len(feature_sigmas)} features...")
    pbar = tqdm(
        total=total_computations,
        desc="Computing zonal statistics",
        unit="stat",
    )

    for feature_name, sigmas in feature_sigmas.items():
        for sigma in sigmas:
            # Load the base feature raster
            feature_path = features_dir / f"{feature_name}_sigma{sigma}.tif"

            if not feature_path.exists():
                print(f"Warning: Feature file not found: {feature_path}, skipping...")
                continue

            # Load feature raster and get its profile
            with rio.open(feature_path) as src:
                feature_raster = src.read(1).astype(np.float64)
                profile = src.profile.copy()
                nodata_value = src.nodata

            # Compute each statistic for each window size
            for window in windows:
                for stat in stats:
                    # Output path: feature_sigma{sigma}_{stat}_w{window}.tif
                    out_path = (
                        features_dir
                        / f"{feature_name}_sigma{sigma}_{stat}_w{window}.tif"
                    )

                    # Skip if already exists
                    if out_path.exists():
                        pbar.update(1)
                        pbar.set_postfix_str(f"Skipped {out_path.name}")
                        continue

                    # Compute zonal statistic
                    # Pass nodata value (or np.nan if None) to handle nodata correctly
                    result = compute_zonal_statistic(
                        feature_raster,
                        window_radius=window,
                        statistic=stat,
                        nodata=nodata_value if nodata_value is not None else np.nan,
                    )

                    # Save output
                    profile.update(dtype=rio.float32, nodata=-9999.0)
                    result_to_save = np.where(
                        np.isnan(result), profile["nodata"], result
                    ).astype(rio.float32)

                    with rio.open(out_path, "w", **profile) as dst:
                        dst.write(result_to_save, 1)

                    pbar.update(1)
                    pbar.set_postfix_str(f"Saved {out_path.name}")

    pbar.close()
    print("\nZonal statistics computation complete!")


# Map attribute names to computation functions
ATTRIBUTE_FUNCTIONS: Dict[str, Callable] = {
    "elevation": compute_elevation,
    "slope": compute_slope,
    "slope_break_index": compute_slope_break_index,
    "profile_curvature": compute_profile_curvature,
    "planform_curvature": compute_planform_curvature,
    "mean_curvature": compute_mean_curvature,
    "max_curvature": compute_max_curvature,
    "min_curvature": compute_min_curvature,
}


def extract_terrain_features(
    dem_path: Path,
    features_dir: Path,
    terrain_attrs: List[Dict],
) -> None:
    """
    Extract terrain features at multiple scales from a DEM.

    Args:
        dem_path: Path to input DEM raster
        features_dir: Directory to save output feature rasters
        terrain_attrs: List of terrain attributes to compute. Each item must be a dict with:
            - 'name': attribute name (string)
            - 'sigmas': list of sigmas to compute (list of floats)
    """
    features_dir.mkdir(parents=True, exist_ok=True)

    if not dem_path.exists():
        print(f"Warning: DEM not found at {dem_path}")
        return

    print(f"\n=== Terrain Feature Extraction Plan ===")
    print(f"DEM: {dem_path}")
    print(f"Output directory: {features_dir}\n")

    # Parse and validate config
    feature_configs = {}
    total_computations = 0

    for attr_config in terrain_attrs:
        attr_name = getattr(attr_config, "name", None)
        if not attr_name:
            print(f"Warning: Skipping invalid attribute config (missing 'name')")
            continue

        sigmas = getattr(attr_config, "sigmas", None)
        if sigmas is None:
            print(f"Warning: Skipping '{attr_name}' (missing 'sigmas')")
            continue

        sigmas = (
            list(sigmas)
            if hasattr(sigmas, "__iter__") and not isinstance(sigmas, str)
            else []
        )
        if not sigmas:
            print(f"Warning: Skipping '{attr_name}' (empty sigmas list)")
            continue

        if attr_name not in ATTRIBUTE_FUNCTIONS:
            print(f"Warning: Unknown attribute '{attr_name}', skipping...")
            continue

        feature_configs[attr_name] = sigmas
        total_computations += len(sigmas)

    if not feature_configs:
        print("No valid terrain features to compute.")
        return

    # Show what would be computed
    print("=== Features that would be computed ===\n")
    for attr_name, sigmas in feature_configs.items():
        print(f"{attr_name}:")
        for sigma in sigmas:
            out_path = features_dir / f"{attr_name}_sigma{sigma}.tif"
            exists = out_path.exists()
            status = "✓ exists" if exists else "→ would compute"
            print(f"  sigma={sigma}: {out_path.name} {status}")

    print(f"\n=== Summary ===")
    print(f"Total features to compute: {total_computations}")
    print(f"Unique attributes: {len(feature_configs)}")

    # COMMENTED OUT: Actual computation
    # Uncomment below to enable computation

    print(f"\nLoading DEM from {dem_path}...")
    with rio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)
        valid_mask = ~np.isnan(dem)

    pbar = tqdm(
        total=total_computations, desc="Computing terrain features", unit="feature"
    )

    for attr_name, sigmas in feature_configs.items():
        compute_func = ATTRIBUTE_FUNCTIONS[attr_name]
        sig = inspect.signature(compute_func)
        needs_pixel_size = len(sig.parameters) == 4

        for sigma in sigmas:
            out_path = features_dir / f"{attr_name}_sigma{sigma}.tif"
            if out_path.exists():
                pbar.update(1)
                continue

            if needs_pixel_size:
                result = compute_func(dem, pixel_size_x, pixel_size_y, sigma)
            else:
                result = compute_func(dem, sigma)

            result = np.where(valid_mask, result, np.nan)
            profile.update(dtype=rio.float32, nodata=-9999.0)
            result_to_save = np.where(
                np.isnan(result), profile["nodata"], result
            ).astype(rio.float32)

            with rio.open(out_path, "w", **profile) as dst:
                dst.write(result_to_save, 1)

            pbar.update(1)

    pbar.close()
    print("\nTerrain feature extraction complete!")
