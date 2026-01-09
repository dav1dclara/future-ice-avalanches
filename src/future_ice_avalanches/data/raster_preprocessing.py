"""Raster preprocessing functions for creating base layers and terrain analysis."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, uniform_filter
from shapely.geometry import mapping
from shapely.ops import unary_union


def create_base_layer(
    swissalti3d_path: Path,
    ice_thickness_path: Path,
    sgi1850_path: Path,
    out_path: Path,
    buffer_distance: float = 500.0,
):
    """
    Create bedrock elevation base layer from DEM and ice thickness data.

    This function computes bedrock elevation by subtracting ice thickness from surface
    elevation (DEM). The process involves:
    1. Loading SwissALTI3D DEM and ice thickness rasters
    2. Reprojecting ice thickness to match DEM grid
    3. Computing bedrock = DEM - Ice Thickness
    4. Clipping to SGI 1850 glacier extents (with buffer)
    5. Saving the resulting bedrock elevation raster

    Args:
        swissalti3d_path: Path to SwissALTI3D DEM raster
        ice_thickness_path: Path to ice thickness raster
        sgi1850_path: Path to SGI 1850 glacier outlines shapefile
        out_path: Path to output bedrock elevation raster
        buffer_distance: Buffer distance in meters for glacier outlines
    """
    print("=" * 60)
    print("Processing Base Layer (Bedrock Elevation)")
    print("=" * 60)
    print(f"Input DEM: {swissalti3d_path}")
    print(f"Input Ice Thickness: {ice_thickness_path}")
    print(f"Input SGI 1850: {sgi1850_path}")
    print(f"Output Bedrock: {out_path}")
    print(f"Buffer Distance: {buffer_distance}m")
    print()

    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load DEM ---
    # Load the surface elevation DEM (SwissALTI3D) which represents the top of the ice/glacier
    print("Loading DEM...")
    file_size_bytes = swissalti3d_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb >= 1024:
        print(f"  -> File size: {file_size_mb / 1024:.2f} GB")
    else:
        print(f"  -> File size: {file_size_mb:.2f} MB")
    with rio.open(swissalti3d_path) as dem_ds:
        dem = dem_ds.read(1, masked=False)
        dem_nodata = dem_ds.nodata
        dem_crs = dem_ds.crs
        dem_transform = dem_ds.transform
        dem_shape = (dem_ds.height, dem_ds.width)
        profile = dem_ds.profile
        print(f"  -> DEM shape: {dem_shape[0]} x {dem_shape[1]} pixels")
        print(f"  -> DEM CRS: {dem_crs}")
        print(f"  -> DEM transform: {dem_transform}")
        print(f"  -> DEM dtype: {dem.dtype}")
        print(f"  -> DEM nodata: {dem_nodata}")

    # Build DEM mask: identify invalid/nodata pixels that should be excluded
    dem_mask = (dem == dem_nodata) if dem_nodata is not None else ~np.isfinite(dem)
    valid_pixels = np.sum(~dem_mask)
    print(f"  -> Valid pixels: {valid_pixels:,} ({100 * valid_pixels / dem.size:.1f}%)")
    print()

    # --- Load & reproject Ice Thickness to DEM grid ---
    # Ice thickness raster may have different CRS/resolution than DEM
    # Reproject to match DEM grid for accurate subtraction
    print("Loading Ice Thickness...")
    file_size_bytes = ice_thickness_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb >= 1024:
        print(f"  -> File size: {file_size_mb / 1024:.2f} GB")
    else:
        print(f"  -> File size: {file_size_mb:.2f} MB")
    with rio.open(ice_thickness_path) as ice_ds:
        ice = ice_ds.read(1, masked=False)
        ice_nodata = ice_ds.nodata
        ice_crs = ice_ds.crs
        ice_transform = ice_ds.transform
        ice_shape = (ice_ds.height, ice_ds.width)
        print(f"  -> Ice Thickness shape: {ice_shape[0]} x {ice_shape[1]} pixels")
        print(f"  -> Ice Thickness CRS: {ice_ds.crs}")
        print(f"  -> Ice Thickness transform: {ice_ds.transform}")
        print(f"  -> Ice Thickness dtype: {ice.dtype}")
        print(f"  -> Ice Thickness nodata: {ice_nodata}")

        # Build ice mask to count valid pixels
        ice_mask = (ice == ice_nodata) if ice_nodata is not None else ~np.isfinite(ice)
        ice_valid_pixels = np.sum(~ice_mask)
        print(
            f"  -> Valid pixels: {ice_valid_pixels:,} ({100 * ice_valid_pixels / ice.size:.1f}%)"
        )
        print()

    # Reproject ice thickness to DEM grid
    print("Reprojecting Ice Thickness to DEM grid...")
    print(
        f"  -> Source grid: {ice_shape[0]} x {ice_shape[1]} = {ice_shape[0] * ice_shape[1]:,} pixels"
    )
    print(
        f"  -> Destination grid (DEM): {dem_shape[0]} x {dem_shape[1]} = {dem_shape[0] * dem_shape[1]:,} pixels"
    )

    # Initialize destination with nodata (safe)
    dst_nodata = ice_nodata if ice_nodata is not None else np.nan
    ice_aligned = np.full(dem_shape, dst_nodata, dtype=np.float32)

    # Use nearest neighbor resampling to preserve ice thickness values
    # (avoid interpolation that could smooth or alter thickness measurements)
    reproject(
        source=ice,
        destination=ice_aligned,
        src_transform=ice_transform,
        src_crs=ice_crs,
        src_nodata=ice_nodata,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,  # avoids smoothing thickness
    )
    print("  -> Reprojection complete")

    # Identify pixels where ice thickness data is missing after reprojection
    # These areas will use DEM value directly (no ice to subtract)
    ice_is_nodata = (
        (~np.isfinite(ice_aligned))
        if np.isnan(dst_nodata)
        else (ice_aligned == dst_nodata)
    )
    ice_pixels = np.sum(~ice_is_nodata)
    print(
        f"  -> Ice pixels after reprojection: {ice_pixels:,} ({100 * ice_pixels / ice_aligned.size:.1f}% of DEM grid)"
    )
    if ice_pixels != ice_valid_pixels:
        diff = ice_pixels - ice_valid_pixels
        print(
            f"  -> Note: {diff:+,} pixels difference from source (due to resampling/grid alignment)"
        )
    print()

    # --- Compute bedrock elevation ---
    # Bedrock = Surface elevation (DEM) - Ice thickness
    # Where ice data is missing, use DEM value (assumes no ice or ice thickness unknown)
    # Where DEM is invalid, set to NaN
    print("Computing bedrock elevation (DEM - Ice Thickness)...")
    out_arr = np.where(ice_is_nodata, dem, dem - ice_aligned)
    out_arr = np.where(dem_mask, np.nan, out_arr).astype(np.float32)
    bedrock_pixels = np.sum(np.isfinite(out_arr))
    print(
        f"  -> Bedrock pixels: {bedrock_pixels:,} ({100 * bedrock_pixels / out_arr.size:.1f}%)"
    )
    if bedrock_pixels > 0:
        print(
            f"  -> Bedrock elevation range: {np.nanmin(out_arr):.1f}m to {np.nanmax(out_arr):.1f}m"
        )
    print()

    # --- Buffer shapefile and clip ---
    # Load SGI 1850 glacier outlines and buffer them to include areas around glaciers
    # This ensures we capture bedrock in areas that may have had glaciers historically
    print("Loading SGI 1850 outlines...")
    gdf = gpd.read_file(sgi1850_path)
    print(f"  -> Loaded {len(gdf)} glacier outlines")
    print()
    if gdf.crs != dem_crs:
        print(f"Reprojecting to {dem_crs}")
        gdf = gdf.to_crs(dem_crs)
        print("  -> Reprojection complete")
        print()

    # Buffer each glacier outline and union into single geometry
    # Use envelope to create square/rectangular buffers instead of round
    print(f"Buffering glacier outlines by {buffer_distance}m (square buffers)...")
    buffered = gdf.buffer(buffer_distance)  # meters (assumes raster CRS in meters)
    # Get bounding box (envelope) of each buffered geometry to make them square/rectangular
    buffered_square = buffered.envelope
    geom_union = unary_union(buffered_square)
    print("  -> Buffering and union complete")
    print()

    # Clip bedrock raster to buffered glacier extents
    print("Clipping bedrock to buffered glacier extents...")
    shapes = [mapping(geom_union)]

    # Use rasterio.mask to clip and mask in one step
    with MemoryFile() as memfile:
        tmp_profile = profile.copy()
        tmp_profile.update(dtype=rio.float32, nodata=np.nan, count=1)
        with memfile.open(**tmp_profile) as tmp_dst:
            tmp_dst.write(out_arr, 1)
            data_clipped, transform_clipped = mask(
                tmp_dst, shapes=shapes, crop=True, filled=True, nodata=np.nan
            )

    data_clipped = np.squeeze(data_clipped).astype(np.float32)
    aligned_height, aligned_width = data_clipped.shape
    clipped_pixels = np.sum(np.isfinite(data_clipped))
    print(f"  -> Clipped shape: {aligned_height} x {aligned_width} pixels")
    print(f"  -> Valid pixels after clipping: {clipped_pixels:,}")
    print()

    # --- Convert to uint16 for file size reduction ---
    # uint16 range: 0-65535 covers elevations from 0m to 65535m (1m precision)
    print("Converting to uint16 (1m precision)...")
    nodata_value = 65535  # Use max uint16 value for nodata (allows 0m elevation)

    data_squeezed = np.squeeze(data_clipped)

    # Convert: round to integer, convert to uint16
    # Set nodata where original data was NaN
    data_rounded = np.round(data_squeezed).astype(np.float64)
    data_uint16 = np.where(
        np.isnan(data_squeezed),
        nodata_value,
        np.clip(
            data_rounded, 0, nodata_value - 1
        ),  # Clip to uint16 range, excluding nodata value
    ).astype(np.uint16)

    # Verify no data loss (check if any values were clipped)
    max_elevation = np.nanmax(data_rounded) if np.any(np.isfinite(data_squeezed)) else 0
    max_allowed = nodata_value - 1
    if max_elevation > max_allowed:
        print(
            f"  -> Warning: Max elevation ({max_elevation:.0f}m) exceeds uint16 range (0-{max_allowed}m), some values clipped"
        )
    else:
        print(
            f"  -> Max elevation: {max_elevation:.0f}m (within uint16 range 0-{max_allowed})"
        )

    valid_uint16_pixels = np.sum(data_uint16 != nodata_value)
    print(f"  -> Valid pixels after conversion: {valid_uint16_pixels:,}")

    # Check actual max in uint16 array (excluding nodata)
    valid_data = data_uint16[data_uint16 != nodata_value]
    if len(valid_data) > 0:
        max_uint16 = np.max(valid_data)
        print(f"  -> Max elevation in uint16 array: {max_uint16}m")
    print()

    # --- Save final raster (cropped, uint16, compressed) ---
    print("Saving bedrock elevation raster...")
    out_profile = {
        "driver": "GTiff",
        "crs": dem_crs,
        "height": aligned_height,
        "width": aligned_width,
        "transform": transform_clipped,
        "dtype": rio.uint16,
        "nodata": nodata_value,
        "count": 1,
        "compress": "deflate",
        "predictor": 2,  # Horizontal differencing predictor (works well with uint)
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "NUM_THREADS": "ALL_CPUS",
    }

    with rio.open(out_path, "w", **out_profile) as dst:
        dst.write(data_uint16, 1)

    print(f"  -> Saved: {out_path}")
    print("  -> Data type: uint16 (1m precision)")
    print(f"  -> Nodata value: {nodata_value}")

    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


def create_base_layer_with_smoothed_dem(
    swissalti3d_path: Path,
    ice_thickness_path: Path,
    sgi1850_path: Path,
    out_path: Path,
    buffer_distance: float = 500.0,
    dem_smoothing_sigma: float = None,
):
    """
    Create bedrock elevation base layer with smoothed DEM to match ice thickness smoothness.

    This function is identical to create_base_layer, except it smooths the DEM before
    subtracting ice thickness. This helps match the spatial resolution/smoothness of the
    ice thickness model, reducing artifacts at glacier margins.

    The smoothing sigma can be:
    - Auto-estimated from ice thickness resolution (if dem_smoothing_sigma=None)
    - Manually specified (if dem_smoothing_sigma is provided)

    Args:
        swissalti3d_path: Path to SwissALTI3D DEM raster
        ice_thickness_path: Path to ice thickness raster
        sgi1850_path: Path to SGI 1850 glacier outlines shapefile
        out_path: Path to output bedrock elevation raster (will be saved to temp directory)
        buffer_distance: Buffer distance in meters for glacier outlines
        dem_smoothing_sigma: Optional smoothing sigma in pixels. If None, auto-estimates
            from ice thickness resolution.
    """
    print("=" * 60)
    print("Processing Base Layer (Bedrock Elevation with Smoothed DEM)")
    print("=" * 60)
    print(f"Input DEM: {swissalti3d_path}")
    print(f"Input Ice Thickness: {ice_thickness_path}")
    print(f"Input SGI 1850: {sgi1850_path}")
    print(f"Output Bedrock: {out_path}")
    print(f"Buffer Distance: {buffer_distance}m")
    print()

    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load DEM ---
    # Load the surface elevation DEM (SwissALTI3D) which represents the top of the ice/glacier
    print("Loading DEM...")
    file_size_bytes = swissalti3d_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb >= 1024:
        print(f"  -> File size: {file_size_mb / 1024:.2f} GB")
    else:
        print(f"  -> File size: {file_size_mb:.2f} MB")
    with rio.open(swissalti3d_path) as dem_ds:
        dem = dem_ds.read(1, masked=False)
        dem_nodata = dem_ds.nodata
        dem_crs = dem_ds.crs
        dem_transform = dem_ds.transform
        dem_shape = (dem_ds.height, dem_ds.width)
        profile = dem_ds.profile
        dem_pixel_size_x = abs(dem_transform[0])  # pixel width in meters
        dem_pixel_size_y = abs(dem_transform[4])  # pixel height in meters
        print(f"  -> DEM shape: {dem_shape[0]} x {dem_shape[1]} pixels")
        print(f"  -> DEM CRS: {dem_crs}")
        print(f"  -> DEM transform: {dem_transform}")
        print(f"  -> DEM pixel size: {dem_pixel_size_x:.2f}m x {dem_pixel_size_y:.2f}m")
        print(f"  -> DEM dtype: {dem.dtype}")
        print(f"  -> DEM nodata: {dem_nodata}")

    # Build DEM mask: identify invalid/nodata pixels that should be excluded
    dem_mask = (dem == dem_nodata) if dem_nodata is not None else ~np.isfinite(dem)
    valid_pixels = np.sum(~dem_mask)
    print(f"  -> Valid pixels: {valid_pixels:,} ({100 * valid_pixels / dem.size:.1f}%)")
    print()

    # --- Load & reproject Ice Thickness to DEM grid ---
    # Ice thickness raster may have different CRS/resolution than DEM
    # Reproject to match DEM grid for accurate subtraction
    print("Loading Ice Thickness...")
    file_size_bytes = ice_thickness_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb >= 1024:
        print(f"  -> File size: {file_size_mb / 1024:.2f} GB")
    else:
        print(f"  -> File size: {file_size_mb:.2f} MB")
    with rio.open(ice_thickness_path) as ice_ds:
        ice = ice_ds.read(1, masked=False)
        ice_nodata = ice_ds.nodata
        ice_crs = ice_ds.crs
        ice_transform = ice_ds.transform
        ice_shape = (ice_ds.height, ice_ds.width)
        ice_pixel_size_x = abs(ice_transform[0])  # pixel width in meters
        ice_pixel_size_y = abs(ice_transform[4])  # pixel height in meters
        print(f"  -> Ice Thickness shape: {ice_shape[0]} x {ice_shape[1]} pixels")
        print(f"  -> Ice Thickness CRS: {ice_crs}")
        print(f"  -> Ice Thickness transform: {ice_transform}")
        print(
            f"  -> Ice Thickness pixel size: {ice_pixel_size_x:.2f}m x {ice_pixel_size_y:.2f}m"
        )
        print(f"  -> Ice Thickness dtype: {ice.dtype}")
        print(f"  -> Ice Thickness nodata: {ice_nodata}")

        # Build ice mask to count valid pixels
        ice_mask = (ice == ice_nodata) if ice_nodata is not None else ~np.isfinite(ice)
        ice_valid_pixels = np.sum(~ice_mask)
        print(
            f"  -> Valid pixels: {ice_valid_pixels:,} ({100 * ice_valid_pixels / ice.size:.1f}%)"
        )
        print()

    # --- Estimate DEM smoothing sigma to match ice thickness smoothness ---
    if dem_smoothing_sigma is None:
        # Auto-estimate by comparing local variance (roughness) of DEM vs ice thickness
        # Local variance = standard deviation in a window (measure of roughness)
        print("  -> Analyzing smoothness (local variance method)...")

        # Reproject ice thickness first to get it on DEM grid for comparison
        ice_aligned_temp = np.full(dem_shape, np.nan, dtype=np.float32)
        reproject(
            source=ice,
            destination=ice_aligned_temp,
            src_transform=ice_transform,
            src_crs=ice_crs,
            src_nodata=ice_nodata,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        # Compute local variance (roughness) using a small window (e.g., 5x5 pixels)
        # This measures how much variation exists locally
        window_size = 5
        dem_valid = ~dem_mask
        ice_valid = np.isfinite(ice_aligned_temp)
        overlap_valid = dem_valid & ice_valid

        if np.sum(overlap_valid) > 1000:
            # Compute local variance for DEM
            dem_float = dem.astype(np.float64)
            dem_float[dem_mask] = np.nan
            dem_mean = uniform_filter(
                dem_float, size=window_size, mode="constant", cval=np.nan
            )
            dem_mean_sq = uniform_filter(
                dem_float**2, size=window_size, mode="constant", cval=np.nan
            )
            dem_variance = dem_mean_sq - dem_mean**2
            dem_variance = np.maximum(dem_variance, 0)  # Handle floating point errors
            dem_std = np.sqrt(dem_variance)

            # Compute local variance for ice thickness
            ice_mean = uniform_filter(
                ice_aligned_temp.astype(np.float64),
                size=window_size,
                mode="constant",
                cval=np.nan,
            )
            ice_mean_sq = uniform_filter(
                ice_aligned_temp.astype(np.float64) ** 2,
                size=window_size,
                mode="constant",
                cval=np.nan,
            )
            ice_variance = ice_mean_sq - ice_mean**2
            ice_variance = np.maximum(ice_variance, 0)
            ice_std = np.sqrt(ice_variance)

            # Compare: if DEM is much rougher (higher std), need smoothing
            dem_std_mean = np.nanmean(dem_std[overlap_valid])
            ice_std_mean = np.nanmean(ice_std[overlap_valid])

            if ice_std_mean > 0 and dem_std_mean > 0:
                roughness_ratio = dem_std_mean / ice_std_mean
                # Conservative estimate: sigma ≈ sqrt(roughness_ratio) - 1
                # This is more conservative than gradient-based approach
                estimated_sigma = max(np.sqrt(roughness_ratio) - 1.0, 0.5)
                estimated_sigma = min(estimated_sigma, 5.0)  # Cap at 5 pixels
                print(f"  -> DEM local std dev: {dem_std_mean:.2f} m")
                print(f"  -> Ice thickness local std dev: {ice_std_mean:.2f} m")
                print(f"  -> Roughness ratio: {roughness_ratio:.2f}")
                print(f"  -> Estimated smoothing sigma: {estimated_sigma:.2f} pixels")
            else:
                estimated_sigma = 1.0
                print(
                    f"  -> Could not compute variance, using default: {estimated_sigma:.2f} pixels"
                )
        else:
            estimated_sigma = 1.0
            print(
                f"  -> Insufficient overlap, using default: {estimated_sigma:.2f} pixels"
            )

        dem_smoothing_sigma = estimated_sigma
    else:
        print(f"  -> Using manual smoothing sigma: {dem_smoothing_sigma:.2f} pixels")

    print()

    # --- Smooth DEM to match ice thickness smoothness ---
    print(f"Smoothing DEM (sigma={dem_smoothing_sigma:.2f} pixels)...")
    # Convert to float64 for smoothing
    dem_float = dem.astype(np.float64)

    # Smooth DEM: gaussian_filter doesn't handle NaN well, so we'll:
    # 1. Fill NaN with 0 temporarily for smoothing
    # 2. Create a mask of valid pixels
    # 3. Smooth the filled array
    # 4. Restore NaN where original was invalid
    dem_filled = np.where(dem_mask, 0.0, dem_float)
    dem_smoothed = gaussian_filter(dem_filled, sigma=dem_smoothing_sigma)

    # Restore NaN where original DEM was invalid
    dem_smoothed[dem_mask] = np.nan

    # Convert back to original dtype range (but keep as float for subtraction)
    dem_smoothed = dem_smoothed.astype(np.float64)

    print(f"  -> DEM smoothing complete")
    print()

    # Reproject ice thickness to DEM grid
    print("Reprojecting Ice Thickness to DEM grid...")
    print(
        f"  -> Source grid: {ice_shape[0]} x {ice_shape[1]} = {ice_shape[0] * ice_shape[1]:,} pixels"
    )
    print(
        f"  -> Destination grid (DEM): {dem_shape[0]} x {dem_shape[1]} = {dem_shape[0] * dem_shape[1]:,} pixels"
    )

    # Initialize destination with nodata (safe)
    dst_nodata = ice_nodata if ice_nodata is not None else np.nan
    ice_aligned = np.full(dem_shape, dst_nodata, dtype=np.float32)

    # Use nearest neighbor resampling to preserve ice thickness values
    # (avoid interpolation that could smooth or alter thickness measurements)
    reproject(
        source=ice,
        destination=ice_aligned,
        src_transform=ice_transform,
        src_crs=ice_crs,
        src_nodata=ice_nodata,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,  # avoids smoothing thickness
    )
    print("  -> Reprojection complete")

    # Identify pixels where ice thickness data is missing after reprojection
    # These areas will use DEM value directly (no ice to subtract)
    ice_is_nodata = (
        (~np.isfinite(ice_aligned))
        if np.isnan(dst_nodata)
        else (ice_aligned == dst_nodata)
    )
    ice_pixels = np.sum(~ice_is_nodata)
    print(
        f"  -> Ice pixels after reprojection: {ice_pixels:,} ({100 * ice_pixels / ice_aligned.size:.1f}% of DEM grid)"
    )
    if ice_pixels != ice_valid_pixels:
        diff = ice_pixels - ice_valid_pixels
        print(
            f"  -> Note: {diff:+,} pixels difference from source (due to resampling/grid alignment)"
        )
    print()

    # --- Compute bedrock elevation ---
    # Bedrock = Surface elevation (smoothed DEM) - Ice thickness
    # Where ice data is missing, use smoothed DEM value (assumes no ice or ice thickness unknown)
    # Where DEM is invalid, set to NaN
    print("Computing bedrock elevation (Smoothed DEM - Ice Thickness)...")
    out_arr = np.where(ice_is_nodata, dem_smoothed, dem_smoothed - ice_aligned)
    out_arr = np.where(dem_mask, np.nan, out_arr).astype(np.float32)
    bedrock_pixels = np.sum(np.isfinite(out_arr))
    print(
        f"  -> Bedrock pixels: {bedrock_pixels:,} ({100 * bedrock_pixels / out_arr.size:.1f}%)"
    )
    if bedrock_pixels > 0:
        print(
            f"  -> Bedrock elevation range: {np.nanmin(out_arr):.1f}m to {np.nanmax(out_arr):.1f}m"
        )
    print()

    # --- Buffer shapefile and clip ---
    # Load SGI 1850 glacier outlines and buffer them to include areas around glaciers
    # This ensures we capture bedrock in areas that may have had glaciers historically
    print("Loading SGI 1850 outlines...")
    gdf = gpd.read_file(sgi1850_path)
    print(f"  -> Loaded {len(gdf)} glacier outlines")
    print()
    if gdf.crs != dem_crs:
        print(f"Reprojecting to {dem_crs}")
        gdf = gdf.to_crs(dem_crs)
        print("  -> Reprojection complete")
        print()

    # Buffer each glacier outline and union into single geometry
    # Use envelope to create square/rectangular buffers instead of round
    print(f"Buffering glacier outlines by {buffer_distance}m (square buffers)...")
    buffered = gdf.buffer(buffer_distance)  # meters (assumes raster CRS in meters)
    # Get bounding box (envelope) of each buffered geometry to make them square/rectangular
    buffered_square = buffered.envelope
    geom_union = unary_union(buffered_square)
    print("  -> Buffering and union complete")
    print()

    # Clip bedrock raster to buffered glacier extents
    print("Clipping bedrock to buffered glacier extents...")
    shapes = [mapping(geom_union)]

    # Use rasterio.mask to clip and mask in one step
    with MemoryFile() as memfile:
        tmp_profile = profile.copy()
        tmp_profile.update(dtype=rio.float32, nodata=np.nan, count=1)
        with memfile.open(**tmp_profile) as tmp_dst:
            tmp_dst.write(out_arr, 1)
            data_clipped, transform_clipped = mask(
                tmp_dst, shapes=shapes, crop=True, filled=True, nodata=np.nan
            )

    data_clipped = np.squeeze(data_clipped).astype(np.float32)
    aligned_height, aligned_width = data_clipped.shape
    clipped_pixels = np.sum(np.isfinite(data_clipped))
    print(f"  -> Clipped shape: {aligned_height} x {aligned_width} pixels")
    print(f"  -> Valid pixels after clipping: {clipped_pixels:,}")
    print()

    # --- Convert to uint16 for file size reduction ---
    # uint16 range: 0-65535 covers elevations from 0m to 65535m (1m precision)
    print("Converting to uint16 (1m precision)...")
    nodata_value = 65535  # Use max uint16 value for nodata (allows 0m elevation)

    data_squeezed = np.squeeze(data_clipped)

    # Convert: round to integer, convert to uint16
    # Set nodata where original data was NaN
    data_rounded = np.round(data_squeezed).astype(np.float64)
    data_uint16 = np.where(
        np.isnan(data_squeezed),
        nodata_value,
        np.clip(
            data_rounded, 0, nodata_value - 1
        ),  # Clip to uint16 range, excluding nodata value
    ).astype(np.uint16)

    # Verify no data loss (check if any values were clipped)
    max_elevation = np.nanmax(data_rounded) if np.any(np.isfinite(data_squeezed)) else 0
    max_allowed = nodata_value - 1
    if max_elevation > max_allowed:
        print(
            f"  -> Warning: Max elevation ({max_elevation:.0f}m) exceeds uint16 range (0-{max_allowed}m), some values clipped"
        )
    else:
        print(
            f"  -> Max elevation: {max_elevation:.0f}m (within uint16 range 0-{max_allowed})"
        )

    valid_uint16_pixels = np.sum(data_uint16 != nodata_value)
    print(f"  -> Valid pixels after conversion: {valid_uint16_pixels:,}")

    # Check actual max in uint16 array (excluding nodata)
    valid_data = data_uint16[data_uint16 != nodata_value]
    if len(valid_data) > 0:
        max_uint16 = np.max(valid_data)
        print(f"  -> Max elevation in uint16 array: {max_uint16}m")
    print()

    # --- Save final raster (cropped, uint16, compressed) ---
    print("Saving bedrock elevation raster...")
    out_profile = {
        "driver": "GTiff",
        "crs": dem_crs,
        "height": aligned_height,
        "width": aligned_width,
        "transform": transform_clipped,
        "dtype": rio.uint16,
        "nodata": nodata_value,
        "count": 1,
        "compress": "deflate",
        "predictor": 2,  # Horizontal differencing predictor (works well with uint)
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "NUM_THREADS": "ALL_CPUS",
    }

    with rio.open(out_path, "w", **out_profile) as dst:
        dst.write(data_uint16, 1)

    print(f"  -> Saved: {out_path}")
    print("  -> Data type: uint16 (1m precision)")
    print(f"  -> Nodata value: {nodata_value}")
    print(f"  -> DEM smoothing sigma: {dem_smoothing_sigma:.2f} pixels")

    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


def create_label_raster(
    release_zones_path: Path,
    base_layer_path: Path,
    sgi1850_path: Path,
    out_path: Path,
    buffer_distance: float = 100.0,
) -> None:
    """
    Create a label raster from release zones with exclusion buffers.

    This function creates a multi-class label raster where:
    - 0: Invalid/no data (outside SGI1850 outlines or bedrock no data)
    - 1: Valid bedrock (unlabeled, available for training)
    - 2: Exclusion buffer around release zones (excluded from training to avoid autocorrelation)
    - 3: Release zones (positive labels)

    Args:
        release_zones_path: Path to release zones GeoPackage
        base_layer_path: Path to bedrock elevation raster (used as reference for CRS, transform, and shape)
        sgi1850_path: Path to SGI 1850 glacier outlines shapefile
        out_path: Path to output label raster
        buffer_distance: Buffer distance in meters around release zones to exclude from training
    """
    # Read reference DEM to get transform and crs
    with rio.open(base_layer_path) as bedrock_src:
        bedrock = bedrock_src.read(1, masked=True)
        bedrock_crs = bedrock_src.crs
        bedrock_shape = bedrock_src.shape
        bedrock_transform = bedrock_src.transform

    # Read release zones and convert to reference CRS
    release_zones = gpd.read_file(release_zones_path)
    if release_zones.crs != bedrock_crs:
        release_zones = release_zones.to_crs(bedrock_crs)
        print(f"Reprojected release zones to bedrock CRS: {bedrock_crs}")

    # Read SGI1850 outlines and convert to reference CRS
    sgi1850 = gpd.read_file(sgi1850_path)
    if sgi1850.crs != bedrock_crs:
        sgi1850 = sgi1850.to_crs(bedrock_crs)
        print(f"Reprojected SGI1850 outlines to bedrock CRS: {bedrock_crs}")

    # Step 1: Create raster with value 1 where base layer is valid (not no data)
    valid_raster = np.where(~bedrock.mask, 1, 0).astype("uint8")

    # Step 1b: Create mask from SGI1850 outlines (1 inside, 0 outside)
    sgi1850_mask = rasterize(
        [(geom, 1) for geom in sgi1850.geometry],
        out_shape=bedrock_shape,
        transform=bedrock_transform,
        fill=0,
        dtype="uint8",
    )

    # Apply SGI1850 mask: pixels outside outlines become 0 (invalid)
    valid_raster = valid_raster * sgi1850_mask

    # Step 2: Create buffered exclusion zones around release zones (value = 2)
    # This avoids autocorrelation by excluding nearby pixels from training
    release_zones_buffered = release_zones.copy()
    release_zones_buffered.geometry = release_zones_buffered.geometry.buffer(
        buffer_distance
    )
    buffer_raster = rasterize(
        [(geom, 2) for geom in release_zones_buffered.geometry],
        out_shape=bedrock_shape,
        transform=bedrock_transform,
        fill=0,
        dtype="uint8",
    )

    # Step 3: Rasterize release zones (value = 3), overwriting where they overlap
    zones_raster = rasterize(
        [(geom, 3) for geom in release_zones.geometry],
        out_shape=bedrock_shape,
        transform=bedrock_transform,
        fill=0,
        dtype="uint8",
    )

    # Step 4: Combine with priority: release zones (3) > exclusion buffer (2) > valid bedrock (1) > invalid (0)
    label_raster = np.maximum(np.maximum(valid_raster, buffer_raster), zones_raster)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        out_path,
        "w",
        driver="GTiff",
        height=label_raster.shape[0],
        width=label_raster.shape[1],
        count=1,
        dtype=label_raster.dtype,
        crs=bedrock_crs,
        transform=bedrock_transform,
    ) as dst:
        dst.write(label_raster, 1)
    print(f"Labels saved to '{out_path}'.")
