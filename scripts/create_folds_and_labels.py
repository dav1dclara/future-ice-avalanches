"""
Create rectangular patches over the bedrock elevation raster.

For now this script simply divides the raster into non-overlapping patches
of a given pixel size and writes them to a GeoPackage.
"""

import geopandas as gpd
import hydra
import numpy as np
import rasterio as rio
from omegaconf import DictConfig
from rasterio.features import rasterize

from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Construct input paths from datasets config
    bedrock_path = PROJECT_ROOT / cfg.datasets.bedrock_elevation_file
    sgi1850_path = PROJECT_ROOT / cfg.datasets.sgi1850_outlines_file
    release_zones_path = PROJECT_ROOT / cfg.datasets.release_zones_file
    folds_output_path = PROJECT_ROOT / cfg.datasets.folds_file
    labels_output_path = PROJECT_ROOT / cfg.datasets.labels_file

    # Read files
    print("Reading files...")

    # Read bedrock raster
    with rio.open(bedrock_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        # nodata = src.nodata
        # bedrock = src.read(1, masked=True)
    print(f"  -> Bedrock: '{bedrock_path}'")

    sgi_gdf = gpd.read_file(sgi1850_path)
    print(f"  -> SGI1850 outlines: '{sgi1850_path}'")
    if sgi_gdf.crs != crs:
        sgi_gdf = sgi_gdf.to_crs(crs)
        print(f"     -> Reprojected SGI1850 outlines to '{crs}'")

    # sgi2016_gdf = gpd.read_file(sgi2016_path)
    # print(f"  -> SGI2016 outlines: '{sgi2016_path}'")
    # if sgi2016_gdf.crs != crs:
    #     sgi2016_gdf = sgi2016_gdf.to_crs(crs)
    #     print(f"    -> Reprojected SGI2016 outlines to '{crs}'")

    release_gdf = gpd.read_file(release_zones_path)
    print(f"  -> Release zones: '{release_zones_path}'")
    if release_gdf.crs != crs:
        release_gdf = release_gdf.to_crs(crs)
        print(f"    -> Reprojected release zones to '{crs}'")

    print("-" * 60)

    # # TODO: consider if using 1850 and 2016 outlines is necessary

    # Rasterize the SGI1850 outlines with fold values (-1 = doesn't touch, fold value = touches polygon)
    print("Rasterizing SGI1850 outlines with fold values...")

    # Filter out glaciers without fold values
    sgi_gdf_with_fold = sgi_gdf[sgi_gdf["fold"].notna()].copy()
    n_filtered = len(sgi_gdf) - len(sgi_gdf_with_fold)
    if n_filtered > 0:
        print(f"  -> Filtered out {n_filtered} glaciers without fold values")

    # Create shapes list: (geometry, fold_value)
    sgi1850_shapes = []
    for idx, glacier in sgi_gdf_with_fold.iterrows():
        fold_val = glacier["fold"]
        if fold_val is not None and not (
            isinstance(fold_val, float) and np.isnan(fold_val)
        ):
            sgi1850_shapes.append((glacier.geometry, int(fold_val)))

    sgi1850_raster = rasterize(
        sgi1850_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,  # -1 = does not touch SGI1850 polygons
        default_value=0,  # This won't be used since we're using fill=-1
        dtype=np.int16,
        all_touched=True,  # Include all pixels that touch the polygon
    )

    n_touches = np.sum(sgi1850_raster != -1)
    n_no_touch = np.sum(sgi1850_raster == -1)
    unique_folds, counts = np.unique(
        sgi1850_raster[sgi1850_raster != -1], return_counts=True
    )
    print(f"  -> Pixels touching SGI1850: {n_touches:,}")
    print(f"  -> Pixels not touching SGI1850: {n_no_touch:,}")
    if len(unique_folds) > 0:
        print("  -> Pixels per fold:")
        for fold_val, count in zip(unique_folds, counts):
            print(f"      Fold {fold_val}: {count:,} pixels")

    # Create output directory if needed
    folds_output_path.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(
        folds_output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=sgi1850_raster.dtype,
        crs=crs,
        transform=transform,
        nodata=-1,  # Use -1 for nodata since pixels that don't touch polygons are -1
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(sgi1850_raster, 1)
        dst.set_band_description(1, "fold")

    print(f"  -> Saved SGI1850 fold raster to '{folds_output_path}'")
    print("-" * 60)

    # Rasterize release zones and combine with fold layer
    print("Rasterizing release zones...")

    release_shapes = [(geom, 1) for geom in release_gdf.geometry]

    release_raster = rasterize(
        release_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # 0 = does not intersect release zones
        default_value=1,  # 1 = intersects release zones
        dtype=np.int16,
        all_touched=True,  # Include all pixels that touch the polygon
    )

    # Combine with fold layer:
    # -1 where fold is -1 (nodata in fold layer)
    # 1 where fold is not -1 AND release zone touches
    # 0 where fold is not -1 AND release zone doesn't touch
    release_combined = np.full_like(release_raster, -1, dtype=np.int16)
    fold_valid_mask = sgi1850_raster != -1  # Pixels that are valid in fold layer
    release_combined[fold_valid_mask] = release_raster[fold_valid_mask]

    n_release_pixels = np.sum(release_combined == 1)
    n_non_release_pixels = np.sum(release_combined == 0)
    n_nodata_pixels = np.sum(release_combined == -1)
    print(f"  -> Pixels touching release zones (valid in fold): {n_release_pixels:,}")
    print(
        f"  -> Pixels not touching release zones (valid in fold): {n_non_release_pixels:,}"
    )
    print(f"  -> Total pixels: {n_release_pixels + n_non_release_pixels:,}")

    # Create output directory if needed
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(
        labels_output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=release_combined.dtype,
        crs=crs,
        transform=transform,
        nodata=-1,  # Use -1 for nodata (consistent with fold raster)
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(release_combined, 1)
        dst.set_band_description(1, "release_zones")

    print(f"  -> Saved release zones raster to '{labels_output_path}'")
    print("-" * 60)


if __name__ == "__main__":
    main()
