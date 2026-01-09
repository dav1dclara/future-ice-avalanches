"""
Script to predict ice avalanche risk on the full extent using a trained model.

This script:
1. Loads a trained model, scaler, and config from data/models/{run_name}
2. Loads features for the full extent using the saved config
3. Predicts risk scores for all valid pixels
4. Saves the risk map as a raster

Usage:
    python scripts/modelling/predict.py run_name=my_run_name
"""

from pathlib import Path

import geopandas as gpd
import hydra
import joblib
import numpy as np
import rasterio as rio
from omegaconf import DictConfig, OmegaConf
from rasterio.features import rasterize
from tqdm import tqdm

from future_ice_avalanches.models.data import filter_features_by_set
from future_ice_avalanches.paths import PROJECT_ROOT


def predict(cfg: DictConfig):
    """Predict ice avalanche risk on the full extent.

    Args:
        cfg: Configuration dict with 'run_name' key.
    """
    run_name = cfg.run_name
    print("-" * 80)
    print(f"LOADING MODEL, SCALER, AND CONFIG FOR RUN: {run_name}")

    # Get models directory from config
    default_config = PROJECT_ROOT / "configs" / "train.yaml"
    if default_config.exists():
        cfg_default = OmegaConf.load(default_config)
        base_models_dir = PROJECT_ROOT / cfg_default.paths.models
    else:
        base_models_dir = PROJECT_ROOT / "data" / "models"

    models_dir = base_models_dir / run_name

    model_path = models_dir / "model.joblib"
    scaler_path = models_dir / "scaler.joblib"
    config_path = models_dir / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load training config (for features and model info)
    cfg_train = OmegaConf.load(config_path)
    print(f"  -> Loaded config from '{config_path}'")

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"  -> Loaded model from '{model_path}'")
    print(f"  -> Loaded scaler from '{scaler_path}'")
    print("-" * 80)

    print("PREPARING FOR WINDOWED PREDICTION...")
    features_dir = PROJECT_ROOT / cfg_train.paths.features

    # Filter features based on feature_set if it was used during training
    features_to_use = cfg_train.features
    feature_set = getattr(cfg_train, "feature_set", None)
    if feature_set is not None:
        features_to_use = filter_features_by_set(cfg_train.features, int(feature_set))
        print(f"  -> Using feature_set {feature_set}: {len(features_to_use)} features")
    else:
        print(f"  -> Using all features from config: {len(features_to_use)} features")

    # Build feature file paths and names
    feature_files = []
    feature_names = []
    for feat_cfg in features_to_use:
        feat_name = feat_cfg.name
        feat_sigma = feat_cfg.sigma

        stat = getattr(feat_cfg, "stat", None)
        window = getattr(feat_cfg, "window", None)

        if stat is not None and window is not None:
            feat_file = (
                features_dir / f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}.tif"
            )
            feat_display_name = f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}"
        else:
            feat_file = features_dir / f"{feat_name}_sigma{feat_sigma}.tif"
            feat_display_name = f"{feat_name}_sigma{feat_sigma}"

        if not feat_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_file}")

        feature_files.append(feat_file)
        feature_names.append(feat_display_name)

    # Get raster dimensions and metadata from first feature file
    with rio.open(feature_files[0]) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        first_feature_path = feature_files[0]

    print(f"  -> Raster dimensions: {width} x {height} pixels")
    print(f"  -> Number of features: {len(feature_files)}")
    print("-" * 80)

    print("LOADING SGI1850 GLACIER EXTENT...")
    sgi1850_path = PROJECT_ROOT / "data" / "raw" / "sgi1850" / "SGI_1850.shp"
    if not sgi1850_path.exists():
        raise FileNotFoundError(f"SGI1850 shapefile not found: {sgi1850_path}")

    sgi1850_gdf = gpd.read_file(sgi1850_path)
    if sgi1850_gdf.crs != crs:
        sgi1850_gdf = sgi1850_gdf.to_crs(crs)
        print(f"  -> Reprojected SGI1850 to '{crs}'")

    # Rasterize SGI1850 outlines (1 = inside glaciers, 0 = outside)
    sgi1850_mask = rasterize(
        [(geom, 1) for geom in sgi1850_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True,
    ).astype(bool)

    n_glacier_pixels = np.sum(sgi1850_mask)
    print(f"  -> SGI1850 glacier pixels: {n_glacier_pixels:,}")
    print("-" * 80)

    print("SETTING UP OUTPUT RASTER...")
    # Get output profile from first feature file
    with rio.open(first_feature_path) as src:
        output_profile = src.profile.copy()
        output_profile.update(
            {
                "dtype": "float32",
                "count": 1,
                "nodata": -9999.0,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

    # Get output path
    output_dir_name = (
        getattr(cfg.paths, "output", "data/predictions")
        if hasattr(cfg, "paths")
        else "data/predictions"
    )
    output_dir = PROJECT_ROOT / output_dir_name
    output_path = output_dir / f"{run_name}.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> Output path: '{output_path}'")
    print("-" * 80)

    print("PROCESSING PREDICTIONS IN WINDOWS...")
    # Window size: process in chunks to manage memory
    # Use 2048x2048 windows (adjust based on available memory)
    window_size = getattr(cfg, "window_size", 2048)  # Default 2048x2048
    print(f"  -> Window size: {window_size}x{window_size} pixels")

    # Initialize output raster
    risk_map = np.full((height, width), -9999.0, dtype=np.float32)
    total_valid_pixels = 0
    total_predicted_pixels = 0

    # Process in windows
    n_windows_x = (width + window_size - 1) // window_size
    n_windows_y = (height + window_size - 1) // window_size
    total_windows = n_windows_x * n_windows_y

    with tqdm(total=total_windows, desc="Processing windows") as pbar:
        for win_y in range(n_windows_y):
            for win_x in range(n_windows_x):
                # Calculate window bounds
                row_start = win_y * window_size
                row_end = min(row_start + window_size, height)
                col_start = win_x * window_size
                col_end = min(col_start + window_size, width)

                win_height = row_end - row_start
                win_width = col_end - col_start

                # Create window object for rasterio
                window = rio.windows.Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=win_width,
                    height=win_height,
                )

                # Load features for this window (one at a time)
                window_features = []
                window_valid_mask = None

                for feat_idx, feat_file in enumerate(feature_files):
                    with rio.open(feat_file) as src:
                        arr = src.read(1, window=window, masked=True).astype(np.float32)
                        arr = np.where(arr.mask, np.nan, arr.data)

                    # Stack features: shape (win_height, win_width, n_features)
                    if window_valid_mask is None:
                        # First feature: initialize valid mask
                        window_valid_mask = np.isfinite(arr)
                    else:
                        # Subsequent features: combine with AND
                        window_valid_mask = window_valid_mask & np.isfinite(arr)

                    window_features.append(arr)

                # Stack features: shape (win_height, win_width, n_features)
                X_window = np.stack(window_features, axis=-1)

                # Get SGI1850 mask for this window
                sgi1850_window = sgi1850_mask[row_start:row_end, col_start:col_end]

                # Combine masks: valid features AND within SGI1850
                valid_window_mask = window_valid_mask & sgi1850_window
                n_valid_window = np.sum(valid_window_mask)

                if n_valid_window > 0:
                    # Extract valid pixels
                    X_valid_window = X_window[
                        valid_window_mask
                    ]  # Shape: (n_valid, n_features)

                    # Scale features
                    X_valid_scaled = scaler.transform(X_valid_window)

                    # Predict
                    risk_scores = model.predict_proba(X_valid_scaled)[:, 1]

                    # Store predictions in output array
                    risk_window = np.full(
                        (win_height, win_width), -9999.0, dtype=np.float32
                    )
                    risk_window[valid_window_mask] = risk_scores
                    risk_map[row_start:row_end, col_start:col_end] = risk_window

                    total_predicted_pixels += n_valid_window

                total_valid_pixels += n_valid_window
                pbar.update(1)
                pbar.set_postfix_str(f"Predicted: {total_predicted_pixels:,} pixels")

    print(f"  -> Total valid pixels: {total_valid_pixels:,}")
    print(f"  -> Predicted pixels: {total_predicted_pixels:,}")
    print(f"    - Min risk: {np.min(risk_map[risk_map != -9999.0]):.4f}")
    print(f"    - Max risk: {np.max(risk_map[risk_map != -9999.0]):.4f}")
    print(f"    - Mean risk: {np.mean(risk_map[risk_map != -9999.0]):.4f}")
    print("-" * 80)

    print("SAVING RISK MAP...")
    with rio.open(output_path, "w", **output_profile) as dst:
        dst.write(risk_map, 1)

    print(f"  -> Saved risk map to '{output_path}'")
    print("-" * 80)
    print("Prediction complete!")


@hydra.main(version_base=None, config_path="../../configs", config_name="predict")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
