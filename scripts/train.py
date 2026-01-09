from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import rasterio as rio
from omegaconf import DictConfig, OmegaConf

import wandb
from future_ice_avalanches.models.data import (
    create_fold_samples,
    extract_sample_features_lazy,
    filter_features_by_set,
)
from future_ice_avalanches.models.models import get_model
from future_ice_avalanches.models.training import (
    train_and_save_final_model,
    train_cross_validation,
)
from future_ice_avalanches.paths import PROJECT_ROOT
from future_ice_avalanches.utils import seed_all


def train(cfg: DictConfig):
    seed_all()
    print("-" * 80)

    features_dir = Path(cfg.paths.features)

    print("LOADING FOLDS & LABELS...")
    folds_path = Path(cfg.paths.folds)
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds file not found: {folds_path}")

    with rio.open(folds_path) as src:
        folds_img = src.read(1, masked=True)
        folds_nodata = src.nodata
        # Convert to float and use NaN for nodata
        folds_img = folds_img.astype(np.float32)
        folds_img = np.where(folds_img.mask, np.nan, folds_img.data)
        folds_dtype = folds_img.dtype
    print(f"-> Loaded folds from '{folds_path}'")

    # Print unique fold values and counts
    unique_folds, unique_counts = np.unique(
        folds_img[~np.isnan(folds_img)], return_counts=True
    )
    total_valid = np.sum(unique_counts)
    print("    - Fold distribution:")
    for fold_val, count in zip(unique_folds, unique_counts):
        percentage = (count / total_valid) * 100
        print(f"       {int(fold_val)}:\t{count:,} ({percentage:.1f}%)")
    print(f"       Total:\t{total_valid:,} (100.0%)")
    print()

    labels_path = Path(cfg.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with rio.open(labels_path) as src:
        labels_img = src.read(1, masked=True)
        labels_nodata = src.nodata
        labels_profile = src.profile.copy()
        # Convert to float and use NaN for nodata
        labels_img = labels_img.astype(np.float32)
        labels_img = np.where(labels_img.mask, np.nan, labels_img.data)
        labels_dtype = labels_img.dtype
    print(f"-> Loaded labels from '{labels_path}'")

    # Print unique label values and counts
    unique_labels, unique_counts = np.unique(
        labels_img[~np.isnan(labels_img)], return_counts=True
    )
    total_valid = np.sum(unique_counts)
    print("    - Label distribution:")
    for label_val, count in zip(unique_labels, unique_counts):
        percentage = (count / total_valid) * 100
        print(f"       {int(label_val)}:\t{count:,} ({percentage:.1f}%)")
    print(f"       Total:\t{total_valid:,} (100.0%)")
    print()

    print("-" * 80)

    print("EXTRACTING SAMPLES...")
    samples, sample_counts = create_fold_samples(
        folds_img=folds_img,
        labels_img=labels_img,
        n_folds=cfg.sampling.folds,
        buffer_pixels=cfg.sampling.buffer_pixels,
        unlabeled_ratio=cfg.sampling.unlabeled_ratio,
        seed=cfg.sampling.seed,
        n_jobs=-1,
        verbose=True,
        # save_outputs=True,
        # output_dir=PROJECT_ROOT / "data" / "processed" / "samples",
        # labels_profile=labels_profile,
    )
    print("-" * 80)

    print("EXTRACTING SAMPLE FEATURES (LAZY LOADING)...")
    # Filter features based on feature_set if provided
    features_to_use = cfg.features
    feature_set = getattr(cfg, "feature_set", None)
    if feature_set is not None:
        features_to_use = filter_features_by_set(cfg.features, int(feature_set))
        print(f"- Using feature_set {feature_set}: {len(features_to_use)} features")
    else:
        print(f"- Using all features from config: {len(features_to_use)} features")

    # Build feature names
    feature_names = []
    for feat_cfg in features_to_use:
        feat_name = feat_cfg.name
        feat_sigma = feat_cfg.sigma

        stat = getattr(feat_cfg, "stat", None)
        window = getattr(feat_cfg, "window", None)

        if stat is not None and window is not None:
            feat_display_name = f"{feat_name}_sigma{feat_sigma}_{stat}_w{window}"
        else:
            feat_display_name = f"{feat_name}_sigma{feat_sigma}"

        feature_names.append(feat_display_name)

    # Lazy feature loading: load one feature at a time and extract values for sampled pixels
    X_samples, y_samples, fold_samples = extract_sample_features_lazy(
        features_dir=features_dir,
        features=features_to_use,
        samples=samples,
        feature_names=feature_names,
        unlabeled_ratio=cfg.sampling.unlabeled_ratio,
        seed=cfg.sampling.seed,
        verbose=True,
    )
    print("-> Extracted sample features")
    print("    - Features shape: ", X_samples.shape)
    print("    - Labels shape: ", y_samples.shape)
    print("    - Fold samples shape: ", fold_samples.shape)
    print("-" * 80)

    # Create feature set description for logging
    feature_set = getattr(cfg, "feature_set", None)
    feature_set_desc = None
    if feature_set is not None:
        feature_set_descriptions = {
            1: "No elevation, No zonal, Sigma 1 only",
            2: "No elevation, No zonal, Sigma 1 and 3",
            3: "No elevation, No zonal, All sigmas (1,3,5)",
            4: "No elevation, With zonal, Sigma 1 only",
            5: "No elevation, With zonal, Sigma 1 and 3",
            6: "No elevation, With zonal, All sigmas",
            7: "With elevation, No zonal, Sigma 1 only",
            8: "With elevation, No zonal, Sigma 1 and 3",
            9: "With elevation, No zonal, All sigmas",
            10: "With elevation, With zonal, Sigma 1 only",
            11: "With elevation, With zonal, Sigma 1 and 3",
            12: "With elevation, With zonal, All sigmas",
            13: "slope (sigma 1)",
            14: "slope, slope_break_index (sigma 1)",
            15: "profile_curvature, planform_curvature, mean_curvature (sigma 1)",
            16: "slope, profile_curvature, planform_curvature, mean_curvature (sigma 1)",
            17: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma 1)",
            18: "elevation (sigma 0), slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma 1)",
            19: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [1])",
            20: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [3])",
            21: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [5])",
            22: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [1,3])",
            23: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [3,5])",
            24: "slope, slope_break_index, profile_curvature, planform_curvature, mean_curvature (sigma [1,3,5])",
            25: "Set 24 + zonal std, window [3]",
            26: "Set 24 + zonal range, window [3]",
            27: "Set 24 + zonal std and range, window [3]",
            28: "Set 24 + zonal std, window [3,5,7]",
            29: "Set 24 + zonal range, window [3,5,7]",
            30: "Set 24 + zonal std and range, window [3,5,7]",
        }
        feature_set_desc = feature_set_descriptions.get(
            int(feature_set), f"Unknown feature_set {feature_set}"
        )

    print("GETTING MODEL...")
    model = get_model(cfg)
    print("-" * 80)

    # W&B - initialize first to get generated run ID, then set meaningful name
    wandb_run = None
    run_name = None
    model_type = cfg.model.type
    base_estimator_type = getattr(cfg.model.base_estimator, "type")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.wandb.enabled:
        # Prepare config for wandb, adding feature information
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        if feature_set_desc is not None:
            wandb_config["feature_set_description"] = feature_set_desc
            wandb_config["n_features"] = len(feature_names)
            wandb_config["feature_names"] = feature_names  # Log actual feature names

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_config,
        )
        # Create run name: timestamp + model_type + base_estimator + part of wandb ID (first 8 chars)
        id_part = wandb_run.id[:8]
        run_name = f"{timestamp}_{model_type}_{base_estimator_type}_{id_part}"
        # Update wandb run name
        wandb_run.name = run_name
        print(f"W&B logging enabled. Run name: {run_name}")
    else:
        print("W&B logging disabled.")
        # Generate a default name if wandb is disabled
        run_name = f"{timestamp}_{model_type}_{base_estimator_type}"
    print("-" * 80)

    print("TRAINING MODEL (CROSS-VALIDATION)...")

    # Always save model in a subdirectory with the run name
    models_dir = Path(cfg.paths.models) / run_name

    cv_metrics = train_cross_validation(
        X_samples=X_samples,
        y_samples=y_samples,
        fold_samples=fold_samples,
        model=model,
        n_folds=cfg.sampling.folds,
        feature_names=feature_names,
        output_dir=models_dir,
        verbose=True,
        wandb_run=wandb_run,
    )
    print("-" * 80)

    print("TRAINING AND SAVING FINAL MODEL...")
    print(f"  -> Run name: {run_name}")

    train_and_save_final_model(
        X_samples=X_samples,
        y_samples=y_samples,
        model=model,
        cfg=cfg,
        output_dir=models_dir,
        feature_names=feature_names,
        verbose=True,
    )
    print("-" * 80)

    # close wandb run
    if cfg.wandb.enabled:
        wandb_run.finish()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
