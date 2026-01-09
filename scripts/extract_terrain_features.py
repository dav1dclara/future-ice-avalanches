"""
Script to extract terrain features at multiple sigmas.

This script computes terrain attributes (slope, aspect, curvature, etc.) at different
spatial sigmas (Gaussian filter standard deviations) using rasterio, numpy, and scipy.
Each attribute is computed at multiple sigmas as specified in the configuration.
"""

import hydra
from omegaconf import DictConfig

from future_ice_avalanches.data.terrain_features import (
    compute_zonal_statistics_for_features,
    extract_terrain_features,
)
from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig):
    bedrock_path = PROJECT_ROOT / cfg.datasets.bedrock_elevation_file
    features_dir = PROJECT_ROOT / cfg.datasets.terrain_features_dir
    terrain_attrs = cfg.terrain_attributes

    # Extract base terrain features
    extract_terrain_features(
        dem_path=bedrock_path,
        features_dir=features_dir,
        terrain_attrs=terrain_attrs,
    )

    # Compute zonal statistics if configured
    zonal_config = getattr(cfg, "zonal_stats", None)
    if zonal_config is not None:
        compute_zonal_statistics_for_features(
            features_dir=features_dir,
            zonal_config=zonal_config,
            terrain_attrs=terrain_attrs,
        )


if __name__ == "__main__":
    main()
