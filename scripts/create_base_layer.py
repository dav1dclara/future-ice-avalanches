"""Script to create bedrock elevation layer from DEM and ice thickness data."""

import hydra
from omegaconf import DictConfig

from future_ice_avalanches.data.raster_preprocessing import create_base_layer
from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig):
    """Main entry point."""
    # Construct input paths from datasets config
    swissalti3d_path = PROJECT_ROOT / cfg.datasets.sgt2020.swissalti3d_file
    ice_thickness_path = PROJECT_ROOT / cfg.datasets.sgt2020.ice_thickness_file
    sgi1850_path = PROJECT_ROOT / cfg.datasets.sgi1850_outlines_file
    out_path = PROJECT_ROOT / cfg.datasets.bedrock_elevation_file

    # Processing parameters
    buffer_distance = 500.0  # meters

    if out_path.exists():
        print(f"Output file already exists: {out_path}")

    else:
        create_base_layer(
            swissalti3d_path=swissalti3d_path,
            ice_thickness_path=ice_thickness_path,
            sgi1850_path=sgi1850_path,
            out_path=out_path,
            buffer_distance=buffer_distance,
        )


if __name__ == "__main__":
    main()
