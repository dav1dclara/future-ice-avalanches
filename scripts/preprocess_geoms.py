"""Script to clean and deduplicate glacier hazard geometries."""

import hydra
from omegaconf import DictConfig

from future_ice_avalanches.data.events_preprocessing import preprocess_geoms
from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig):
    """Main entry point."""
    # Paths from config
    input_path = PROJECT_ROOT / cfg.datasets.events.geoms_file
    output_path = PROJECT_ROOT / cfg.datasets.events.geoms_processed_file

    preprocess_geoms(input_path, output_path)


if __name__ == "__main__":
    main()
