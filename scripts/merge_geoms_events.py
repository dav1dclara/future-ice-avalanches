"""Script to create events GeoDataFrame with combined geometries."""

import hydra
from omegaconf import DictConfig

from future_ice_avalanches.data.events_preprocessing import append_geoms_to_events
from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig):
    """Main entry point."""
    # Paths from config
    events_dir = PROJECT_ROOT / cfg.datasets.events.processed_dir
    events_path = PROJECT_ROOT / cfg.datasets.events.events_filtered_file
    geoms_path = PROJECT_ROOT / cfg.datasets.events.geoms_processed_file
    output_path = PROJECT_ROOT / cfg.datasets.events.events_geoms_file

    append_geoms_to_events(events_path, geoms_path, output_path)


if __name__ == "__main__":
    main()
