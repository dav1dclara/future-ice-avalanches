"""Script to filter glacier hazard events."""

import hydra
import pandas as pd
from omegaconf import DictConfig

from future_ice_avalanches.paths import PROJECT_ROOT


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="config"
)
def main(cfg: DictConfig):
    """Main entry point."""
    # Paths from config
    events_file = PROJECT_ROOT / cfg.datasets.events.events_file
    output_file = PROJECT_ROOT / cfg.datasets.events.events_filtered_file

    # Load data
    print("Loading events data...")
    df = pd.read_excel(events_file)
    print(f"  -> Loaded {len(df)} total events")

    # Filter by hazard type (glacier/ice avalanche only)
    print("Filtering by hazard type...")
    hazard_filter = df["hazard_main_type"].str.startswith(
        "Glacier- or ice avalanche", na=False
    )
    df_filtered = df[hazard_filter].copy()
    print(f"  -> After hazard filter: {len(df_filtered)} events")

    # Print statistics
    print("Statistics:")
    print(f"  -> Unique events: {df_filtered['pk'].nunique()}")
    print(f"  -> Unique glaciers: {df_filtered['sgi_id'].nunique()}")

    # Save filtered dataframe
    print("Saving filtered events...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_file, index=False)
    print(f"  -> Saved to {output_file}")


if __name__ == "__main__":
    main()
