"""Functions for processing and merging label geometries."""

import hashlib
import json
import uuid
from itertools import chain
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import MultiLineString


def compute_geometry_hash(geom):
    """Return a stable hash for a geometry (independent of coordinate order)."""
    if geom is None or geom.is_empty:
        return None
    geom = shapely.make_valid(geom)
    geom = shapely.normalize(geom)
    wkb = shapely.to_wkb(geom, hex=False)
    return hashlib.sha1(wkb).hexdigest()


def hash_to_uuid(geom_hash):
    """Convert geometry hash to UUID string."""
    if geom_hash is None:
        return None
    # Use first 32 chars of hash to create UUID
    return str(uuid.UUID(geom_hash[:32]))


def combine_events(row, event_cols):
    """
    Combine event columns into a single list, handling semicolon-separated values.

    Validates that all events from original columns are preserved in the combined result.
    """
    # Count events in original columns
    original_count = 0
    for val in row[event_cols]:
        if pd.notna(val) and str(val).strip() not in ["", "nan", "None"]:
            # Count semicolon-separated events
            original_count += len([e for e in str(val).split(";") if e.strip()])

    # Combine events
    events = []
    for val in row[event_cols]:
        if pd.isna(val) or val in ["", "nan", "None"]:
            continue
        # Split on semicolon and clean each event
        for e in str(val).split(";"):
            e = e.strip()
            if e:
                events.append(e)

    # Validate: ensure all events were combined
    combined_count = len(events)
    if original_count != combined_count:
        raise ValueError(
            f"Event count mismatch in row! Original columns: {original_count} events, "
            f"Combined column: {combined_count} events. Some events may have been lost."
        )

    return events


def preprocess_geoms(input_path: Path, output_path: Path) -> None:
    """
    Clean and deduplicate glacier hazard geometries.

    Args:
        input_path: Path to input shapefile
        output_path: Path to output GeoPackage
    """
    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: Loading data")
    print("=" * 80)
    gdf = gpd.read_file(input_path)
    print(f"Loaded {input_path.stem}")
    print(f"  -> Initial shape: {gdf.shape}")
    print(f"  -> CRS: {gdf.crs}")
    print("  -> Columns:")
    for col in gdf.columns:
        print(f"       - {col}")

    # Step 2: Combine event columns and keep only needed columns
    print("\n" + "=" * 80)
    print("STEP 2: Combining event columns and selecting columns")
    print("=" * 80)
    event_cols = ["event_1", "event_2", "event_3", "event_4", "event_5"]
    keep_cols = ["events", "geometry"]

    print(f"Combining {len(event_cols)} event columns...")
    gdf["events"] = gdf.apply(lambda row: combine_events(row, event_cols), axis=1)

    # Count unique events across all rows
    all_events = [event for events_list in gdf["events"] for event in events_list]
    unique_events = len(set(all_events))
    print(f"  -> Total unique events: {unique_events}")

    print(f"  -> Keeping only columns: {keep_cols}")
    gdf = gdf[keep_cols].copy()

    # Step 3: Compute geometry hashes and convert to UUIDs
    print("\n" + "=" * 80)
    print("STEP 3: Computing geometry hashes and converting to UUIDs")
    print("=" * 80)
    print("Computing geometry hashes...")
    gdf["geom_hash"] = gdf.geometry.apply(compute_geometry_hash)
    print("Converting hashes to UUIDs...")
    gdf["geom_uuid"] = gdf["geom_hash"].apply(hash_to_uuid)
    print(f"  -> Computed {len(gdf)} geometry hashes")

    # Step 4: Find duplicate geometries
    print("\n" + "=" * 80)
    print("STEP 4: Finding duplicate geometries")
    print("=" * 80)

    # Count duplicates
    duplicate_mask = gdf["geom_hash"].duplicated(keep=False)
    n_duplicates = duplicate_mask.sum()
    n_unique_geoms = gdf["geom_uuid"].nunique()

    print("Finding duplicate geometries...")
    print(f"  -> Total geometries: {len(gdf)}")
    print(f"  -> Unique geometries: {n_unique_geoms}")
    print(f"  -> Duplicates: {n_duplicates}")

    if n_duplicates > 0:
        duplicate_groups = gdf[duplicate_mask].groupby("geom_uuid")

        print("\nDuplicate groups:")
        print("-" * 80)

        for geom_uuid, group in duplicate_groups:
            if len(group) > 1:
                # Check if events are the same
                event_sets = [set(row["events"]) for _, row in group.iterrows()]
                events_are_same = (
                    len(set(tuple(sorted(events)) for events in event_sets)) == 1
                )

                print(f"\nGeometry UUID: {geom_uuid}")
                print(f"  Number of duplicates: {len(group)}")
                print(
                    f"  Events are identical: {'✓ YES' if events_are_same else '✗ NO'}"
                )
                print("  Events before merge:")
                for idx, row in group.iterrows():
                    events_str = ", ".join(row["events"]) if row["events"] else "(none)"
                    print(f"    - Row {idx}: {events_str}")

        print("\n" + "-" * 80)

    geometries_before = len(gdf)
    print("Merging duplicate geometries...")

    # Track events before merging for reporting
    events_before_merge = {
        geom_uuid: [list(row["events"]) for _, row in group.iterrows()]
        for geom_uuid, group in gdf.groupby("geom_uuid")
        if len(group) > 1
    }

    # Preserve CRS before groupby
    original_crs = gdf.crs

    gdf = gdf.groupby("geom_uuid", as_index=False, dropna=False).agg(
        {
            "geometry": "first",
            "events": lambda x: list(
                chain.from_iterable(x)
            ),  # Combine all events from duplicates
        }
    )
    # Convert back to GeoDataFrame and set geometry column
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=original_crs)
    geometries_after = len(gdf)
    print(f"  -> Geometries before: {geometries_before}")
    print(f"  -> Geometries after: {geometries_after}")
    print(f"  -> Geometries removed: {geometries_before - geometries_after}")

    # Verify UUID uniqueness after merging
    n_unique_uuids = gdf["geom_uuid"].nunique()
    if n_unique_uuids != len(gdf):
        raise ValueError(
            f"UUID uniqueness violation! Expected {len(gdf)} unique UUIDs, "
            f"but found {n_unique_uuids}. Some geometries have duplicate UUIDs."
        )
    if n_unique_uuids != geometries_after:
        raise ValueError(
            f"UUID count mismatch! Number of geometries ({geometries_after}) "
            f"does not match number of unique UUIDs ({n_unique_uuids})."
        )
    print(f"  -> UUID uniqueness: verified ({n_unique_uuids} unique UUIDs)")

    # Show how events were combined for duplicate groups
    if events_before_merge:
        print("\n  Events combined from duplicate geometries:")
        for geom_uuid, event_lists in events_before_merge.items():
            total_before = sum(len(events) for events in event_lists)
            merged_events = gdf[gdf["geom_uuid"] == geom_uuid]["events"].iloc[0]
            unique_after = len(set(merged_events))
            print(
                f"    UUID {geom_uuid}: {total_before} total events -> {unique_after} unique events"
            )

    # Step 5: Remove duplicate events within each geometry (keep all unique events)
    print("\n" + "=" * 80)
    print("STEP 5: Removing duplicate events within geometries")
    print("=" * 80)
    events_before = sum(len(events) for events in gdf["events"])
    gdf["events"] = gdf["events"].apply(
        lambda x: list(dict.fromkeys(x))
    )  # Remove duplicates, preserve order
    events_after = sum(len(events) for events in gdf["events"])
    print(f"  -> Events before deduplication: {events_before}")
    print(f"  -> Events after deduplication: {events_after}")
    print(f"  -> Duplicate events removed: {events_before - events_after}")

    # Count unique events across all geometries after deduplication
    all_events_after = [event for events_list in gdf["events"] for event in events_list]
    unique_events_after = len(set(all_events_after))
    print(f"  -> Unique events across all geometries: {unique_events_after}")

    # Step 6: Final data preparation and validation
    print("\n" + "=" * 80)
    print("STEP 6: Final data preparation and validation")
    print("=" * 80)

    # Reorder columns
    final_cols = ["geom_uuid", "events", "geometry"]
    print("Reordering columns...")
    gdf = gdf[final_cols].copy()
    print(f"  -> Columns: {final_cols}")

    # Convert geometry to EPSG:2056 (Swiss LV95)
    target_crs = "EPSG:2056"
    if gdf.crs is None:
        raise ValueError("Geometry has no CRS defined. Cannot convert to EPSG:2056.")
    if str(gdf.crs) != target_crs:
        print("Converting geometry CRS...")
        print(f"  -> From: {gdf.crs}")
        gdf = gdf.to_crs(target_crs)
        print(f"  -> To: {target_crs}")
    else:
        print("Geometry CRS...")
        print(f"  -> Already in {target_crs}, no conversion needed")

    # Enforce data types
    print("Enforcing data types...")
    gdf["geom_uuid"] = gdf["geom_uuid"].astype("string")
    print("  -> geom_uuid: string")

    # Convert events to JSON strings
    print("Converting events to JSON strings...")
    gdf["events"] = gdf["events"].apply(json.dumps)
    print("  -> All events converted to JSON format")

    # Validation
    print("Validating data...")
    if gdf.geometry.dtype.name != "geometry":
        raise ValueError("Geometry column is not valid")
    print("  -> Geometry column: valid")

    # Verify UUID uniqueness (final check)
    n_unique_uuids = gdf["geom_uuid"].nunique()
    if n_unique_uuids != len(gdf):
        raise ValueError(
            f"UUID uniqueness violation! Expected {len(gdf)} unique UUIDs, "
            f"but found {n_unique_uuids}. Some geometries have duplicate UUIDs."
        )
    print(f"  -> UUID uniqueness: verified ({n_unique_uuids} unique UUIDs)")

    # Check for potential hash collisions (different geometries with same UUID)
    # This should never happen with SHA1, but we verify just in case
    uuid_groups = gdf.groupby("geom_uuid")
    for geom_uuid, group in uuid_groups:
        if len(group) > 1:
            # Check if geometries are actually identical
            first_geom = group.iloc[0].geometry
            for idx, row in group.iloc[1:].iterrows():
                if not first_geom.equals(row.geometry):
                    raise ValueError(
                        f"Hash collision detected! UUID {geom_uuid} is assigned to "
                        f"different geometries. This indicates a hash collision."
                    )

    for idx, events_json in enumerate(gdf["events"]):
        if not isinstance(events_json, str):
            raise ValueError(f"Events at index {idx} is not a JSON string")
        # Verify it's valid JSON
        try:
            json.loads(events_json)
        except json.JSONDecodeError:
            raise ValueError(f"Events at index {idx} is not valid JSON")
    print(f"  -> Event entries ({len(gdf)}): all valid JSON")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Final shape: {gdf.shape}")
    print("\nColumn dtypes:")
    print(gdf.dtypes)
    print("\nExample row:")
    print(gdf.iloc[0])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    print(f"\nSaved to {output_path}")


def append_geoms_to_events(
    events_path: Path, geoms_path: Path, output_path: Path
) -> gpd.GeoDataFrame:
    """
    Append combined geometries to events based on event IDs.

    Args:
        events_path: Path to events CSV file
        geoms_path: Path to geometries GeoPackage file
        output_path: Path to save output GeoPackage

    Returns:
        GeoDataFrame with events and combined geometries
    """
    print("=" * 80)
    print("Loading data")
    print("=" * 80)

    # Load events
    print(f"Loading events from {events_path}...")
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    events_df = pd.read_csv(events_path)
    print(f"  -> Loaded {len(events_df)} events")
    print("  -> Columns:")
    for col in events_df.columns:
        print(f"       - {col}")

    # Load geometries
    print(f"\nLoading geometries from {geoms_path}...")
    if not geoms_path.exists():
        raise FileNotFoundError(f"Geometries file not found: {geoms_path}")
    geoms_gdf = gpd.read_file(geoms_path)
    print(f"  -> Loaded {len(geoms_gdf)} geometries")
    print("  -> Columns:")
    for col in geoms_gdf.columns:
        print(f"       - {col}")

    # Parse events JSON strings in geometries
    print("\nParsing events from geometries...")
    geoms_gdf["events_parsed"] = geoms_gdf["events"].apply(json.loads)

    # Count geometries for each event (pk)
    print("\nCounting geometries for each event...")
    event_geom_counts = {}

    for idx, row in events_df.iterrows():
        pk = row["pk"]
        # Count geometries that contain this pk in their events list
        count = 0
        matching_geoms = []
        for geom_idx, geom_row in geoms_gdf.iterrows():
            events_list = geom_row["events_parsed"]
            if pk in events_list:
                count += 1
                matching_geoms.append(geom_idx)
        event_geom_counts[pk] = {"count": count, "geom_indices": matching_geoms}

    # Add count to events dataframe
    events_df["n_geometries"] = events_df["pk"].map(
        lambda pk: event_geom_counts[pk]["count"]
    )

    # Assert all geometries are line geometries
    print("\nVerifying geometry types...")
    for idx, row in geoms_gdf.iterrows():
        geom_type = row.geometry.geom_type
        assert geom_type in ["LineString", "MultiLineString"], (
            f"Geometry at index {idx} is not a line geometry. Found type: {geom_type}"
        )
    print("  -> All geometries are line geometries")

    # Create combined geometry for each event
    print("\nCreating combined geometries for each event...")
    event_geometries = []

    for idx, row in events_df.iterrows():
        pk = row["pk"]
        geom_indices = event_geom_counts[pk]["geom_indices"]

        if len(geom_indices) == 0:
            # No geometries for this event
            event_geometries.append(None)
        elif len(geom_indices) == 1:
            # Single geometry - use it directly
            geom_idx = geom_indices[0]
            event_geometries.append(geoms_gdf.iloc[geom_idx].geometry)
        else:
            # Multiple geometries - combine into MultiLineString
            geometries = [
                geoms_gdf.iloc[geom_idx].geometry for geom_idx in geom_indices
            ]

            # Extract LineStrings from geometries
            line_strings = []
            for geom in geometries:
                if geom.geom_type == "LineString":
                    line_strings.append(geom)
                elif geom.geom_type == "MultiLineString":
                    # Extract individual LineStrings from MultiLineString
                    line_strings.extend(list(geom.geoms))

            # Create MultiLineString from list of LineStrings
            if len(line_strings) == 1:
                event_geometries.append(line_strings[0])
            else:
                multi_geom = MultiLineString(line_strings)
                event_geometries.append(multi_geom)

    # Add geometry column to events dataframe
    events_df["geometry"] = event_geometries

    # Convert to GeoDataFrame (preserve CRS from geometries)
    events_gdf = gpd.GeoDataFrame(events_df, geometry="geometry", crs=geoms_gdf.crs)

    # Print statistics
    print("\nGeometry count statistics:")
    print(f"  -> Events with 0 geometries: {(events_df['n_geometries'] == 0).sum()}")
    print(f"  -> Events with 1 geometry: {(events_df['n_geometries'] == 1).sum()}")
    print(f"  -> Events with 2+ geometries: {(events_df['n_geometries'] >= 2).sum()}")
    print(f"  -> Max geometries per event: {events_df['n_geometries'].max()}")

    print(f"\nFinal events GeoDataFrame shape: {events_gdf.shape}")
    print(f"Events with geometry: {(events_gdf.geometry.notna()).sum()}")
    print(f"Events without geometry: {(events_gdf.geometry.isna()).sum()}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_gdf.to_file(output_path, driver="GPKG")
    print(f"\nSaved to {output_path}")

    return events_gdf
