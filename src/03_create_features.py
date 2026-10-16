
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "environmental_data_cleaned.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "environmental_features_5min.csv"

HORIZON = pd.Timedelta(minutes=5)
TOLERANCE = pd.Timedelta(minutes=1)

SENSOR_COLUMNS = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "pm25_ugm3",
    "pm10_ugm3",
    "noise_dBA",
    "light_lux",
    "uv_index",
]

RISK_CLASSES = {"Safe", "Warning", "Critical"}


def main():
    print("=" * 60)
    print("FIVE-MINUTE FUTURE RISK TARGET CONSTRUCTION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run python src/02_prepare_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"timestamp", "zone_id", "overall_zone_risk"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    df["overall_zone_risk"] = (
        df["overall_zone_risk"].astype("string").str.strip().str.title()
    )

    df = df.dropna(
        subset=["timestamp", "zone_id", "overall_zone_risk"]
    ).copy()

    unexpected = set(df["overall_zone_risk"].unique()) - RISK_CLASSES
    if unexpected:
        raise ValueError(
            f"Unexpected risk labels: {sorted(unexpected)}. "
            "Resolve these labels before training."
        )

    sensor_columns = [
        col for col in SENSOR_COLUMNS if col in df.columns
    ]
    if not sensor_columns:
        raise ValueError("No expected sensor columns were found.")

    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    # Duplicate timestamps can make the future match ambiguous.
    duplicate_mask = df.duplicated(
        subset=["zone_id", "timestamp"], keep=False
    )
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} rows sharing timestamps within "
            "a zone. Resolve duplicate timestamps before proceeding."
        )

    results = []

    for zone, zone_df in df.groupby("zone_id", sort=False):
        zone_df = zone_df.sort_values("timestamp").copy()

        # Report observed sampling intervals for this zone.
        intervals = zone_df["timestamp"].diff().dropna()
        print(f"\nZone: {zone}")
        print("Observations:", len(zone_df))

        if not intervals.empty:
            print("Median sampling interval:", intervals.median())
            print("Intervals close to five minutes:",
                  int(intervals.between(
                      pd.Timedelta(minutes=4),
                      pd.Timedelta(minutes=6),
                  ).sum()))

        # Features contain current-time sensor values only.
        current = zone_df[
            ["timestamp", "zone_id"] + sensor_columns
        ].copy()

        current["target_time"] = current["timestamp"] + HORIZON

        # Future table supplies the risk label and its actual timestamp.
        future = zone_df[["timestamp", "overall_zone_risk"]].copy()
        future = future.rename(
            columns={
                "timestamp": "future_timestamp",
                "overall_zone_risk": "target_future_risk",
            }
        )

        # Match only observations at or after current time + 5 minutes.
        # Each zone is processed independently to prevent zone mixing.
        matched = pd.merge_asof(
            current.sort_values("target_time"),
            future.sort_values("future_timestamp"),
            left_on="target_time",
            right_on="future_timestamp",
            direction="forward",
            tolerance=TOLERANCE,
        )

        matched["target_offset_minutes"] = (
            matched["future_timestamp"] - matched["timestamp"]
        ).dt.total_seconds() / 60

        unmatched = int(matched["target_future_risk"].isna().sum())
        print("Rows without a suitable future target:", unmatched)

        matched = matched.dropna(
            subset=["future_timestamp", "target_future_risk"]
        ).copy()

        # Missing current sensor readings cannot be used by most models.
        matched = matched.dropna(subset=sensor_columns).copy()

        print("Usable feature rows:", len(matched))
        results.append(matched)

    if not results:
        raise ValueError("No zone data was available for feature creation.")

    output = pd.concat(results, ignore_index=True)
    output = output.sort_values(
        ["zone_id", "timestamp"]
    ).reset_index(drop=True)

    output = output[
        [
            "timestamp",
            "zone_id",
            "target_time",
            "future_timestamp",
            "target_offset_minutes",
            *sensor_columns,
            "target_future_risk",
        ]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("FUTURE TARGET CONSTRUCTION COMPLETED")
    print("=" * 60)
    print("Output rows:", len(output))
    print("\nFuture risk class counts:")
    print(output["target_future_risk"].value_counts())
    print("\nRows per zone:")
    print(output["zone_id"].value_counts())
    print("\nActual target offset (minutes):")
    print(output["target_offset_minutes"].describe())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
