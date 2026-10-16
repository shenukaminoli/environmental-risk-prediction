
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "sensebox_risk_labeled (1).xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "environmental_data_cleaned.csv"

# Expected sensor columns. The script checks which ones exist.
NUMERIC_COLUMNS = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "pm25_ugm3",
    "pm10_ugm3",
    "noise_dBA",
    "light_lux",
    "uv_index",
]

REQUIRED_COLUMNS = ["timestamp", "overall_zone_risk"]

EXPECTED_SHEETS = [
    "Zone 1 (Node 1)",
    "Zone 2 (Node 2)",
]


def main():
    print("=" * 60)
    print("ENVIRONMENTAL RISK - DATA PREPARATION")
    print("=" * 60)

    # 1. Check the input file
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nExcel file not found:\n{RAW_FILE}\n"
            "Check the filename and the data/raw folder."
        )

    # 2. Check available Excel sheets
    excel = pd.ExcelFile(RAW_FILE)
    print("\nAvailable sheets:", excel.sheet_names)

    missing_sheets = [
        name for name in EXPECTED_SHEETS
        if name not in excel.sheet_names
    ]

    if missing_sheets:
        raise ValueError(
            f"Expected sheets not found: {missing_sheets}\n"
            f"Available sheets: {excel.sheet_names}\n"
            "Update EXPECTED_SHEETS to match the workbook."
        )

    cleaned_zones = []

    # 3. Prepare each zone independently
    for sheet_name in EXPECTED_SHEETS:
        print(f"\n{'-' * 60}")
        print(f"Processing: {sheet_name}")

        df = pd.read_excel(RAW_FILE, sheet_name=sheet_name)
        df.columns = df.columns.astype(str).str.strip()

        print("Original rows:", len(df))
        print("Columns:", df.columns.tolist())

        # Check essential columns
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{sheet_name} is missing required columns: "
                f"{missing_columns}"
            )

        original_rows = len(df)

        # 4. Parse timestamps and report invalid values
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        invalid_timestamps = int(df["timestamp"].isna().sum())
        print("Invalid timestamps:", invalid_timestamps)

        # 5. Clean risk labels
        df["overall_zone_risk"] = (
            df["overall_zone_risk"]
            .astype("string")
            .str.strip()
            .str.title()
        )

        df["overall_zone_risk"] = df[
            "overall_zone_risk"
        ].replace("", pd.NA)

        # 6. Convert available sensor columns to numeric
        available_numeric = [
            col for col in NUMERIC_COLUMNS
            if col in df.columns
        ]

        for col in available_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print("Sensor columns checked:", available_numeric)

        # 7. Remove rows without timestamp or target label
        missing_labels = int(
            df["overall_zone_risk"].isna().sum()
        )

        df = df.dropna(
            subset=["timestamp", "overall_zone_risk"]
        ).copy()

        print("Rows removed for invalid timestamp/label:",
              original_rows - len(df))

        # 8. Keep track of the original zone
        df["zone_id"] = sheet_name

        # 9. Sort chronologically within this zone
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Report duplicate timestamps. Do not silently remove them.
        duplicate_timestamps = int(
            df.duplicated(subset=["timestamp"], keep=False).sum()
        )
        print("Rows sharing a timestamp in this zone:",
              duplicate_timestamps)

        # Report labels rather than assuming the classes are correct
        print("\nRisk label counts:")
        print(df["overall_zone_risk"].value_counts(dropna=False))

        unexpected_labels = sorted(
            set(df["overall_zone_risk"].dropna().unique())
            - {"Safe", "Warning", "Critical"}
        )

        if unexpected_labels:
            print("WARNING: Unexpected risk labels:",
                  unexpected_labels)

        print("Prepared rows:", len(df))
        cleaned_zones.append(df)

    # 10. Combine only after each zone has been processed separately
    combined = pd.concat(cleaned_zones, ignore_index=True)

    # Sort for readability; zone_id remains available to separate zones
    combined = combined.sort_values(
        ["zone_id", "timestamp"]
    ).reset_index(drop=True)

    # 11. Save the processed data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print("Combined rows:", len(combined))
    print("Columns:", combined.columns.tolist())
    print("\nRows per zone:")
    print(combined["zone_id"].value_counts())
    print("\nTimestamp range:")
    print(combined["timestamp"].min())
    print(combined["timestamp"].max())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
