
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sensebox_risk_labeled (1).xlsx"
)

ZONE_SHEETS = [
    "Zone 1 (Node 1)",
    "Zone 2 (Node 2)",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Excel dataset not found: {DATA_FILE}"
        )

    print("Environmental Risk Prediction")
    print("=" * 40)

    excel = pd.ExcelFile(DATA_FILE)
    print("\nAvailable sheets:")
    print(excel.sheet_names)

    all_zones = []

    for sheet_name in ZONE_SHEETS:
        df = pd.read_excel(
            DATA_FILE,
            sheet_name=sheet_name,
        )

        print(f"\n--- {sheet_name} ---")
        print("Rows and columns:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        if "timestamp" not in df.columns:
            print("WARNING: timestamp column not found.")

        if "overall_zone_risk" not in df.columns:
            print("WARNING: risk label column not found.")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            print("\nTimestamp range:")
            print(df["timestamp"].agg(["min", "max"]))

            print(
                "Invalid timestamps:",
                df["timestamp"].isna().sum(),
            )

        if "overall_zone_risk" in df.columns:
            print("\nRisk distribution:")
            print(
                df["overall_zone_risk"]
                .value_counts(dropna=False)
            )

        print("\nMissing values by column:")
        print(df.isna().sum())

        print(
            "\nDuplicate rows:",
            df.duplicated().sum(),
        )

        df["zone_id"] = sheet_name
        all_zones.append(df)

    combined = pd.concat(
        all_zones,
        ignore_index=True,
    )

    print("\n--- COMBINED DATASET ---")
    print("Total rows:", len(combined))
    print("Total columns:", len(combined.columns))

    print("\nRows by zone:")
    print(combined["zone_id"].value_counts())

    print("\nFirst five records:")
    print(combined.head())

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
