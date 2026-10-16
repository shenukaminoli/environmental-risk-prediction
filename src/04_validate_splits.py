
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    ROOT / "data" / "processed"
    / "environmental_features_5min.csv"
)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {
        "timestamp",
        "future_timestamp",
        "zone_id",
        "target_future_risk",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    df["future_timestamp"] = pd.to_datetime(
        df["future_timestamp"], errors="coerce", utc=True
    )
    df = df.dropna(
        subset=["timestamp", "future_timestamp", "zone_id"]
    ).copy()

    # Check target horizon.
    df["horizon_minutes"] = (
        df["future_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 60

    print("\n1. TARGET HORIZON")
    print(df["horizon_minutes"].describe())
    print("Outside 5–6 minutes:",
          int((~df["horizon_minutes"].between(5, 6)).sum()))

    # Check duplicate feature timestamps within zones.
    print("\n2. DUPLICATE TIMESTAMPS")
    duplicates = df.duplicated(
        subset=["zone_id", "timestamp"], keep=False
    )
    print("Duplicate rows within zones:", int(duplicates.sum()))

    # Check shared timestamps across zones.
    print("\n3. SHARED TIMESTAMPS ACROSS ZONES")
    zone_counts = df.groupby("timestamp")["zone_id"].nunique()
    print("Timestamps shared by multiple zones:",
          int((zone_counts > 1).sum()))

    # Recreate the chronological split boundaries.
    unique_times = df["timestamp"].drop_duplicates().sort_values().to_numpy()

    if len(unique_times) < 10:
        raise ValueError("Too few unique timestamps.")

    train_cutoff = unique_times[int(len(unique_times) * 0.70)]
    test_cutoff = unique_times[int(len(unique_times) * 0.85)]

    print("\n4. CHRONOLOGICAL SPLIT BOUNDARIES")
    print("Training ends before:", train_cutoff)
    print("Validation ends before:", test_cutoff)

    # A training target must be observed before validation starts.
    train = df[df["timestamp"] < train_cutoff]
    train_purged = train[
        train["future_timestamp"] < train_cutoff
    ]

    # A validation target must be observed before testing starts.
    validation = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < test_cutoff)
    ]
    validation_purged = validation[
        validation["future_timestamp"] < test_cutoff
    ]

    test = df[df["timestamp"] >= test_cutoff]

    print("\n5. SPLIT OVERLAP CHECK")
    print("Original training rows:", len(train))
    print("Training rows removed at boundary:",
          len(train) - len(train_purged))
    print("Safe training rows:", len(train_purged))

    print("\nOriginal validation rows:", len(validation))
    print("Validation rows removed at boundary:",
          len(validation) - len(validation_purged))
    print("Safe validation rows:", len(validation_purged))
    print("Test rows:", len(test))

    if not train_purged.empty:
        print("\nLatest training target:",
              train_purged["future_timestamp"].max())

    if not validation_purged.empty:
        print("Latest validation target:",
              validation_purged["future_timestamp"].max())

    print("\nValidation completed. Review these results before retraining.")


if __name__ == "__main__":
    main()
