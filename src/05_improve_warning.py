
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "environmental_features_5min.csv"
)

RESULTS_DIR = ROOT / "results"

TARGET = "target_future_risk"

CLASSES = ["Safe", "Warning", "Critical"]

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


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def main():
    print("=" * 65)
    print("WARNING CLASS IMPROVEMENT EXPERIMENTS")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}\n"
            "Run 02_prepare_data.py and 03_create_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "timestamp",
        "future_timestamp",
        "zone_id",
        TARGET,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    df["future_timestamp"] = pd.to_datetime(
        df["future_timestamp"], errors="coerce", utc=True
    )

    df[TARGET] = (
        df[TARGET].astype("string").str.strip().str.title()
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "future_timestamp",
            "zone_id",
            TARGET,
        ]
    ).copy()

    # Ensure that the future target is exactly five minutes ahead.
    horizon = df["future_timestamp"] - df["timestamp"]

    df = df[horizon == pd.Timedelta(minutes=5)].copy()

    unexpected = set(df[TARGET].unique()) - set(CLASSES)

    if unexpected:
        raise ValueError(
            f"Unexpected risk labels: {sorted(unexpected)}"
        )

    features = [
        column
        for column in SENSOR_COLUMNS
        if column in df.columns
    ]

    if not features:
        raise ValueError("No sensor feature columns were found.")

    for column in features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=features, how="all").copy()

    print("\nDataset rows:", len(df))
    print("Features:", features)


    # ========================================================
    # 3. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
    # ========================================================

    unique_times = np.sort(
        df["timestamp"].drop_duplicates().to_numpy()
    )

    if len(unique_times) < 10:
        raise ValueError(
            "Too few unique timestamps for chronological splitting."
        )

    train_cutoff = unique_times[
        int(len(unique_times) * 0.70)
    ]

    validation_cutoff = unique_times[
        int(len(unique_times) * 0.85)
    ]

    # Remove training targets that cross into validation.
    train = df[
        (df["timestamp"] < train_cutoff)
        & (df["future_timestamp"] < train_cutoff)
    ].copy()

    # Remove validation targets that cross into the test period.
    validation = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < validation_cutoff)
        & (df["future_timestamp"] < validation_cutoff)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Training or validation data is empty."
        )

    print("\nTraining rows:", len(train))
    print("Validation rows:", len(validation))

    print("\nTraining class distribution:")
    print(train[TARGET].value_counts())

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    # Important fix: classes must be a NumPy array.
    present_classes = np.array(sorted(y_train.unique()))

    if set(present_classes) != set(CLASSES):
        raise ValueError(
            "Training data must contain Safe, Warning, and Critical."
        )


    # ========================================================
    # 4. CALCULATE CLASS WEIGHTS
    # ========================================================

    balanced_values = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )

    balanced_weights = dict(
        zip(present_classes, balanced_values)
    )

    weight_options = {
        "balanced": balanced_weights,
        "warning_1_5x": {
            **balanced_weights,
            "Warning": balanced_weights["Warning"] * 1.5,
        },
        "warning_2x": {
            **balanced_weights,
            "Warning": balanced_weights["Warning"] * 2.0,
        },
    }

    print("\nClass-weight configurations:")
    for name, weights in weight_options.items():
        print(name, weights)


    # ========================================================
    # 5. TRAIN AND COMPARE MODELS ON VALIDATION DATA
    # ========================================================

    results = []
    reports = []

    for name, weights in weight_options.items():
        print(f"\nTraining experiment: {name}")

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight=weights,
                    n_jobs=-1,
                ),
            ),
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        accuracy = accuracy_score(
            y_validation, predictions
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )

        report_dict = classification_report(
            y_validation,
            predictions,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        warning_metrics = report_dict["Warning"]

        results.append({
            "experiment": name,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "warning_precision": warning_metrics["precision"],
            "warning_recall": warning_metrics["recall"],
            "warning_f1": warning_metrics["f1-score"],
        })

        reports.append(
            f"\n{'=' * 60}\n"
            f"Experiment: {name}\n"
            + classification_report(
                y_validation,
                predictions,
                labels=CLASSES,
                zero_division=0,
            )
        )

        print("Validation accuracy:", round(accuracy, 4))
        print("Validation macro F1:", round(macro_f1, 4))
        print(
            "Warning precision:",
            round(warning_metrics["precision"], 4),
        )
        print(
            "Warning recall:",
            round(warning_metrics["recall"], 4),
        )


    # ========================================================
    # 6. SAVE COMPARISON RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    comparison_path = (
        RESULTS_DIR / "warning_weight_experiments.csv"
    )

    results_df.to_csv(comparison_path, index=False)

    reports_path = (
        RESULTS_DIR / "warning_weight_reports.txt"
    )

    with open(
        reports_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Warning Class-Weight Experiments\n")
        file.write("Evaluation uses validation data only.\n\n")
        file.write(results_df.to_string(index=False))
        file.write("\n")
        file.writelines(reports)

    print("\n" + "=" * 65)
    print("VALIDATION COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nSaved comparison:", comparison_path)
    print("Saved reports:", reports_path)

    print(
        "\nCompleted. The held-out test dataset was not used "
        "for selecting these class weights."
    )


if __name__ == "__main__":
    main()
