
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "environmental_features_5min.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET = "target_future_risk"

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

CLASSES = ["Safe", "Warning", "Critical"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def main():
    print("=" * 65)
    print("ENVIRONMENTAL RISK PREDICTION")
    print("RANDOM FOREST BASELINE TRAINING")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_FILE}\n"
            "Run 02_prepare_data.py and 03_create_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "timestamp",
        "future_timestamp",
        "zone_id",
        TARGET,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    # Parse timestamps.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    df["future_timestamp"] = pd.to_datetime(
        df["future_timestamp"], errors="coerce", utc=True
    )

    # Normalize target labels.
    df[TARGET] = (
        df[TARGET]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "future_timestamp",
            "zone_id",
            TARGET,
        ]
    ).copy()

    # Check target classes.
    unexpected_classes = set(df[TARGET].unique()) - set(CLASSES)

    if unexpected_classes:
        raise ValueError(
            f"Unexpected target classes: {sorted(unexpected_classes)}"
        )

    # Check prediction horizon.
    df["horizon_minutes"] = (
        df["future_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 60

    invalid_horizon = ~np.isclose(
        df["horizon_minutes"], 5.0
    )

    if invalid_horizon.any():
        raise ValueError(
            f"{int(invalid_horizon.sum())} rows do not have "
            "an exact five-minute target horizon."
        )

    # Use available sensor columns only.
    features = [
        column
        for column in SENSOR_COLUMNS
        if column in df.columns
    ]

    if not features:
        raise ValueError("No expected sensor columns were found.")

    for column in features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Exclude rows with no usable sensor measurements.
    df = df.dropna(
        subset=features,
        how="all"
    ).copy()

    print("\nDATASET SUMMARY")
    print("-" * 65)
    print("Dataset rows:", len(df))
    print("Sensor features:", features)
    print("Zones:", sorted(df["zone_id"].unique()))
    print("\nTarget class distribution:")
    print(df[TARGET].value_counts())


    # ========================================================
    # 3. CHRONOLOGICAL SPLITTING
    # ========================================================

    # Use unique timestamps so simultaneous zone observations
    # are assigned to the same chronological period.
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

    # Training: target must be observed before validation starts.
    train_df = df[
        (df["timestamp"] < train_cutoff)
        & (df["future_timestamp"] < train_cutoff)
    ].copy()

    # Validation: target must be observed before testing starts.
    validation_df = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < validation_cutoff)
        & (df["future_timestamp"] < validation_cutoff)
    ].copy()

    # Test: latest chronological period.
    test_df = df[
        df["timestamp"] >= validation_cutoff
    ].copy()

    if min(
        len(train_df),
        len(validation_df),
        len(test_df),
    ) == 0:
        raise ValueError(
            "At least one chronological split is empty."
        )

    print("\nCHRONOLOGICAL SPLITS")
    print("-" * 65)

    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Testing rows:", len(test_df))

    print("\nTraining period:")
    print(train_df["timestamp"].min(), "to",
          train_df["timestamp"].max())

    print("\nValidation period:")
    print(validation_df["timestamp"].min(), "to",
          validation_df["timestamp"].max())

    print("\nTesting period:")
    print(test_df["timestamp"].min(), "to",
          test_df["timestamp"].max())

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts())

    # Check that each class exists in training data.
    missing_training_classes = (
        set(CLASSES) - set(train_df[TARGET].unique())
    )

    if missing_training_classes:
        raise ValueError(
            "Training data is missing classes: "
            f"{sorted(missing_training_classes)}"
        )

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_validation = validation_df[features]
    y_validation = validation_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]


    # ========================================================
    # 4. TRAIN RANDOM FOREST
    # ========================================================

    # Calculate class weights using training data only.
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASSES),
        y=y_train,
    )

    class_weights = dict(zip(CLASSES, weights))

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
                class_weight=class_weights,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTRAINING RANDOM FOREST")
    print("-" * 65)

    model.fit(X_train, y_train)

    print("Training completed.")


    # ========================================================
    # 5. VALIDATION EVALUATION
    # ========================================================

    validation_predictions = model.predict(X_validation)

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    print("\nVALIDATION RESULTS")
    print("-" * 65)
    print("Accuracy:", round(validation_accuracy, 4))
    print("Macro F1:", round(validation_macro_f1, 4))


    # ========================================================
    # 6. FINAL TEST EVALUATION
    # ========================================================

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=CLASSES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Predicted_{c}" for c in CLASSES],
    )

    print("\nTEST RESULTS")
    print("-" * 65)
    print("Accuracy:", round(test_accuracy, 4))
    print("Macro F1:", round(test_macro_f1, 4))

    print("\nCLASSIFICATION REPORT")
    print(report)

    print("CONFUSION MATRIX")
    print(matrix_df)


    # ========================================================
    # 7. SAVE MODEL AND RESULTS LOCALLY
    # ========================================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "random_forest_baseline.joblib"

    joblib.dump(
        {
            "model": model,
            "features": features,
            "classes": CLASSES,
            "target": TARGET,
            "prediction_horizon_minutes": 5,
        },
        model_path,
    )

    matrix_path = (
        RESULTS_DIR / "baseline_confusion_matrix.csv"
    )

    matrix_df.to_csv(matrix_path)

    metrics_path = RESULTS_DIR / "baseline_metrics.txt"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Random Forest Baseline Results\n")
        file.write("=" * 40 + "\n")
        file.write(f"Dataset rows: {len(df)}\n")
        file.write(f"Features: {features}\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Validation rows: {len(validation_df)}\n")
        file.write(f"Testing rows: {len(test_df)}\n")
        file.write(
            f"Validation accuracy: {validation_accuracy:.4f}\n"
        )
        file.write(
            f"Validation macro F1: {validation_macro_f1:.4f}\n"
        )
        file.write(
            f"Test accuracy: {test_accuracy:.4f}\n"
        )
        file.write(
            f"Test macro F1: {test_macro_f1:.4f}\n\n"
        )
        file.write("Classification report:\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(matrix_df.to_string())

    print("\nSAVED FILES")
    print("-" * 65)
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Confusion matrix:", matrix_path)

    print("\nBASELINE TRAINING AND EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
