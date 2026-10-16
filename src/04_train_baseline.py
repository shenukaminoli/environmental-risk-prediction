
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
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


def main():
    print("=" * 60)
    print("RANDOM FOREST - BASELINE TRAINING")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}\n"
            "Run 02_prepare_data.py and 03_create_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {"timestamp", "zone_id", TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse timestamps and normalize labels.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()

    df = df.dropna(
        subset=["timestamp", "zone_id", TARGET]
    ).copy()

    unexpected = set(df[TARGET].unique()) - set(CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected target classes: {sorted(unexpected)}")

    features = [col for col in SENSOR_COLUMNS if col in df.columns]

    if not features:
        raise ValueError("No sensor feature columns were found.")

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where all input sensors are missing.
    df = df.dropna(subset=features, how="all").copy()

    # Use chronological cutoffs based on unique timestamps.
    # This prevents rows sharing a timestamp from crossing splits.
    timestamps = np.sort(df["timestamp"].unique())

    if len(timestamps) < 10:
        raise ValueError(
            "Too few unique timestamps for a meaningful time-based split."
        )

    train_cutoff = timestamps[int(len(timestamps) * 0.70)]
    validation_cutoff = timestamps[int(len(timestamps) * 0.85)]

    train_df = df[df["timestamp"] < train_cutoff].copy()
    validation_df = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < validation_cutoff)
    ].copy()
    test_df = df[df["timestamp"] >= validation_cutoff].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError("One of the chronological data splits is empty.")

    print("\nDataset rows:", len(df))
    print("Features:", features)
    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Testing rows:", len(test_df))
    print("Training period:", train_df["timestamp"].min(),
          "to", train_df["timestamp"].max())
    print("Validation period:", validation_df["timestamp"].min(),
          "to", validation_df["timestamp"].max())
    print("Testing period:", test_df["timestamp"].min(),
          "to", test_df["timestamp"].max())

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts())

    # Fit preprocessing and model using training data only.
    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_validation = validation_df[features]
    y_validation = validation_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    # Balanced weights help when some risk categories are less common.
    present_classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    class_weights = dict(zip(present_classes, weights))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weights,
            n_jobs=-1,
        )),
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Validation is reported separately; test data is not used for fitting.
    validation_predictions = model.predict(X_validation)
    validation_accuracy = accuracy_score(
        y_validation, validation_predictions
    )
    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    print("\nVALIDATION RESULTS")
    print("Accuracy:", round(validation_accuracy, 4))
    print("Macro F1:", round(validation_f1, 4))

    # Final evaluation on the held-out chronological test period.
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(
        y_test,
        test_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    print("\nTEST RESULTS")
    print("Accuracy:", round(test_accuracy, 4))
    print("Macro F1:", round(test_f1, 4))

    print("\nClassification report:")
    report = classification_report(
        y_test,
        test_predictions,
        labels=CLASSES,
        zero_division=0,
    )
    print(report)

    matrix = confusion_matrix(
        y_test, test_predictions, labels=CLASSES
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Predicted_{c}" for c in CLASSES],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "classes": CLASSES,
            "target": TARGET,
            "prediction_horizon_minutes": 5,
        },
        MODEL_DIR / "random_forest_baseline.joblib",
    )

    matrix_df.to_csv(RESULTS_DIR / "baseline_confusion_matrix.csv")

    with open(
        RESULTS_DIR / "baseline_metrics.txt", "w", encoding="utf-8"
    ) as file:
        file.write("Random Forest Baseline\n")
        file.write(f"Features: {features}\n")
        file.write(f"Validation accuracy: {validation_accuracy:.4f}\n")
        file.write(f"Validation macro F1: {validation_f1:.4f}\n")
        file.write(f"Test accuracy: {test_accuracy:.4f}\n")
        file.write(f"Test macro F1: {test_f1:.4f}\n\n")
        file.write(report)
        file.write("\n\nConfusion matrix:\n")
        file.write(matrix_df.to_string())

    print("\nSaved model:", MODEL_DIR / "random_forest_baseline.joblib")
    print("Saved metrics:", RESULTS_DIR / "baseline_metrics.txt")
    print("Saved confusion matrix:",
          RESULTS_DIR / "baseline_confusion_matrix.csv")


if __name__ == "__main__":
    main()
