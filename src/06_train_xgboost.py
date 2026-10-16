
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "environmental_features_5min.csv"
)

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TARGET = "target_future_risk"

CLASSES = ["Safe", "Warning", "Critical"]

# XGBoost requires numeric class labels: 0, 1, 2.
CLASS_TO_ID = {
    "Safe": 0,
    "Warning": 1,
    "Critical": 2,
}

ID_TO_CLASS = {
    value: key for key, value in CLASS_TO_ID.items()
}

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
    print("ENVIRONMENTAL RISK PREDICTION")
    print("XGBOOST MODEL TRAINING")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run 02_prepare_data.py and 03_create_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "timestamp",
        "future_timestamp",
        "zone_id",
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}"
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

    unexpected = set(df[TARGET].unique()) - set(CLASSES)

    if unexpected:
        raise ValueError(
            f"Unexpected target classes: {sorted(unexpected)}"
        )

    # Keep only rows with an exact five-minute target horizon.
    horizon = df["future_timestamp"] - df["timestamp"]

    df = df[horizon == pd.Timedelta(minutes=5)].copy()

    features = [
        column for column in SENSOR_COLUMNS
        if column in df.columns
    ]

    if not features:
        raise ValueError("No sensor features were found.")

    for column in features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=features, how="all").copy()

    print("\nDataset rows:", len(df))
    print("Features:", features)
    print("\nRisk class distribution:")
    print(df[TARGET].value_counts())


    # ========================================================
    # 3. CHRONOLOGICAL SPLIT
    # ========================================================

    # Split using unique timestamps so simultaneous zone records
    # stay in the same period.
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

    train = df[
        (df["timestamp"] < train_cutoff)
        & (df["future_timestamp"] < train_cutoff)
    ].copy()

    validation = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < validation_cutoff)
        & (df["future_timestamp"] < validation_cutoff)
    ].copy()

    test = df[
        df["timestamp"] >= validation_cutoff
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A chronological split is empty.")

    print("\nCHRONOLOGICAL SPLITS")
    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Testing rows:", len(test))

    print("\nTraining period:")
    print(train["timestamp"].min(), "to",
          train["timestamp"].max())

    print("\nValidation period:")
    print(validation["timestamp"].min(), "to",
          validation["timestamp"].max())

    print("\nTesting period:")
    print(test["timestamp"].min(), "to",
          test["timestamp"].max())


    # ========================================================
    # 4. PREPARE FEATURES AND LABELS
    # ========================================================

    X_train = train[features]
    X_validation = validation[features]
    X_test = test[features]

    y_train = train[TARGET].map(CLASS_TO_ID).astype(int)
    y_validation = validation[TARGET].map(CLASS_TO_ID).astype(int)
    y_test = test[TARGET].map(CLASS_TO_ID).astype(int)

    if set(y_train.unique()) != {0, 1, 2}:
        raise ValueError(
            "Training data must contain Safe, Warning, and Critical."
        )

    # Compute balanced sample weights from training data only.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # Increase Warning's training weight by 2x, based on the
    # earlier validation experiment. This is a candidate strategy.
    sample_weights = sample_weights * np.where(
        y_train.to_numpy() == CLASS_TO_ID["Warning"],
        2.0,
        1.0,
    )


    # ========================================================
    # 5. TRAIN XGBOOST
    # ========================================================

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ])

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights,
    )

    print("Training completed.")


    # ========================================================
    # 6. VALIDATION EVALUATION
    # ========================================================

    validation_ids = model.predict(X_validation).astype(int)

    validation_predictions = pd.Series(
        validation_ids
    ).map(ID_TO_CLASS).to_numpy()

    y_validation_names = y_validation.map(ID_TO_CLASS).to_numpy()

    validation_accuracy = accuracy_score(
        y_validation_names,
        validation_predictions,
    )

    validation_macro_f1 = f1_score(
        y_validation_names,
        validation_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    validation_report = classification_report(
        y_validation_names,
        validation_predictions,
        labels=CLASSES,
        zero_division=0,
    )

    print("\nVALIDATION RESULTS")
    print("-" * 65)
    print("Accuracy:", round(validation_accuracy, 4))
    print("Macro F1:", round(validation_macro_f1, 4))
    print("\nClassification report:")
    print(validation_report)


    # ========================================================
    # 7. FINAL TEST EVALUATION
    # ========================================================

    test_ids = model.predict(X_test).astype(int)

    test_predictions = pd.Series(
        test_ids
    ).map(ID_TO_CLASS).to_numpy()

    y_test_names = y_test.map(ID_TO_CLASS).to_numpy()

    test_accuracy = accuracy_score(
        y_test_names,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test_names,
        test_predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    test_report = classification_report(
        y_test_names,
        test_predictions,
        labels=CLASSES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test_names,
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
    print("\nClassification report:")
    print(test_report)
    print("\nConfusion matrix:")
    print(matrix_df)


    # ========================================================
    # 8. SAVE MODEL AND RESULTS LOCALLY
    # ========================================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_risk_model.joblib"

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
        RESULTS_DIR / "xgboost_confusion_matrix.csv"
    )
    matrix_df.to_csv(matrix_path)

    metrics_path = RESULTS_DIR / "xgboost_metrics.txt"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("XGBoost Environmental Risk Model\n")
        file.write("=" * 40 + "\n")
        file.write(f"Features: {features}\n")
        file.write(f"Training rows: {len(train)}\n")
        file.write(f"Validation rows: {len(validation)}\n")
        file.write(f"Testing rows: {len(test)}\n")
        file.write(
            f"Validation accuracy: {validation_accuracy:.4f}\n"
        )
        file.write(
            f"Validation macro F1: {validation_macro_f1:.4f}\n"
        )
        file.write(f"Test accuracy: {test_accuracy:.4f}\n")
        file.write(f"Test macro F1: {test_macro_f1:.4f}\n\n")
        file.write("Validation report:\n")
        file.write(validation_report)
        file.write("\nTest report:\n")
        file.write(test_report)
        file.write("\nConfusion matrix:\n")
        file.write(matrix_df.to_string())

    print("\nSAVED FILES")
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Confusion matrix:", matrix_path)

    print("\nXGBOOST TRAINING AND EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
