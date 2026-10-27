"""Train and evaluate the TrafficSense AI ANN using chronological splits."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import ROOT, TARGET_COLUMN, load_dataset, make_features, traffic_volume_to_class

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "traffic_sense_ann.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

NUMERIC_FEATURES = ["hour", "day_of_week", "month", "temp", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_FEATURES = ["holiday", "weather_main"]
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("Date range:", df["date_time"].min(), "to", df["date_time"].max())
    print("Missing values by column:\n", df.isna().sum().to_string())
    print("Exact duplicate rows after cleanup:", int(df.duplicated().sum()))

    # Chronological 70/15/15 split: later observations remain unseen until evaluation.
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Dataset is too small for the planned chronological split.")

    # Thresholds are learned from training-period traffic volume only.
    low_threshold = float(train_df[TARGET_COLUMN].quantile(1 / 3))
    high_threshold = float(train_df[TARGET_COLUMN].quantile(2 / 3))
    if low_threshold >= high_threshold:
        raise ValueError("Training traffic-volume quantiles do not form three distinct ranges.")

    def split_xy(part):
        X = make_features(part)
        y = traffic_volume_to_class(part[TARGET_COLUMN], low_threshold, high_threshold)
        return X[FEATURE_NAMES], pd.Series(y, index=part.index, name="congestion_class")

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.001, max_iter=250,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=15, random_state=42,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("ann", model)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
    for warning in caught:
        print(f"WARNING during training: {warning.message}")

    # Validation results are reported for transparency; final metrics use test data.
    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)
    labels = ["Low Congestion", "Moderate Congestion", "High Congestion"]
    report = classification_report(y_test, test_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "dataset_source": "UCI Machine Learning Repository: Metro Interstate Traffic Volume",
        "dataset_url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        "dataset_license": "CC BY 4.0 (as listed by UCI)",
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(train_df)), "validation_rows": int(len(val_df)), "test_rows": int(len(test_df)),
        "date_start": str(df["date_time"].min()), "date_end": str(df["date_time"].max()),
        "low_volume_threshold_inclusive": low_threshold,
        "moderate_volume_upper_threshold_inclusive": high_threshold,
        "features": FEATURE_NAMES,
        "target_definition": "Derived traffic-volume classes; not independently observed congestion labels.",
        "model": "Scikit-learn MLPClassifier; hidden layers (64, 32), ReLU, Adam, random_state=42",
        "iterations": int(pipeline.named_steps["ann"].n_iter_),
        "classes": labels,
        "class_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().items()},
    }
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save useful plots based only on actual dataset values.
    plot_df = df.copy()
    plot_df["hour"] = plot_df["date_time"].dt.hour
    plot_df["derived_class"] = traffic_volume_to_class(plot_df[TARGET_COLUMN], low_threshold, high_threshold)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x="derived_class", order=labels)
    plt.title("Derived Traffic-Volume Category Distribution")
    plt.xlabel("Derived category")
    plt.ylabel("Number of observations")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=plot_df.groupby("hour", as_index=False)[TARGET_COLUMN].mean(), x="hour", y=TARGET_COLUMN, marker="o")
    plt.title("Average Recorded Traffic Volume by Hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Average traffic volume")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traffic_by_hour.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("ANN Confusion Matrix — Test Set")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual derived category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nFINAL TEST RESULTS (later chronological holdout)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    print(classification_report(y_test, test_pred, labels=labels, zero_division=0))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved charts in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
