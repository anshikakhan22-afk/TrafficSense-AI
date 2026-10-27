"""Automated unit and integration test suite for TrafficSense AI."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_preprocessing import (
    ROOT,
    DATA_DIR,
    TARGET_COLUMN,
    load_dataset,
    make_features,
    traffic_volume_to_class,
)

MODEL_PATH = ROOT / "models" / "traffic_sense_ann.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
OUTPUT_DIR = ROOT / "outputs"


def test_dataset_loading():
    """Verify that dataset loads cleanly with expected shape and no NaNs in target."""
    df = load_dataset()
    assert len(df) == 48187, f"Expected 48,187 rows, got {len(df)}"
    assert TARGET_COLUMN in df.columns, "Target column missing"
    assert "date_time" in df.columns, "date_time column missing"
    assert df[TARGET_COLUMN].isna().sum() == 0, "Target column contains NaNs"


def test_feature_engineering():
    """Verify that feature engineering extracts valid features and excludes target."""
    df = load_dataset().head(100)
    features = make_features(df)
    
    # Target leakage check
    assert TARGET_COLUMN not in features.columns, "CRITICAL: Target leakage detected in features!"
    
    expected_cols = ["hour", "day_of_week", "month", "temp", "rain_1h", "snow_1h", "clouds_all", "holiday", "weather_main"]
    for col in expected_cols:
        assert col in features.columns, f"Missing feature: {col}"
    
    assert features["hour"].between(0, 23).all(), "Hour out of 0-23 bounds"
    assert features["day_of_week"].between(0, 6).all(), "Day of week out of 0-6 bounds"
    assert features["month"].between(1, 12).all(), "Month out of 1-12 bounds"


def test_quantile_labeling():
    """Verify that traffic volume tercile mapping is accurate and complete."""
    low_th = 2154.0
    high_th = 4555.0
    test_volumes = pd.Series([500, 2154, 2155, 4555, 4556, 7000])
    labels = traffic_volume_to_class(test_volumes, low_th, high_th)
    
    expected = [
        "Low Congestion",
        "Low Congestion",
        "Moderate Congestion",
        "Moderate Congestion",
        "High Congestion",
        "High Congestion",
    ]
    assert list(labels) == expected, f"Labeling mismatch: got {list(labels)}, expected {expected}"


def test_model_loading_and_inference():
    """Verify that the trained pipeline loads and performs valid probabilistic inference."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    
    sample_input = pd.DataFrame([{
        "hour": 8,
        "day_of_week": 1,
        "month": 6,
        "temp": 293.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20,
        "holiday": "None",
        "weather_main": "Clear"
    }])
    
    pred = pipeline.predict(sample_input)
    assert len(pred) == 1, "Prediction output shape mismatch"
    assert pred[0] in ["Low Congestion", "Moderate Congestion", "High Congestion"], f"Invalid class: {pred[0]}"
    
    proba = pipeline.predict_proba(sample_input)[0]
    assert len(proba) == 3, f"Expected 3 class probabilities, got {len(proba)}"
    assert np.isclose(np.sum(proba), 1.0, atol=1e-4), "Probabilities do not sum to 1.0"


def test_unseen_categories_handling():
    """Verify that pipeline handles rare/unseen weather categories gracefully without crashing."""
    pipeline = joblib.load(MODEL_PATH)
    novel_input = pd.DataFrame([{
        "hour": 14,
        "day_of_week": 5,
        "month": 11,
        "temp": 270.0,
        "rain_1h": 12.0,
        "snow_1h": 0.0,
        "clouds_all": 95,
        "holiday": "Random Unseen Holiday Name",
        "weather_main": "Volcanic Ash"
    }])
    pred = pipeline.predict(novel_input)
    assert pred[0] in ["Low Congestion", "Moderate Congestion", "High Congestion"]


def test_model_metadata_consistency():
    """Verify that saved metadata JSON exists and holds accurate metrics."""
    assert METADATA_PATH.exists(), "model_metadata.json missing"
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    
    assert "accuracy" in meta, "Missing accuracy key in metadata"
    assert "confusion_matrix" in meta, "Missing confusion matrix in metadata"
    assert meta["accuracy"] >= 0.85, f"Model accuracy unexpectedly low: {meta['accuracy']}"
    assert meta["train_rows"] == 33730, f"Expected 33,730 train rows, got {meta['train_rows']}"
    assert meta["test_rows"] == 7229, f"Expected 7,229 test rows, got {meta['test_rows']}"


def test_visualizations_exist():
    """Verify that all training output charts were generated and non-empty."""
    required_charts = [
        OUTPUT_DIR / "category_distribution.png",
        OUTPUT_DIR / "traffic_by_hour.png",
        OUTPUT_DIR / "confusion_matrix.png",
    ]
    for chart in required_charts:
        assert chart.exists(), f"Chart file missing: {chart.name}"
        assert chart.stat().st_size > 5000, f"Chart file appears empty or corrupted: {chart.name}"


if __name__ == "__main__":
    print("Running TrafficSense AI test suite...")
    test_dataset_loading()
    print("[PASS] Dataset loading test passed")
    test_feature_engineering()
    print("[PASS] Feature engineering & leakage prevention test passed")
    test_quantile_labeling()
    print("[PASS] Quantile labeling test passed")
    test_model_loading_and_inference()
    print("[PASS] Model loading and inference test passed")
    test_unseen_categories_handling()
    print("[PASS] Unseen categorical handling test passed")
    test_model_metadata_consistency()
    print("[PASS] Metadata consistency test passed")
    test_visualizations_exist()
    print("[PASS] Visualization outputs test passed")
    print("\nALL 7 TESTS PASSED SUCCESSFULLY!")
