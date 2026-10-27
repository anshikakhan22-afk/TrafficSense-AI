"""Load and prepare the UCI Metro Interstate Traffic Volume dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_CANDIDATES = [
    DATA_DIR / "Metro_Interstate_Traffic_Volume.csv",
    DATA_DIR / "Metro_Interstate_Traffic_Volume.csv.gz",
    DATA_DIR / "Metro_Interstate_Traffic_Volume.csv.zip",
]
TARGET_COLUMN = "traffic_volume"


def find_dataset() -> Path:
    """Return the first supported dataset file, or raise a helpful error."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    expected = "\n".join(f"  - {path.name}" for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Traffic dataset not found. Download it from the UCI repository and "
        f"place it in {DATA_DIR}. Expected one of:\n{expected}\n"
        "Dataset page: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
    )


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load data and validate the columns required by this project."""
    source = Path(path) if path else find_dataset()
    if source.suffix.lower() == ".zip":
        df = pd.read_csv(source, compression="zip")
    else:
        df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date_time", TARGET_COLUMN, "holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=["date_time", TARGET_COLUMN]).copy()
    df = df.sort_values("date_time").drop_duplicates().reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create prediction-time features; never include traffic_volume as input."""
    dt = pd.to_datetime(df["date_time"])
    features = pd.DataFrame(index=df.index)
    features["hour"] = dt.dt.hour
    features["day_of_week"] = dt.dt.dayofweek
    features["month"] = dt.dt.month
    features["holiday"] = df["holiday"].fillna("None").astype(str)
    features["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    features["rain_1h"] = pd.to_numeric(df["rain_1h"], errors="coerce")
    features["snow_1h"] = pd.to_numeric(df["snow_1h"], errors="coerce")
    features["clouds_all"] = pd.to_numeric(df["clouds_all"], errors="coerce")
    features["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)
    return features


def traffic_volume_to_class(values, low_threshold: float, high_threshold: float):
    """Map measured traffic volume to derived, volume-based categories."""
    import numpy as np
    return np.select(
        [values <= low_threshold, values <= high_threshold],
        ["Low Congestion", "Moderate Congestion"],
        default="High Congestion",
    )
