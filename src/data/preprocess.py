# src/data/preprocess.py
"""
Preprocess raw time-series parquet into supervised numpy arrays for LSTM.
Usage:
    python src/data/preprocess.py --symbol BTC-USD --window 60
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_data(symbol: str) -> pd.DataFrame:
    raw_path = Path("data/raw") / f"{symbol.lower()}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_parquet(raw_path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - drop missing rows
    - ensure datetime index and sorted order
    - ensure consistent column names (Open/High/Low/Close/Volume)
    """
    df = df.dropna().copy()

    # If Date is a column, set it as index
    if not isinstance(df.index, pd.DatetimeIndex):
        # try common column names
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            # if index is string-like, try converting
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("Could not convert index to DatetimeIndex. Ensure parquet has a date index or a Date column.")

    # Ensure sorted by time (oldest -> newest)
    df = df.sort_index()

    return df


def create_sequences(df: pd.DataFrame, window_size: int = 60, target_col: str = "Close"):
    """
    Build X, y for LSTM:
    - X shape: (num_samples, window_size)
    - y shape: (num_samples,)
    Anti-leakage: X contains only past values relative to y.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

    data = df[target_col].values.astype(float)  # ensure numeric numpy array
    X, y = [], []

    for i in range(window_size, len(data)):
        seq_x = data[i - window_size:i]  # past window
        seq_y = data[i]  # next-step target (no future in X)
        X.append(seq_x)
        y.append(seq_y)

    if len(X) == 0:
        return np.empty((0, window_size)), np.empty((0,))

    return np.array(X), np.array(y)


def save_processed_data(X: np.ndarray, y: np.ndarray, symbol: str):
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    X_path = processed_path / f"{symbol.lower()}_X.npy"
    y_path = processed_path / f"{symbol.lower()}_y.npy"

    np.save(X_path, X)
    np.save(y_path, y)

    print(f"Saved X to {X_path} ({X.shape})")
    print(f"Saved y to {y_path} ({y.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw price data")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol (e.g., BTC-USD)")
    parser.add_argument("--window", type=int, default=60, help="Window size (number of timesteps for X)")

    args = parser.parse_args()

    df_raw = load_raw_data(args.symbol)
    df_clean = basic_cleaning(df_raw)

    print("Raw shape:", df_raw.shape)
    print("Clean shape:", df_clean.shape)

    X, y = create_sequences(df_clean, window_size=args.window, target_col="Close")
    print("Sequences created: X shape:", X.shape, "y shape:", y.shape)

    save_processed_data(X, y, args.symbol)
