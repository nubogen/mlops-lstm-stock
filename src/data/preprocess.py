# src/data/preprocess.py
"""
Preprocess raw time-series parquet into supervised numpy arrays for LSTM,
with Option C normalization: log-transform + MinMaxScaler fitted on training portion.
Saves:
 - data/processed/<symbol>_X.npy       (full X, for backward compat)
 - data/processed/<symbol>_y.npy       (full y, for backward compat)
 - data/processed/<symbol>_X_train.npy
 - data/processed/<symbol>_y_train.npy
 - data/processed/<symbol>_X_val.npy
 - data/processed/<symbol>_y_val.npy
 - data/processed/<symbol>_scaler.pkl  (saved scaler for inference)
Usage:
    python -m src.data.preprocess --symbol BTC-USD --window 60 --val_split 0.1
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

EPS = 1e-9  # small constant to avoid log(0)


# ----------------------
# IO / Loading
# ----------------------
def load_raw_data(symbol: str) -> pd.DataFrame:
    raw_path = Path("data/raw") / f"{symbol.lower()}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    df = pd.read_parquet(raw_path)
    return df


# ----------------------
# Basic cleaning
# ----------------------
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()

    # Ensure datetime index (we will treat it as time series)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("Could not convert index to DatetimeIndex. Ensure parquet has a date index or a Date column.")

    df = df.sort_index()
    return df


# ----------------------
# Normalization helpers
# ----------------------
def log_transform(series: pd.Series) -> pd.Series:
    # Apply stable log transform
    return np.log(series.astype(float) + EPS)


def fit_scaler_on_train(series_log: np.ndarray, n_train: int) -> MinMaxScaler:
    scaler = MinMaxScaler()
    # Fit only on training portion to avoid leakage
    train_slice = series_log[:n_train].reshape(-1, 1)
    scaler.fit(train_slice)
    return scaler


# ----------------------
# Sequence creation
# ----------------------
def create_sequences_from_array(arr: np.ndarray, window_size: int):
    """
    arr: 1D numpy array of transformed values (already scaled)
    returns X (num_samples, window_size, 1), y (num_samples, 1)
    """
    X, y = [], []
    for i in range(window_size, len(arr)):
        seq_x = arr[i - window_size:i]
        seq_y = arr[i]
        X.append(seq_x.reshape(window_size, 1))
        y.append(np.array([seq_y]))
    if len(X) == 0:
        return np.empty((0, window_size, 1)), np.empty((0, 1))
    return np.array(X), np.array(y)


# ----------------------
# Save helpers
# ----------------------
def save_npy(array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_scaler(scaler: MinMaxScaler, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


# ----------------------
# Main flow
# ----------------------
def preprocess(symbol: str, window: int = 60, val_split: float = 0.1):
    # Load and clean
    df = load_raw_data(symbol)
    df = basic_cleaning(df)

    if "Close" not in df.columns:
        raise KeyError("Expected 'Close' column in dataframe")

    close = df["Close"].values.astype(float)  # numeric np array

    # Compute sizes for train/val on raw timeline (no sequences yet)
    n_total = len(close)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Not enough data for the given val_split")

    # 1) Log transform (on entire series, but scaler fit only on training slice)
    close_log = np.log(close + EPS)

    # 2) Fit scaler on training portion (avoids leakage)
    scaler = fit_scaler_on_train(close_log, n_train)

    # 3) Transform the whole series using the scaler fitted on training
    close_scaled = scaler.transform(close_log.reshape(-1, 1)).reshape(-1)

    # 4) Create sequences from scaled series
    X_all, y_all = create_sequences_from_array(close_scaled, window)

    # 5) Split sequences into train/val by using the last n_val sequences
    # number of sequences = len(close_scaled) - window
    n_seq = X_all.shape[0]
    n_val_seq = max(1, int(n_seq * val_split))
    n_train_seq = n_seq - n_val_seq

    X_train = X_all[:n_train_seq]
    y_train = y_all[:n_train_seq]
    X_val = X_all[n_train_seq:]
    y_val = y_all[n_train_seq:]

    # 6) Save artifacts (full and train/val)
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    symbol_base = symbol.lower()
    save_npy(X_all, proc_dir / f"{symbol_base}_X.npy")
    save_npy(y_all, proc_dir / f"{symbol_base}_y.npy")

    save_npy(X_train, proc_dir / f"{symbol_base}_X_train.npy")
    save_npy(y_train, proc_dir / f"{symbol_base}_y_train.npy")
    save_npy(X_val, proc_dir / f"{symbol_base}_X_val.npy")
    save_npy(y_val, proc_dir / f"{symbol_base}_y_val.npy")

    # save scaler for future inverse_transform during inference
    save_scaler(scaler, proc_dir / f"{symbol_base}_scaler.pkl")

    # Print summary
    print(f"Raw len: {n_total} | n_train: {n_train} | n_val: {n_val}")
    print(f"Sequences total: {n_seq} | seq_train: {n_train_seq} | seq_val: {n_val_seq}")
    print(f"Saved processed files to {proc_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw price data (log + MinMax scaling)")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol (e.g., BTC-USD)")
    parser.add_argument("--window", type=int, default=60, help="Window size for sequences")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation (chronological tail)")

    args = parser.parse_args()
    preprocess(args.symbol, window=args.window, val_split=args.val_split)
