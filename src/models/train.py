# src/models/train.py
"""
Training engine for LSTMRegressor (PyTorch) with MLflow tracking (hyperparams, metrics, model, scaler).
This version prefers explicit train/val numpy files produced by preprocess.py.
Run as module:
    python -m src.models.train --symbol BTC-USD --epochs 2
"""

from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch
import joblib

from src.models.model import build_model


# ----------------------
# Data utilities (prefer explicit train/val files)
# ----------------------
def load_processed_explicit(symbol: str):
    proc = Path("data/processed")
    base = symbol.lower()

    X_train_path = proc / f"{base}_X_train.npy"
    y_train_path = proc / f"{base}_y_train.npy"
    X_val_path = proc / f"{base}_X_val.npy"
    y_val_path = proc / f"{base}_y_val.npy"

    # scaler path (may not exist in very old runs)
    scaler_path = proc / f"{base}_scaler.pkl"

    if X_train_path.exists() and X_val_path.exists():
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_val = np.load(X_val_path)
        y_val = np.load(y_val_path)
        return (X_train, y_train, X_val, y_val, scaler_path if scaler_path.exists() else None)

    # fallback to combined files (older behaviour)
    X_path = proc / f"{base}_X.npy"
    y_path = proc / f"{base}_y.npy"
    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        # fallback split: last val_split fraction -> val
        # in this function we don't know val_split; caller will do a quick split by last 10%
        n = len(X)
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:]
        y_val = y[n_train:]
        return (X_train, y_train, X_val, y_val, scaler_path if scaler_path.exists() else None)

    raise FileNotFoundError("Processed numpy files not found. Run src.data.preprocess first.")


def make_dataloaders_from_arrays(X_train, y_train, X_val, y_val, batch_size=32):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_v, y_v)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    return train_dl, val_dl


# ----------------------
# Training / Validation
# ----------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = torch.nn.functional.mse_loss(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = torch.nn.functional.mse_loss(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


# ----------------------
# Model persistence helpers
# ----------------------
def save_model_local(model, symbol: str):
    out_path = Path("models")
    out_path.mkdir(exist_ok=True)
    file = out_path / f"{symbol.lower()}_lstm.pth"
    torch.save(model.state_dict(), file)
    return file


# ----------------------
# Main training with MLflow (logs scaler if available)
# ----------------------
def main(args):
    # MLflow local tracking folder inside repo
    mlflow.set_tracking_uri("file:mlruns")
    experiment_name = args.experiment or "mlops-lstm-stock"
    mlflow.set_experiment(experiment_name)

    # Load processed explicit files (or fallback)
    X_train, y_train, X_val, y_val, scaler_path = load_processed_explicit(args.symbol)
    train_dl, val_dl = make_dataloaders_from_arrays(X_train, y_train, X_val, y_val, batch_size=args.batch)

    # Build model + optimizer
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, optimizer = build_model(input_shape, hidden_size=args.hidden, dropout=args.dropout)

    # Device (use MPS on Apple Silicon if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Training on:", device)
    model.to(device)

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("symbol", args.symbol)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("hidden_size", args.hidden)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("train_samples", int(len(X_train)))
        mlflow.log_param("val_samples", int(len(X_val)))

        # Optional: log requirements
        if Path("requirements.txt").exists():
            mlflow.log_artifact("requirements.txt")

        # Epoch loop
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_dl, optimizer, device)
            val_loss = validate(model, val_dl, device)

            # Print to console
            print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # Log metrics to MLflow (per-epoch)
            mlflow.log_metric("train_loss", float(train_loss), step=epoch + 1)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch + 1)

        # Save the final model locally and log it to MLflow
        local_model_path = save_model_local(model, args.symbol)
        model_cpu = model.to(torch.device("cpu"))
        mlflow.pytorch.log_model(model_cpu, artifact_path="model")
        mlflow.log_artifact(str(local_model_path))

        # If scaler exists, log it as well so the run contains the scaler artifact
        if scaler_path is not None and Path(scaler_path).exists():
            mlflow.log_artifact(str(scaler_path))
            print(f"Logged scaler artifact: {scaler_path}")

        print(f"Saved model to {local_model_path}")
        print("MLflow run id:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTMRegressor with MLflow tracking (PyTorch), using explicit train/val files if available")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name (optional)")

    args = parser.parse_args()
    main(args)
