# src/models/model.py
"""
PyTorch LSTM model builder for time-series prediction.
Provides a compact, testable LSTMRegressor class and a helper to create the model and optimizer.
"""

from typing import Tuple
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        """
        Args:
            input_size: number of features per timestep (e.g., 1)
            hidden_size: LSTM hidden units
            num_layers: number of stacked LSTM layers
            dropout: dropout between LSTM layers (only used when num_layers>1)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM: batch_first=True uses input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

        # simple regressor head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        x: tensor shape (batch, seq_len, features)
        returns: tensor shape (batch, 1)
        """
        # LSTM output: output, (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)  # out shape (batch, seq_len, hidden_size)
        # take last timestep's output
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.head(last)


def build_model(input_shape: Tuple[int, int], hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
    """
    Convenience builder.

    Args:
      input_shape: (timesteps, features)
    Returns:
      model (nn.Module), optimizer (torch.optim.Optimizer)
    """
    timesteps, features = input_shape
    model = LSTMRegressor(input_size=features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer
