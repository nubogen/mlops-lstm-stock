# src/data/fetch_data.py
"""
Download historical price data using yfinance and save as parquet.
Usage:
    python src/data/fetch_data.py --symbol BTC-USD --start 2015-01-01 --end 2023-01-01
"""

import argparse
from pathlib import Path
import yfinance as yf
import pandas as pd


def fetch_data(symbol: str, start: str, end: str | None = None):
    """
    Download OHLCV data for `symbol` from Yahoo Finance and save to data/raw/<symbol>.parquet
    """
    print(f"Downloading {symbol} data from {start} to {end or 'latest'} ...")
    df: pd.DataFrame = yf.download(symbol, start=start, end=end, progress=True)

    output_path = Path("data/raw") / f"{symbol.lower()}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (requires pyarrow or fastparquet)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock/crypto data using yfinance")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol (e.g., BTC-USD)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    fetch_data(args.symbol, args.start, args.end)
