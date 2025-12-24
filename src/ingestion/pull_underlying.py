"""
Ingestion stub for underlying (SPY, VIX).

What it does:
- Demonstrates how to fetch OHLCV for a symbol over a date range,
  then save the raw file to the configured raw path.
- This is intentionally a stub: plug your provider (yfinance/polygon) in the
  fetch_underlying implementation.

Run:
    python -m src.ingestion.pull_underlying
"""

from datetime import timedelta
import logging
from typing import Optional

import pandas as pd

from src.utils.config import UNDERLYING_SYMBOL, raw_underlying_path, FILE_FORMAT, TIMEZONE
from src.utils.trading_calendar import trading_days


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")



def download_with_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Provider implementation using yfinance. Ensures we get raw OHLCV and an 'Adj Close' column.
    Requires: pip install yfinance
    """
    import yfinance as yf

    # Request raw OHLCV and explicit Adj Close by forcing auto_adjust=False
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

    # If yfinance returns a MultiIndex columns (rare), flatten them to single strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build a mapping from whatever came back to our canonical names
    cols_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc == "open":
            cols_map[col] = "open"
        elif lc == "high":
            cols_map[col] = "high"
        elif lc == "low":
            cols_map[col] = "low"
        elif lc == "close":
            cols_map[col] = "close"
        elif lc in ("adj close", "adj_close", "adjusted close"):
            cols_map[col] = "adj_close"
        elif lc == "volume":
            cols_map[col] = "volume"

    df = df.rename(columns=cols_map)
    return df


def fetch_underlying(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch underlying OHLCV between start_date and end_date (inclusive).
    Normalizes index and returns columns:
      ['open','high','low','close','adj_close','volume']
    """
    # Use yfinance provider by default
    df = download_with_yfinance(symbol, start_date, end_date)

    # Normalize index: ensure date-only, timezone-naive
    if not df.empty:
        df.index = pd.to_datetime(df.index).normalize()

        # Ensure canonical columns exist; fill or raise if critical missing
        required = ["open", "high", "low", "close", "volume"]
        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required columns from provider: {missing_required}")

        # If adj_close missing, fallback to close (log warning)
        if "adj_close" not in df.columns:
            logger.warning("adj_close missing from provider; filling adj_close = close")
            df["adj_close"] = df["close"]

        # Keep canonical order and types
        df = df[["open", "high", "low", "close", "adj_close", "volume"]]

        # Ensure numeric dtypes where appropriate
        for col in ["open", "high", "low", "close", "adj_close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    else:
        logger.warning("download returned empty DataFrame for %s", symbol)

    return df



def save_raw_underlying(df: pd.DataFrame, symbol: str) -> None:
    """
    Save raw underlying to configured path. Creates parent dir if missing.
    """
    out = raw_underlying_path(symbol)
    out.parent.mkdir(parents=True, exist_ok=True) # safety step. If the folder data/raw/ doesn't exist yet, Python will crash when trying to save a file there.
    # Write parquet; preserve index (date index)
    df.to_parquet(out)
    logger.info("Wrote raw underlying to %s", out)


def main(symbol: Optional[str] = None):
    symbol = symbol or UNDERLYING_SYMBOL

    # Choose date range: last 252 trading days up to today (deterministic via calendar)
    end = pd.Timestamp.today().normalize()
    # approximate lookback window; trading_days will prune weekends/holidays
    start_guess = end - timedelta(days=365)  # ~1 year back to ensure 252 trading days present
    td = trading_days(start_guess.date(), end.date())
    if len(td) == 0:
        raise RuntimeError("Trading calendar returned no dates for given range")

    start = td[0].date().isoformat()
    end = td[-1].date().isoformat()

    logger.info("Fetching %s from %s to %s", symbol, start, end)
    df = fetch_underlying(symbol, start, end)

    # Save raw output even if empty so the pipeline knows a pull was attempted.
    save_raw_underlying(df, symbol)


if __name__ == "__main__":
    main()
