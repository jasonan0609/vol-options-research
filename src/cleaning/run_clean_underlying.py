"""
Run end-to-end cleaning for the underlying (SPY).

Usage:
    python -m src.cleaning.run_clean_underlying
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.config import UNDERLYING_SYMBOL, raw_underlying_path, clean_underlying_path
from src.cleaning.clean_underlying import reindex_to_calendar, apply_corporate_actions, validate_provider_adj_close
from src.utils.trading_calendar import trading_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_raw(symbol: str) -> pd.DataFrame:
    p = raw_underlying_path(symbol)
    if not p.exists():
        raise FileNotFoundError(f"Raw underlying file not found: {p}")
    df = pd.read_parquet(p)
    logger.info("Loaded raw underlying %s rows=%d cols=%s from %s", symbol, df.shape[0], list(df.columns), p)
    return df


def write_clean(df: pd.DataFrame, symbol: str) -> None:
    out = clean_underlying_path(symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    meta = {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "min_date": str(df.index.min().date()),
        "max_date": str(df.index.max().date()),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))
    logger.info("Wrote clean underlying to %s (rows=%d)", out, df.shape[0])


def infer_bounds_from_raw(df: pd.DataFrame):
    # Ensure index is datelike
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to coerce using 'date' column or index
        if "date" in df.columns:
            df2 = df.set_index(pd.to_datetime(df["date"]).dt.normalize())
        else:
            df2 = df.copy()
            df2.index = pd.to_datetime(df2.index).normalize()
    else:
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()

    if df2.index.empty:
        raise ValueError("Raw dataframe index empty; cannot infer date bounds.")
    start = df2.index.min().date()
    end = df2.index.max().date()
    return start, end


def run(symbol: str = UNDERLYING_SYMBOL):
    # Load raw
    df_raw = load_raw(symbol)

    # Quick provider-adj validator (non-blocking)
    try:
        report = validate_provider_adj_close(df_raw)
        logger.info("Provider adj_close present=%s mismatches=%d", report.get("present"), report.get("n_mismatches"))
    except Exception:
        logger.exception("validate_provider_adj_close failed (non-fatal)")

    # Infer bounds
    start, end = infer_bounds_from_raw(df_raw)
    logger.info("Inferred bounds from raw: %s -> %s", start, end)

    # Strict reindex to calendar (will raise if missing price rows)
    df_reindexed = reindex_to_calendar(df_raw, start, end)
    logger.info("Reindexed to calendar: rows=%d", df_reindexed.shape[0])

    # Apply corporate actions (splits only here; we trust provider adj_close)
    df_adjusted = apply_corporate_actions(df_reindexed, splits=None, dividends=None)
    logger.info("Applied corporate actions (splits None): adj_close preserved if provided")

    # Final checks
    if df_adjusted.index.has_duplicates:
        raise RuntimeError("Cleaned underlying index has duplicates")
    if not df_adjusted.index.is_monotonic_increasing:
        raise RuntimeError("Cleaned underlying index not monotonic increasing")

    # Write clean output
    write_clean(df_adjusted, symbol)

    logger.info("Underlying cleaning complete for %s: %s -> %s", symbol, start, end)


if __name__ == "__main__":
    run()
