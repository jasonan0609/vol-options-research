
"""
Options cleaning: per-date transforms from raw -> clean.

Functions:
- filter_bad_quotes(df) -> df_filtered, drop_report
- enforce_monotonic_strikes(df) -> df_deduped, dedupe_report
- compute_derived(df, spot_price) -> df_with_derived
- clean_options_for_date(trade_date, underlying_symbol='SPY')

Writes clean file to data/clean/options/YYYY-MM-DD.parquet and a .meta.json audit file.

Notes:
- Uses cleaned underlying spot (close) from data/clean/underlying/{symbol}.parquet
- Pure transforms only; no network I/O.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.config import raw_options_date_path, clean_options_date_path, clean_underlying_path
from src.utils.trading_calendar import trading_days

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Canonical columns for clean options (one row per contract)
CANONICAL_COLS = [
    "date", "expiry", "strike", "cp",
    "bid", "ask", "mid",
    "volume", "open_interest", "implied_vol",
    "moneyness", "underlying"
]


def _empty_clean_df(trade_date: date, underlying: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=[c for c in CANONICAL_COLS if c != "date"])
    df["date"] = pd.to_datetime(trade_date).date()
    df["underlying"] = underlying
    return df[CANONICAL_COLS]


def _read_raw_options(trade_date: date) -> pd.DataFrame:
    p = raw_options_date_path(trade_date)
    if not p.exists():
        logger.warning("Raw options file missing for %s: %s", trade_date, p)
        return pd.DataFrame()  # empty
    try:
        df = pd.read_parquet(p)
        return df
    except Exception as e:
        logger.exception("Failed to read raw options parquet %s: %s", p, e)
        return pd.DataFrame()


def _read_spot_for_date(trade_date: date, underlying: str) -> float | None:
    """Read cleaned underlying close for that date. Return float spot or None if unavailable."""
    p = clean_underlying_path(underlying)
    if not p.exists():
        logger.warning("Clean underlying file not found: %s", p)
        return None
    df = pd.read_parquet(p)
    # index expected to be DatetimeIndex; find row for trade_date
    idx = pd.to_datetime(trade_date).normalize()
    if idx not in df.index:
        logger.warning("Underlying spot for %s not present in clean underlying file", trade_date)
        return None
    val = df.at[idx, "close"]
    try:
        return float(val)
    except Exception:
        return None


def filter_bad_quotes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rules:
      - bid <= 0 or ask <= 0
      - bid > ask
      - volume == 0 AND open_interest == 0

    Returns:
      (filtered_df, drop_report)
      drop_report = {"by_bid_le_zero": n, "by_ask_le_zero": n, "by_bid_gt_ask": n, "by_zero_vol_oi": n}
    """
    if df is None or df.empty:
        return df, {}

    initial = len(df)
    # ensure numeric types
    for col in ["bid", "ask", "volume", "openInterest", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # unify open_interest name
    if "openInterest" in df.columns and "open_interest" not in df.columns:
        df = df.rename(columns={"openInterest": "open_interest"})

    report = {}
    # the code uses df.get("bid", pd.Series()) instead of just df["bid"].
    #The Reason: This prevents the code from crashing if a column is missing. If the "bid" column doesn't exist, it returns an empty Series instead of an Error.
    #.fillna(0): Ensures that if there are empty cells (NaN), they are treated as 0 so the math operations (like <= 0) don't fail.
    mask_bid_le_zero = (df.get("bid", pd.Series()).fillna(0) <= 0)
    mask_ask_le_zero = (df.get("ask", pd.Series()).fillna(0) <= 0)
    mask_bid_gt_ask = (df.get("bid", pd.Series()) > df.get("ask", pd.Series()))
    mask_zero_vol_oi = ((df.get("volume", pd.Series()) == 0) & (df.get("open_interest", pd.Series()) == 0))

    # Calculate counts (ensure masks align with df index)
    report["by_bid_le_zero"] = int(mask_bid_le_zero.sum()) if len(mask_bid_le_zero) == len(df) else 0
    report["by_ask_le_zero"] = int(mask_ask_le_zero.sum()) if len(mask_ask_le_zero) == len(df) else 0
    report["by_bid_gt_ask"] = int(mask_bid_gt_ask.sum()) if len(mask_bid_gt_ask) == len(df) else 0
    report["by_zero_vol_oi"] = int(mask_zero_vol_oi.sum()) if len(mask_zero_vol_oi) == len(df) else 0

    # Combined mask of rows to drop
    drop_mask = mask_bid_le_zero | mask_ask_le_zero | mask_bid_gt_ask | mask_zero_vol_oi
    filtered = df.loc[~drop_mask].copy()
    report["kept"] = int(len(filtered))
    report["initial"] = int(initial)
    return filtered, report


def enforce_monotonic_strikes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        return df, {}

    # 1. Standardize column names
    if "openInterest" in df.columns and "open_interest" not in df.columns:
        df = df.rename(columns={"openInterest": "open_interest"})

    before = len(df)

    # 2. Clean data at scale (vectorized)
    df = df.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])

    # Normalize cp column if present
    if "cp" in df.columns:
        df["cp"] = df["cp"].astype(str).str.upper().str.strip()
    else:
        df["cp"] = ""  # safe default to allow grouping

    # 3. Handle Open Interest logic
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
        df = df.sort_values(
            ["expiry", "cp", "strike", "open_interest"],
            ascending=[True, True, True, False],
        )
    else:
        df = df.sort_values(["expiry", "cp", "strike"], ascending=[True, True, True])

    # 4. Deduplicate (keep first row per (expiry, cp, strike) — highest OI wins)
    out = df.drop_duplicates(subset=["expiry", "cp", "strike"], keep="first")

    # 5. Final ordering and report
    out = out.sort_values(["expiry", "cp", "strike"], ascending=[True, True, True]).reset_index(drop=True)
    after = len(out)
    report = {"before": int(before), "after": int(after), "deduped": int(before - after)}

    return out, report




def compute_derived(df: pd.DataFrame, spot_price: float | None, underlying: str) -> pd.DataFrame:
    """
    Compute mid and moneyness.
    - mid = (bid + ask)/2 (if both present)
    - moneyness = strike / spot_price  (None if spot_price is None)
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    # ensure numeric
    for col in ["bid", "ask", "strike"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # mid
    if "mid" not in out.columns:
        out["mid"] = out[["bid", "ask"]].mean(axis=1)

    # moneyness
    if spot_price is None:
        out["moneyness"] = pd.NA
    else:
        out["moneyness"] = out["strike"] / float(spot_price)

    # standardize implied vol name if present
    if "impliedVolatility" in out.columns and "implied_vol" not in out.columns:
        out["implied_vol"] = pd.to_numeric(out["impliedVolatility"], errors="coerce")
    elif "implied_vol" in out.columns:
        out["implied_vol"] = pd.to_numeric(out["implied_vol"], errors="coerce")
    else:
        out["implied_vol"] = pd.NA

    # ensure volume/open_interest types
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")
    if "open_interest" in out.columns:
        out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce").astype("Int64")

    # ensure underlying column
    out["underlying"] = underlying

    # keep canonical columns order
    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[CANONICAL_COLS]


def save_clean_options(df: pd.DataFrame, trade_date: date) -> None:
    out = clean_options_date_path(trade_date)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    meta = {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))
    logger.info("Wrote cleaned options for %s rows=%d", trade_date, df.shape[0])


def clean_options_for_date(trade_date: date, underlying: str = "SPY") -> dict:
    """
    Clean options for one trade_date.
    Returns a report dict with counts and any warnings.
    """
    report = {"date": str(trade_date), "underlying": underlying}
    raw = _read_raw_options(trade_date)
    if raw is None or raw.empty:
        logger.info("No raw options for %s — writing empty clean file", trade_date)
        empty = _empty_clean_df(trade_date, underlying)
        save_clean_options(empty, trade_date)
        report["status"] = "empty_raw"
        report["rows_out"] = 0
        return report

    # Step 1: filter bad quotes
    filtered, drop_report = filter_bad_quotes(raw)
    report.update(drop_report)

    # Step 2: enforce monotonic strikes & dedupe
    deduped, dedupe_report = enforce_monotonic_strikes(filtered)
    report.update(dedupe_report)

    # Step 3: compute derived fields
    spot = _read_spot_for_date(trade_date, underlying)
    derived = compute_derived(deduped, spot, underlying)

    # Step 4: final housekeeping: ensure date column, reorder, and write
    if "date" not in derived.columns:
        derived["date"] = pd.to_datetime(trade_date).date()
    derived = derived[CANONICAL_COLS]

    save_clean_options(derived, trade_date)
    report["status"] = "ok"
    report["rows_out"] = int(derived.shape[0])
    return report


# Runner for ranges / CLI (conservative default: only last trading day)
def _dates_from_args(args) -> list[date]:
    if getattr(args, "date", None):
        return [pd.to_datetime(args.date).date()]
    if getattr(args, "start", None) and getattr(args, "end", None):
        td = trading_days(pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
        return [d.date() for d in td]
    # default: last trading day only
    today = pd.Timestamp.today().normalize().date()
    td = trading_days(today, today)
    return [d.date() for d in td]


def main_cli():
    import argparse
    p = argparse.ArgumentParser(description="Clean options snapshots (one file per trading day).")
    p.add_argument("--date", type=str, help="Single date YYYY-MM-DD")
    p.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    p.add_argument("--symbol", type=str, default="SPY", help="Underlying symbol")
    args = p.parse_args()

    dates = _dates_from_args(args)
    out_reports = []
    for d in dates:
        try:
            r = clean_options_for_date(d, underlying=args.symbol)
            logger.info("Clean report: %s", r)
            out_reports.append(r)
        except Exception as e:
            logger.exception("Failed cleaning for %s: %s", d, e)
    # Optionally write a summary file
    summary = {"reports": out_reports}
    Path("data/clean/options/clean_summary.json").write_text(json.dumps(summary))
    logger.info("Cleaning complete for %d dates", len(out_reports))


if __name__ == "__main__":
    main_cli()
