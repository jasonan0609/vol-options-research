"""
Options ingestion stub: one snapshot per trading day.

Saves raw files to:
    data/raw/options/YYYY-MM-DD.parquet

Contract:
- Raw file must contain (at minimum) the following columns:
  ['date', 'expiry', 'strike', 'cp', 'bid', 'ask', 'mid', 'volume', 'open_interest']
- This module does NOT perform cleaning. Cleaning lives in src/cleaning.
- Implement provider logic in download_options_for_date().

Run examples:
    # single date
    python -m src.ingestion.pull_options --date 2025-01-03

    # range
    python -m src.ingestion.pull_options --start 2025-01-01 --end 2025-01-10
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import List
import yfinance as yf

import pandas as pd

from src.utils.config import raw_options_date_path, UNDERLYING_SYMBOL
from src.utils.trading_calendar import trading_days



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# canonical columns for raw options snapshot (one row per contract)
CANONICAL_COLS = [
    "date",           # ingestion date
    "expiry",         # option expiry (date)
    "strike",         # strike price (float)
    "cp",             # 'C' or 'P'
    "bid",            # float
    "ask",            # float
    "mid",            # float (optional; can be computed later)
    "volume",         # int
    "open_interest",  # int
    "implied_vol",    # float
    "underlying",     # symbol (string)
]


def download_options_for_date(trade_date: date, symbol: str) -> pd.DataFrame:
    """
    Free-provider implementation using yfinance.

    Limitations:
      - yfinance provides the *current* option chain / quotes only.
      - This function will return a snapshot only when run for today's date.
      - For historical snapshots you must use a paid provider or stored snapshots.

    Behavior:
      - If trade_date != today, returns an empty canonical DataFrame (safe default).
      - If trade_date == today, downloads option chains across expiries and returns a DataFrame
        with columns: ['expiry','strike','cp','bid','ask','mid','volume','open_interest','underlying'].
    """

    today = pd.Timestamp.today().date()
    if trade_date != today:
        logger.warning(
            "yfinance only provides current option quotes. trade_date %s != today %s. "
            "Returning empty DataFrame. If you want to force a current snapshot tagged to this date, "
            "call with a force flag (not implemented here).",
            trade_date, today
        )
        return pd.DataFrame(columns=[c for c in CANONICAL_COLS if c != "date"])

    tk = yf.Ticker(symbol)
    expiries = tk.options  # list of expiry strings like '2025-01-17'
    if not expiries:
        logger.warning("No expiries returned by yfinance for %s", symbol)
        return pd.DataFrame(columns=[c for c in CANONICAL_COLS if c != "date"])

    parts = []
    for exp in expiries:
        try:
            oc = tk.option_chain(exp)
            # oc.calls and oc.puts are DataFrames with columns including:
            # ['contractSymbol','lastTradeDate','strike','lastPrice','bid','ask','change',
            #  'percentChange','volume','openInterest','impliedVolatility','inTheMoney']
            for df_side, cp in ((oc.calls, "C"), (oc.puts, "P")):
                if df_side is None or df_side.empty:
                    continue
                # Select relevant columns if present, otherwise fill with NaN
                df_side = df_side.reset_index(drop=True)
                # Ensure expiry column
                df_side["expiry"] = pd.to_datetime(exp).date()
                df_side["strike"] = pd.to_numeric(df_side.get("strike"), errors="coerce")
                # bid/ask may be missing; coerce to numeric
                df_side["bid"] = pd.to_numeric(df_side.get("bid"), errors="coerce")
                df_side["ask"] = pd.to_numeric(df_side.get("ask"), errors="coerce")
                # compute mid where possible
                df_side["mid"] = df_side[["bid", "ask"]].mean(axis=1)
                # implied volatility: check common names, coerce to numeric
                iv_col = None
                for candidate in ("impliedVolatility", "implied_volatility", "impliedVol", "impliedVols"):
                    if candidate in df_side.columns:
                        iv_col = candidate
                        break

                if iv_col is not None:
                    df_side["implied_vol"] = pd.to_numeric(df_side[iv_col], errors="coerce")
                else:
                    df_side["implied_vol"] = pd.NA
                # volume/open_interest: coerce to Int64
                df_side["volume"] = pd.to_numeric(df_side.get("volume"), errors="coerce").astype("Int64")
                oi_col = df_side.get("openInterest")
                if oi_col is None:
                    oi_col = df_side.get("open_interest")
                df_side["open_interest"] = pd.to_numeric(oi_col, errors="coerce").astype("Int64")
                df_side["cp"] = cp
                df_side["underlying"] = symbol

                # Keep only the canonical fields we need
                keep = ["expiry", "strike", "cp", "bid", "ask", "mid", "volume", "open_interest", "implied_vol", "underlying"]
                # If any of the keep columns are missing in df_side, add them as NA
                for k in keep:
                    if k not in df_side.columns:
                        df_side[k] = pd.NA

                parts.append(df_side[keep].copy())
        except Exception as e:
            logger.exception("Failed to fetch option_chain for expiry %s (%s): %s", exp, symbol, e)
            # Continue to next expiry

    if not parts:
        return pd.DataFrame(columns=[c for c in CANONICAL_COLS if c != "date"])

    out = pd.concat(parts, ignore_index=True)

    # Ensure types: strike float, bid/ask/mid float, volume/open_interest Int64
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    for col in ["bid", "ask", "mid"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")
    out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce").astype("Int64")
    out["implied_vol"] = pd.to_numeric(out["implied_vol"], errors="coerce")

    # Deduplicate if necessary (keep first)
    out = out.drop_duplicates(subset=["expiry", "strike", "cp"], keep="first").reset_index(drop=True)

    return out



def save_options_snapshot(df: pd.DataFrame, trade_date: date) -> None:
    """Save snapshot as parquet and write a small meta JSON for audit."""
    out = raw_options_date_path(trade_date)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ensure canonical columns exist (date will be added)
    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # add ingestion date column (date-only)
    df["date"] = pd.to_datetime(trade_date).date()
    # reorder columns to canonical order
    df = df[CANONICAL_COLS]

    # write parquet
    df.to_parquet(out)
    # write meta
    meta = {"rows": int(df.shape[0]), "cols": list(df.columns)}
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))
    logger.info("Wrote options snapshot: %s (rows=%d)", out, df.shape[0])


def pull_for_date(trade_date: date, symbol: str = UNDERLYING_SYMBOL) -> None:
    """Download and save one-date options snapshot."""
    logger.info("Pulling options for %s on %s", symbol, trade_date)
    df = download_options_for_date(trade_date, symbol)
    save_options_snapshot(df, trade_date)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull options snapshots (one file per trading day).")
    p.add_argument("--date", type=str, help="Single date YYYY-MM-DD")
    p.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    p.add_argument("--symbol", type=str, default=UNDERLYING_SYMBOL, help="Underlying symbol")
    return p.parse_args()


def _dates_from_args(args) -> List[date]:
    if args.date:
        return [pd.to_datetime(args.date).date()]
    if args.start and args.end:
        td = trading_days(pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
        return [d.date() for d in td]
    # default: last trading day only (most conservative)
    today = pd.Timestamp.today().normalize().date()
    td = trading_days(today, today)
    return [d.date() for d in td]


def main():
    args = parse_args()
    dates = _dates_from_args(args)
    symbol = args.symbol

    for d in dates:
        try:
            pull_for_date(d, symbol)
        except Exception as e:
            logger.exception("Failed to pull options for %s on %s: %s", symbol, d, e)


if __name__ == "__main__":
    main()
