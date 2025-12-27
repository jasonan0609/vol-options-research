"""
Cleaning utilities for underlying time series.

Functions:
- reindex_to_calendar(df, start_date, end_date)
- apply_corporate_actions(df, splits=None, dividends=None)

Notes:
- These are pure transforms (no network I/O).
- They are intentionally strict: missing data after reindex -> raise.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import logging
import pandas as pd

from src.utils.trading_calendar import trading_days

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CANONICAL_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame that contains at least the canonical columns (may be NA)."""
    out = df.copy()
    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[CANONICAL_COLS]


def reindex_to_calendar(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Normalize underlying df and reindex to trading calendar.

    Parameters
    ----------
    df : pd.DataFrame
        Raw underlying. Can be indexed by date or contain a date column as index-like.
    start_date, end_date : datetime.date or parseable
        Inclusive bounds passed to trading_days().

    Returns
    -------
    pd.DataFrame
        Reindexed DataFrame with index = DatetimeIndex of trading days (midnight naive)
        and columns = ['open','high','low','close','adj_close','volume'].

    Raises
    ------
    ValueError if any required price column contains NaNs after reindexing.
    """
    if df is None:
        raise ValueError("df is None")

    # Copy to avoid mutating input
    df2 = df.copy()

    # If there is a date column and it's not the index, try to set it
    if not isinstance(df2.index, pd.DatetimeIndex):
        # common patterns: 'date' column
        if "date" in df2.columns:
            df2 = df2.set_index(pd.to_datetime(df2["date"]).dt.normalize())
            df2 = df2.drop(columns=["date"], errors="ignore")
        else:
            # attempt to coerce index
            try:
                df2.index = pd.to_datetime(df2.index).normalize()
            except Exception:
                raise ValueError("Unable to coerce DataFrame index to DatetimeIndex. Provide date index or a 'date' column.")

    # Normalize index to date-only (midnight)
    df2.index = pd.to_datetime(df2.index).normalize()

    # Build canonical frame (adds missing canonical columns as NA)
    df2 = _ensure_canonical_columns(df2)

    # Build authoritative trading-days index
    td = trading_days(start_date, end_date)
    if len(td) == 0:
        raise ValueError(f"trading_days returned empty index for {start_date} - {end_date}")

    # Reindex (this will introduce NaNs where data missing)
    df_reindexed = df2.reindex(td)

    # Now assert no NaNs in price columns (open/high/low/close). adj_close may be NA if not provided.
    price_cols = ["open", "high", "low", "close"]
    missing_report = {}
    for col in price_cols:
        na_mask = df_reindexed[col].isna()
        if na_mask.any():
            missing_dates = df_reindexed.index[na_mask].tolist()
            missing_report[col] = missing_dates

    if missing_report:
        # format a concise message
        lines = []
        for col, dates in missing_report.items():
            # show up to first 8 offending dates
            sample = ", ".join(d.isoformat() for d in dates[:8])
            lines.append(f"{col}: {len(dates)} missing (examples: {sample}{'...' if len(dates)>8 else ''})")
        msg = "Underlying reindexing produced missing price data:\n" + "\n".join(lines)
        raise ValueError(msg)

    # Ensure monotonic increasing index and no duplicates
    if not df_reindexed.index.is_monotonic_increasing:
        raise ValueError("Reindexed underlying index is not monotonic increasing.")
    if df_reindexed.index.has_duplicates:
        raise ValueError("Reindexed underlying index contains duplicates.")

    # Ensure dtypes: numeric for prices, Int64 for volume when possible
    for col in ["open", "high", "low", "close", "adj_close"]:
        df_reindexed[col] = pd.to_numeric(df_reindexed[col], errors="coerce")
    if "volume" in df_reindexed.columns:
        try:
            df_reindexed["volume"] = pd.to_numeric(df_reindexed["volume"], errors="coerce").astype("Int64")
        except Exception:
            # keep as-is if conversion fails
            logger.warning("Failed to coerce 'volume' to Int64; keeping original dtype.")

    return df_reindexed


def apply_corporate_actions(df: pd.DataFrame, splits: Optional[pd.DataFrame] = None, dividends: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Apply corporate actions (splits primarily) and produce adj_close.

    Parameters
    ----------
    df : pd.DataFrame
        Underlying series indexed by date (DatetimeIndex), expected to have 'close' column.
    splits : pd.DataFrame, optional
        DataFrame with columns ['date','ratio'] where 'ratio' is the split ratio expressed
        as new_shares/old_shares (e.g., 2.0 for a 2-for-1 split). 'date' is the effective date.
    dividends : pd.DataFrame, optional
        DataFrame with columns ['date','dividend'] representing cash dividends on the given date.
        NOTE: dividend adjustment is not implemented fully here; function will warn and skip dividend adjustment.
    Returns
    -------
    pd.DataFrame
        Copy of df with columns adjusted for splits and 'adj_close' populated (if possible).

    Notes
    -----
    - If df already contains a non-null 'adj_close' column, this function will log that and leave adj_close
      in place (it will still apply splits if splits provided to check consistency).
    - Splits are applied by computing a backward cumulative factor so historical prices are adjusted downward
      to be consistent with current share counts.
    - Dividend adjustment requires careful total-return accounting; not implemented by default.
    """
    if df is None or df.empty:
        raise ValueError("Input df is empty or None")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input df must be indexed by a pandas.DatetimeIndex")

    out = df.copy()

    # Ensure close exists
    if "close" not in out.columns:
        raise ValueError("Input df must contain 'close' column to compute adjustments")

    # If adj_close exists and is non-empty, prefer provider adj_close unless splits provided
    has_provider_adj = ("adj_close" in out.columns) and (out["adj_close"].notna().any())

    # --- Apply splits if provided ---
    if splits is not None:
        # Expect splits DataFrame with columns ['date','ratio'] where ratio is >0
        sp = splits.copy()
        if "date" not in sp.columns or "ratio" not in sp.columns:
            raise ValueError("splits DataFrame must contain 'date' and 'ratio' columns")
        sp["date"] = pd.to_datetime(sp["date"]).dt.normalize()
        sp = sp.set_index("date").sort_index()

        # Create a Series of ratios indexed by date (only split dates)
        ratio_series = sp["ratio"].astype(float)

        # For each historical date t, compute factor = product of ratios for dates > t
        # That factor tells how many current shares correspond to 1 share on date t.
        # We'll compute cumulative product in reverse order.
        # Build a DataFrame aligning all trading dates to check factors
        all_dates = out.index
        # Construct a series of factor per date: start with ones
        factor = pd.Series(1.0, index=all_dates)
        # For each split_date, set factor at that date = ratio
        # Then compute cumulative product forward (from oldest to newest) and take reciprocal for backward adjustment.
        for sd, r in ratio_series.items():
            # only apply split if split date exists in index; if not, warn and apply at nearest earlier date
            if sd in factor.index:
                factor.loc[sd] = float(r)
            else:
                # find the next trading date after sd; if none, apply at last index
                # but better to warn — splits not aligned to trading calendar are applied at the nearest next trading day
                next_idx = factor.index[factor.index.get_loc(sd, method="bfill")] if sd < factor.index[-1] else factor.index[-1]
                logger.warning("Split date %s not in index; applying ratio %s at nearest date %s", sd.date(), r, next_idx.date())
                factor.loc[next_idx] = float(r)

        # Now cumulative product forward (oldest -> newest)
        cumprod = factor.cumprod()
        # To adjust historical prices to current (i.e., reflect today's share count), divide by cumprod
        # Example: if there was a 2-for-1 on 2025-01-01, cumprod on 2024-12-31 = 1, on 2025-01-01 = 2, afterwards = 2
        # Historical price on 2024-12-31 should be divided by 2 to be comparable to post-split prices.
        adjust_factor = cumprod
        # Avoid division by zero
        adjust_factor.replace(0, pd.NA, inplace=True)

        # Broadcast adjust_factor onto price columns
        for col in ["open", "high", "low", "close"]:
            out[col] = out[col] / adjust_factor

        # If provider adj_close exists, do not overwrite; otherwise set adj_close = adjusted close for now
        if not has_provider_adj:
            out["adj_close"] = out["close"]
        else:
            logger.info("Provider adj_close present; splits applied to prices but adj_close preserved from provider.")

    else:
        # No splits provided: if provider adj_close exists, keep it; otherwise leave adj_close = NaN to signal missing
        if has_provider_adj:
            logger.info("No splits provided; using provider adj_close.")
        else:
            out["adj_close"] = pd.NA
            logger.info("No splits or provider adj_close; adj_close left as NA.")

    # --- Dividend handling (not fully implemented) ---
    if dividends is not None:
        logger.warning("Dividends DataFrame provided but dividend-adjustment is NOT implemented in this function. "
                       "Splits were applied (if any). If you need dividend adjustments, request the total-return adjustment method.")
        # Potential place to implement dividend adjustments later.
    # Final dtype normalization
    for col in ["open", "high", "low", "close", "adj_close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" in out.columns:
        try:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")
        except Exception:
            logger.warning("Failed to coerce 'volume' to Int64; keeping original dtype.")

    return out


def validate_provider_adj_close(df: pd.DataFrame, tolerance: float = 1e-6) -> dict:
    """
    Quick validation of provider adj_close vs close.
    - Returns a dict with keys: present (bool), n_mismatches (int), sample_mismatches (list).
    - Does not modify df.
    - tolerance is absolute price difference tolerated (small).
    """
    report = {"present": False, "n_mismatches": 0, "sample_mismatches": []}
    if "adj_close" not in df.columns:
        return report

    report["present"] = True
    # Compare adj_close to close on dates where both present
    mask = df["adj_close"].notna() & df["close"].notna()
    if not mask.any():
        return report

    diffs = (df.loc[mask, "adj_close"].astype(float) - df.loc[mask, "close"].astype(float)).abs()
    # Large diffs may indicate splits/dividend adjustments that aren't applied to close
    mismatch_mask = diffs > tolerance
    n = int(mismatch_mask.sum())
    report["n_mismatches"] = n
    if n > 0:
        # include up to 8 example rows (date, close, adj_close, diff)
        examples = []
        for dt in df.loc[mask].index[mismatch_mask][:8]:
            examples.append({
                "date": pd.to_datetime(dt).date().isoformat(),
                "close": float(df.at[dt, "close"]),
                "adj_close": float(df.at[dt, "adj_close"]),
                "diff": float(abs(df.at[dt, "adj_close"] - df.at[dt, "close"]))
            })
        report["sample_mismatches"] = examples
    return report
