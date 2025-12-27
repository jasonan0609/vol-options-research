"""
Sanity checks for Day-1 pipeline.

Produces:
- textual summary printed to stdout
- JSON summary written to reports/sanity_checks/YYYY-MM-DD_summary.json
- PNG plots written to reports/sanity_checks/

Run:
    python -m src.cleaning.sanity_checks --date 2025-12-24

Notes:
- Uses cleaned files:
    data/clean/underlying/{symbol}.parquet
    data/clean/options/YYYY-MM-DD.parquet
- No network calls. Pure analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import UNDERLYING_SYMBOL
from src.utils.config import clean_underlying_path
from src.utils.config import clean_options_date_path

REPORT_ROOT = Path("reports/sanity_checks")
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def load_clean_underlying(symbol: str):
    p = clean_underlying_path(symbol)
    if not p.exists():
        raise FileNotFoundError(f"Clean underlying file not found: {p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index).normalize()
    return df


def load_clean_options(trade_date: date):
    p = clean_options_date_path(trade_date)
    if not p.exists():
        raise FileNotFoundError(f"Clean options file not found: {p}")
    df = pd.read_parquet(p)
    return df


def check_spot_continuity(df_under: pd.DataFrame, max_daily_return=0.2):
    """
    1) Check for missing dates (holes) in index
    2) Compute daily returns and flag extreme returns > max_daily_return (20% default)
    """
    idx = df_under.index
    # trading days from min->max via trading_calendar could be used; here infer contiguous business days
    missing = []
    # detect gaps: any non-consecutive business days
    diffs = idx.to_series().diff().dropna()
    # convert to days
    gap_days = diffs.dt.days
    # any gap bigger than 1 day indicates missing trading days
    holes = gap_days[gap_days > 1]
    if not holes.empty:
        missing = [{"date_after": str(d.index[i].date()), "gap_days": int(gap_days.iloc[i])}
                   for i in range(len(gap_days)) if gap_days.iloc[i] > 1]

    # daily returns
    returns = df_under["close"].astype(float).pct_change().dropna()
    extreme = returns[returns.abs() > max_daily_return]
    return {"n_rows": len(df_under), "n_holes": len(holes), "holes_example": missing[:5],
            "n_extreme_returns": int(len(extreme)),
            "extreme_examples": [
                {"date": str(d.date()), "ret": float(r)} for d, r in extreme.iloc[:6].items()
            ]}


def check_options_counts(df_opts: pd.DataFrame):
    """
    Count options by expiry and by cp, return summary.
    """
    total = len(df_opts)
    counts_by_expiry = df_opts.groupby("expiry").size().sort_values(ascending=False).to_dict()
    counts_by_cp = df_opts.groupby("cp").size().to_dict()
    return {"total_rows": int(total), "by_expiry": counts_by_expiry, "by_cp": counts_by_cp}


def check_call_put_parity(df_opts: pd.DataFrame, spot: float | None, tolerance=0.05):
    """
    For each expiry and strikes near ATM (moneyness within [0.9,1.1]), compute mean(call.mid - put.mid)
    and compare to (spot - strike). This is a simplified parity check (ignores rates/time-value).
    Reports strikes where abs(diff) > tolerance (in price units).
    """
    if df_opts is None or df_opts.empty:
        return {"checked": False, "reason": "no options"}

    # pivot calls and puts by expiry and strike using mid
    df = df_opts.copy()
    df = df.dropna(subset=["mid", "strike"])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])
    # build wide table
    calls = df[df["cp"] == "C"].set_index(["expiry", "strike"])["mid"]
    puts = df[df["cp"] == "P"].set_index(["expiry", "strike"])["mid"]
    merged = pd.concat([calls.rename("call_mid"), puts.rename("put_mid")], axis=1).dropna()
    if merged.empty:
        return {"checked": False, "reason": "no matching call-put pairs"}

    # select near-ATM pairs using spot if available otherwise use median strike
    results = []
    for expiry, group in merged.reset_index().groupby("expiry"):
        g = group.set_index("strike")
        if spot is None:
            atm = g.index.median()
        else:
            atm = spot
        msk = (g.index / atm).between(0.9, 1.1)
        sub = g[msk]
        if sub.empty:
            continue
        # compute parity residual = call - put - (spot - strike)
        sub = sub.copy()
        sub["residual"] = sub["call_mid"] - sub["put_mid"] - (atm - sub.index)
        # flag large residuals
        flags = sub[~sub["residual"].abs().le(tolerance)]
        for strike_val, row in flags.iterrows():
            results.append({"expiry": str(expiry), "strike": float(strike_val), "residual": float(row["residual"])})
    return {"checked": True, "n_flags": len(results), "flags": results[:20]}


def plot_moneyness_hist(df_opts: pd.DataFrame, trade_date: date, outdir: Path):
    if df_opts is None or df_opts.empty:
        return None
    df = df_opts.copy()
    if "moneyness" not in df.columns:
        return None
    plt.figure()
    df["moneyness"].dropna().astype(float).hist(bins=60)
    plt.title(f"Moneyness histogram {trade_date}")
    plt.xlabel("moneyness (strike/spot)")
    plt.ylabel("count")
    p = outdir / f"moneyness_{trade_date}.png"
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_missing_heatmap(df_opts: pd.DataFrame, trade_date: date, outdir: Path):
    """
    Simple heatmap: rows = expiry, cols = strike buckets, value = fraction missing mid
    For readability, limit to top N expiries by count.
    """
    if df_opts is None or df_opts.empty:
        return None
    df = df_opts.copy()
    # choose top expiries
    top_exps = df["expiry"].value_counts().nlargest(8).index.tolist()
    df = df[df["expiry"].isin(top_exps)]
    # bin strikes into 30 buckets per expiry
    rows = []
    exps = []
    for exp in top_exps:
        g = df[df["expiry"] == exp]
        if g.empty:
            continue
        strikes = pd.qcut(g["strike"].rank(method="first"), q=30, duplicates="drop")
        # compute fraction missing mid per bucket
        missing_frac = g.groupby(strikes)["mid"].apply(lambda s: s.isna().mean())
        rows.append(missing_frac.values)
        exps.append(str(exp))
    if not rows:
        return None
    import numpy as _np
    mat = _np.vstack(rows)
    plt.figure(figsize=(10, max(3, len(rows))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="frac missing mid")
    plt.yticks(range(len(exps)), exps)
    plt.xlabel("strike bucket")
    plt.title(f"Missing-mid heatmap {trade_date}")
    p = outdir / f"missing_mid_heatmap_{trade_date}.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return str(p)


def plot_oi_timeseries(df_opts: pd.DataFrame, trade_date: date, outdir: Path):
    if df_opts is None or df_opts.empty:
        return None
    
    # Sort by index (Date) instead of values (Counts)
    by_exp = df_opts.groupby("expiry")["open_interest"].sum().sort_index()
    
    plt.figure(figsize=(12, 6))
    by_exp.plot(kind="bar", color="skyblue")
    plt.title(f"Total OI by Expiry (Chronological) - {trade_date}")
    plt.ylabel("Open Interest")
    plt.xlabel("Expiry Date")
    plt.xticks(rotation=45)
    
    p = outdir / f"oi_by_expiry_{trade_date}.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return str(p)


def run_checks(trade_date: date, underlying: str = UNDERLYING_SYMBOL):
    summary = {"date": str(trade_date), "underlying": underlying, "run_ts": datetime.utcnow().isoformat()}
    outdir = REPORT_ROOT
    try:
        df_under = load_clean_underlying(underlying)
    except Exception as e:
        df_under = None
        summary["underlying_error"] = str(e)

    try:
        df_opts = load_clean_options(trade_date)
    except Exception as e:
        df_opts = None
        summary["options_error"] = str(e)

    # Spot continuity
    try:
        sc = check_spot_continuity(df_under) if df_under is not None else {"checked": False}
        summary["spot_continuity"] = sc
    except Exception as e:
        summary["spot_continuity_error"] = str(e)

    # Options counts
    try:
        oc = check_options_counts(df_opts) if df_opts is not None else {"checked": False}
        summary["options_counts"] = oc
    except Exception as e:
        summary["options_counts_error"] = str(e)

    # parity
    spot_val = None
    if df_under is not None:
        spot_val = float(df_under.at[pd.to_datetime(trade_date).normalize(), "close"]) if pd.to_datetime(trade_date).normalize() in df_under.index else None
    try:
        parity = check_call_put_parity(df_opts, spot=spot_val)
        summary["parity"] = parity
    except Exception as e:
        summary["parity_error"] = str(e)

    # plots
    try:
        m_hist = plot_moneyness_hist(df_opts, trade_date, outdir)
        summary["plot_moneyness"] = m_hist
    except Exception as e:
        summary["plot_moneyness_error"] = str(e)

    try:
        heat = plot_missing_heatmap(df_opts, trade_date, outdir)
        summary["plot_missing_heatmap"] = heat
    except Exception as e:
        summary["plot_missing_heatmap_error"] = str(e)

    try:
        oi_plot = plot_oi_timeseries(df_opts, trade_date, outdir)
        summary["plot_oi_by_expiry"] = oi_plot
    except Exception as e:
        summary["plot_oi_by_expiry_error"] = str(e)

    # write summary
    final_summary = make_json_ready(summary)
    outp = REPORT_ROOT / f"{trade_date}_summary.json"
    
    # This will now succeed because final_summary only has strings as keys
    outp.write_text(json.dumps(final_summary, indent=2))
    
    print(f"\n✅ SUCCESS: Sanity report created at {outp}")
    return final_summary


def make_json_ready(obj):
        if isinstance(obj, dict):
            # This handles your "keys must be str" error
            return {str(k): make_json_ready(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_ready(item) for item in obj]
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item() # Converts numpy types to standard Python types
        return obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="YYYY-MM-DD", required=True)
    p.add_argument("--symbol", type=str, default=UNDERLYING_SYMBOL)
    return p.parse_args()


def main():
    args = parse_args()
    d = pd.to_datetime(args.date).date()
    run_checks(d, underlying=args.symbol)


if __name__ == "__main__":
    main()
