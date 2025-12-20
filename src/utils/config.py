"""
Central configuration for the project.

This file is the single source of truth for:
- project paths
- instruments and symbols
- data providers
- file formats and conventions

No data logic should live here.
"""

from pathlib import Path
from datetime import date


# =========================
# Project paths
# =========================

# repo_root/src/utils/config.py → repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "clean"


# =========================
# Instruments & symbols
# =========================

UNDERLYING_SYMBOL = "SPY"
VOL_SYMBOL = "VIX"
RATES_SYMBOL = "DTB3"  # 3M T-bill from FRED


# =========================
# Data sources
# =========================

DATA_SOURCES = {
    "underlying": "yahoo",
    "vol": "yahoo",
    "options": "polygon",   # or tradier / wrds
    "rates": "fred",
}


# =========================
# Conventions
# =========================

FILE_FORMAT = "parquet"
TIMEZONE = "US/Eastern"


# =========================
# Path helpers (raw)
# =========================

def raw_underlying_path(symbol: str) -> Path:
    return DATA_RAW / "underlying" / f"{symbol}.{FILE_FORMAT}"


def raw_rates_path() -> Path:
    return DATA_RAW / "rates" / f"{RATES_SYMBOL}.{FILE_FORMAT}"


def raw_options_date_path(dt: date) -> Path:
    """
    One options snapshot per trading day.
    """
    return DATA_RAW / "options" / f"{dt.isoformat()}.{FILE_FORMAT}"


# =========================
# Path helpers (clean)
# =========================

def clean_underlying_path(symbol: str) -> Path:
    return DATA_CLEAN / "underlying" / f"{symbol}.{FILE_FORMAT}"


def clean_rates_path() -> Path:
    return DATA_CLEAN / "rates" / f"{RATES_SYMBOL}.{FILE_FORMAT}"


def clean_options_date_path(dt: date) -> Path:
    return DATA_CLEAN / "options" / f"{dt.isoformat()}.{FILE_FORMAT}"
