"""

Trading calendar utilities.



Function:

trading_days(start_date, end_date, exchange='NYSE') -> pandas.DatetimeIndex



Notes:

- Uses pandas_market_calendars (PMCal) to obtain official exchange schedules.

- Returns timezone-naive DatetimeIndex of trading days (midnight timestamps).

- Keep this centralized — all ingestion/cleaning should use this to decide which

dates must exist.

"""

from datetime import date
from typing import Union
import pandas as pd

# Define the type alias
DateLike = Union[str, date, pd.Timestamp]

try: 
    import pandas_market_calendars as mcl
except ImportError as e:
    raise ImportError(
        "pandas_market_calendars is required. Install with:\n"
        "    pip install pandas-market-calendars"
    ) from e

def _to_iso(d: DateLike) -> str:
    """Normalize input to ISO date string suitable for PMCal."""
    if isinstance(d, str):
        return d
    return pd.to_datetime(d).date().isoformat()

def trading_days(start_date: DateLike, end_date: DateLike, exchange: str = "NYSE") -> pd.DatetimeIndex:
    """
    Return a pandas.DatetimeIndex of trading days (timezone-naive) 
    between start_date and end_date inclusive.
    """
    start_iso = _to_iso(start_date)
    end_iso = _to_iso(end_date)

    cal = mcl.get_calendar(exchange)
    # Fixed the typo here: changed '-' to '='
    schedule = cal.schedule(start_date=start_iso, end_date=end_iso)

    # Convert index to naive midnight timestamps
    dates = pd.to_datetime(schedule.index.date)
    return pd.DatetimeIndex(dates)

if __name__ == "__main__":
    # Test for January 2024
    try:
        idx = trading_days("2024-01-01", "2024-01-10")
        print("Trading days found:")
        print(idx)
    except Exception as e:
        print(f"Error: {e}")