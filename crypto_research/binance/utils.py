"""Utils to use in binance interaction."""
from datetime import datetime, timedelta

from binance.helpers import interval_to_milliseconds


def datetime_to_binance_timestamp(d: datetime):
    """Convert datetime into binance timestamp value."""
    return int(d.timestamp() * 1000)


def split_date_range_to_batches(start: datetime, end: datetime, interval: str, limit: int=1000):
    """Split date range into a batches with size=interval."""
    if interval[-1] == 'M': # do not worry about batches when interval is month.
        yield start, end
    else:
        interval_in_seconds = interval_to_milliseconds(interval)
        step = interval_in_seconds * limit
        while start < end:
            current_end = start + timedelta(milliseconds=step)
            yield start, min(end, current_end)
            start = current_end