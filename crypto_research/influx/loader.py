from datetime import datetime

import pandas as pd

from crypto_research.constants import KLINES_COLUMNS
from crypto_research.influx import influx_client
from crypto_research.influx.constants import FUTURES_BUCKET, KLINES_MEASUREMENT_PREFIX, KLINES_DEFAULT_INTERVAL


async def get_klines(
    *,
    interval: str = KLINES_DEFAULT_INTERVAL,
    pair: str = None,
    start: datetime = 0, stop: datetime = None,
    # aggregate_window: str = None, aggregate_func: str = 'mean',
    measurement: str = KLINES_MEASUREMENT_PREFIX,
    bucket: str = FUTURES_BUCKET
) -> pd.DataFrame:
    """
    Load klines data.
    Args:
        interval: interval of klines. Is used to select proper measurement.
        pair: specify pair. if not specified, all pairs are loaded.
        start: start date. if not specified, all data from 0 timestamp is loaded.
        stop: stop date. if not specified, all data from start_date is loaded.
        measurement: klines measurement prefix.
        bucket: bucket
    """

    if start:
        start = int(start.timestamp())

    if stop:
        stop = int(stop.timestamp())
        range_string = f" |> range(start: {start}, stop:{stop}) "
    else:
        range_string = f" |> range(start: {start}) "

    measurement = f"{measurement}_{interval}"
    #
    # if aggregate_window:
    #     aggregate = f" |> aggregateWindow(every: {aggregate_window}, fn: {aggregate_func}, createEmpty: false) "
    # else:
    #     aggregate = ""

    pair_filter = f'and r["pair"]=="{pair}"' if pair else ""

    query = f'''
        from(bucket: "{bucket}")''' + range_string + f'''
          |> filter(fn: (r) => (r["_measurement"] == "{measurement}" {pair_filter}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    async with influx_client() as client:
        query_api = client.query_api()
        data = await query_api.query_data_frame(query)


    data = data.rename(columns={'_time': 'time'})
    data = data.set_index(['time', 'pair'])

    data = data[KLINES_COLUMNS]
    return data


async def get_klines_pairs(
    tag: str = 'pair',
    interval: str = KLINES_DEFAULT_INTERVAL,
    bucket=FUTURES_BUCKET,
    measurement=KLINES_MEASUREMENT_PREFIX
):
    """Get all pairs that exists in measurement for this interval."""
    measurement = f"{measurement}_{interval}"

    async def get_data(range) -> set:
        query = f'''
        from(bucket: "{bucket}")
          |> {range}
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> keep(columns: ["{tag}"])
          |> group()
          |> distinct(column: "{tag}")
          |> keep(columns: ["_value"])
        '''
        result = await query_api.query(query=query)
        result = set(record.get_value() for record in result[0].records)
        return result

    async with influx_client() as client:
        query_api = client.query_api()
        data = await get_data(f"range(start: 0)")

    return data