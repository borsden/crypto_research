"""
Save data in influxdb database.
"""
import pandas as pd

from crypto_research.influx import influx_client
from crypto_research.influx.constants import KLINES_MEASUREMENT_PREFIX, FUTURES_BUCKET


async def save_klines_influxdb(
    data: pd.DataFrame, *,
    interval: str,
    measurement: str = KLINES_MEASUREMENT_PREFIX,
    bucket: str = FUTURES_BUCKET,
):
    """
    Save klines into influxdb.
    Args:
        data (pd.DataFrame): klines dataframe. Manually set index=time for influxdb.
        interval: data interval. is used for measurement name
        measurement: measurement prefix
        bucket: influx bucket.
    """
    data = data.reset_index()
    data = data.set_index('time')

    measurement = f"{measurement}_{interval}"

    async with influx_client() as client:
        write_api = client.write_api()
        await write_api.write(
            bucket=bucket,
            record=data,
            data_frame_measurement_name=measurement,
            data_frame_tag_columns=['pair']
        )
