import contextlib

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

from crypto_research.constants import INFLUXDB_HOST, INFLUXDB_TOKEN, INFLUXDB_ORG


@contextlib.asynccontextmanager
async def influx_client(timeout: int = 120, *, sync: bool = False) -> InfluxDBClientAsync | InfluxDBClient:
    """
    Contextmanager for influxdb client.
    Allowed to be used in sync mode.
    Args:
        timeout: timeout in seconds
    """
    client_cls = InfluxDBClient if sync else InfluxDBClientAsync
    _client = client_cls(url=INFLUXDB_HOST, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=timeout * 1000)
    if sync:
        with _client as client:
            yield client
    else:
        async with _client as client:
            yield client
