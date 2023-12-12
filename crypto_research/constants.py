import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).parents[1] / ".env"
PRIVATE_ENV_FILE = Path(__file__).parents[1] / ".private.env"

load_dotenv(ENV_FILE, override=True)
load_dotenv(PRIVATE_ENV_FILE, override=True)

INFLUXDB_HOST = "127.0.0.1"
INFLUXDB_PORT = 8086
INFLUXDB_HOST = f"http://{INFLUXDB_HOST}:{INFLUXDB_PORT}"
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')


BINANCE_SOCKET = 'wss://stream.binance.com:9443/ws'
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')


# MONGO_HOST = "mongo"
# INFLUXDB_PORT = 8086
# INFLUXDB_HOST = f"http://{INFLUX_GLOBAL_HOST}:{INFLUXDB_PORT}"
# INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
# INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')
#
KLINES_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', "taker_buy_quote_asset_volume"
]
