"""
Binance API methods.
"""
import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterable

import pandas as pd
import tqdm
from binance import AsyncClient

from crypto_research.binance import binance_client
from crypto_research.binance.rate_limiter import RateLimit
from crypto_research.binance.utils import datetime_to_binance_timestamp, split_date_range_to_batches

import logging

from crypto_research.constants import KLINES_COLUMNS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("binance")
logger.setLevel(logging.DEBUG)

BINANCE_KLINES_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                  'number_of_trades', 'taker_buy_base_asset_volume', "taker_buy_quote_asset_volume", "_"]


async def get_all_trading_pairs(usdt_only: bool = True) -> list[str]:
    """
    Fetch all trading pairs from Binance API.
    Args:
          usdt_only: Flag, indicates, that only usdt trading pairs should be returned.
    """
    async with binance_client() as client:
        exchange_info = await client.futures_exchange_info()
        trading_pairs = (symbol['symbol'].lower() for symbol in exchange_info['symbols'])
        if usdt_only:
            trading_pairs = (pair for pair in trading_pairs if pair.endswith('usdt'))
        return list(trading_pairs)


async def _get_historical_klines(
    trading_pair: str,
    start: datetime,
    end: datetime,
    interval: str,
    api_limit: int = 1000,
    client: AsyncClient = None
) -> pd.DataFrame:
    """Get historical data for trading pair."""
    async with binance_client(client) as client:
        data = await client.futures_historical_klines(
            symbol=trading_pair.lower(), interval=interval,
            start_str=datetime_to_binance_timestamp(start),
            end_str=datetime_to_binance_timestamp(end),
            limit=api_limit
        )
    data = pd.DataFrame(data, columns=BINANCE_KLINES_COLUMNS)
    data['time'] = pd.to_datetime(data['close_time'] / 1000, unit='s')
    data.index = data['time']
    data = data[KLINES_COLUMNS]
    data = data.astype(float)
    data['number_of_trades'] = data['number_of_trades'].astype(float)
    data["pair"] = trading_pair
    return data.set_index(['pair'], append=True)


async def get_historical_klines(
    *trading_pairs,
    start: datetime,
    end: datetime,
    interval: str,
    rate_limit_weight: int = 2400,
    progress: bool = True,
    concatenated: bool = False,
    max_retries: int = 3,
    api_weight: int = 5,
    api_limit: int = 1000
) -> AsyncIterable | pd.DataFrame:
    """
    Receive klines for specified trading pairs.
    Args:
        trading_pairs: trading pairs to retrieve data
        start: start date
        end: end date
        interval: interval in string
        progress: show progress bar
        concatenated: should data be concatenated and sorted or return data when it appears
        max_retries: how many times to retry request if it fails.
        api_weight: weight of request for endpoint to binance. Normally should be function of api_limit
        api_limit: max number of data to receive in one request.
    """

    def split_into_batches():
        date_intervals = list(split_date_range_to_batches(start, end, limit=api_limit, interval=interval))
        for trading_pair in trading_pairs:
            for start_time, end_time in date_intervals:
                yield trading_pair, start_time, end_time

    async def get_data_batches(batches):
        # Use RateLimit to limit api calls in minute.
        with RateLimit(update_period=62, total_weight=rate_limit_weight) as rate_limit:
            async with binance_client() as client:

                tasks = (
                    _get_historical_klines(pair, start=start_time, end=end_time, interval=interval, client=client)
                    for pair, start_time, end_time in batches
                )


                # Weight of every call is 5.
                tasks = [rate_limit.with_take_weight(task=task, weight=api_weight) for task in tasks]
                tasks_iterator = asyncio.as_completed(tasks)
                if progress:
                    tasks_iterator = tqdm.tqdm(tasks_iterator, total=len(tasks))

                for task, batch in zip(tasks_iterator, batches):
                    try:
                        result = await task
                        yield result
                    except Exception as e:
                        # Todo: refactor this.
                        e.batch = batch
                        yield e

    async def get_data():
        """Get data and also retires N times for failed requests."""
        batches = list(split_into_batches())
        errors = []
        async for batch_result in get_data_batches(batches):
            if isinstance(batch_result, Exception):
                errors.append(batch_result)
            else:
                yield batch_result

        retry_count = 0
        while errors and retry_count < max_retries:
            retry_count += 1
            logger.info(f"There are {len(errors)} failed requests. Retrying {retry_count}/{max_retries}...")
            batches = [error.batch for error in errors]
            errors = []
            async for batch_result in get_data_batches(batches):
                if isinstance(batch_result, Exception):
                    errors.append(batch_result)
                else:
                    yield batch_result

        # Todo: Should still be handled
        if errors:
            for error in errors:
                logger.info(f"batch {error.batch} failed with {error}")


    if concatenated:
        data = [data async for data in get_data()]
        data = pd.concat(data).sort_index(level=0, ascending=True)
        return data
    else:

        return get_data()

async def get_half_year_volumes(trading_pairs: list[str]) -> pd.DataFrame:
    """
    Receive last half year trading volume for specified trading pairs.
    """

    end = datetime.now()
    start = end - timedelta(days=365 // 2)

    data = await get_historical_klines(
        *trading_pairs,
        start=start, end=end, interval='1M', concatenated=True
    )
    total_volumes = data['quote_asset_volume'].groupby('pair').sum().sort_values(ascending=False)
    return total_volumes


async def get_trading_pairs_existing_at(trading_pairs: list[str], existing_at: datetime = None) -> list[str]:
    """
    Find and take trading pairs that have data from existing_at.
    """
    if existing_at is None:
        existing_at = datetime.now() - timedelta(days=365 * 2)
    end = existing_at + timedelta(weeks=4)
    data = await get_historical_klines(
        *trading_pairs,
        start=existing_at, end=end, interval='1M', concatenated=True
    )
    return list(data.reset_index()['pair'])
