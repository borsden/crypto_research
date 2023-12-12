import contextlib

from binance import AsyncClient

from crypto_research.constants import BINANCE_API_KEY, BINANCE_API_SECRET


@contextlib.asynccontextmanager
async def binance_client(client: AsyncClient | None = None) -> AsyncClient:
    """Create binance client if not specified. Close after use"""
    if client:
        yield client
    else:
        client = await AsyncClient.create(
            BINANCE_API_KEY, BINANCE_API_SECRET,
            session_params={"timeout": 600},
            requests_params={'timeout': 600}
        )
        yield client
        await client.close_connection()

