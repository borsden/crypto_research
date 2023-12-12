"""
Asynchronous rate limiter for API requests.
It manages request frequency to adhere to Binance API rate limits.

The goal is to use it across different processes as one point of truth for current binance usage.
Implementation via Redis is suggegsted.
"""
import asyncio
import functools


class RateLimit:
    def __init__(self, total_weight=6000, update_period=60):
        self.lock = asyncio.Lock()
        self.total_weight = total_weight
        self.current_weight = total_weight
        self.update_period = update_period
        self._update_weight_task = None

    async def take_weight(self, weight: int):
        """Take value from total weight."""
        async with self.lock:
            while weight > self.current_weight:
                await asyncio.sleep(2)
            self.current_weight -= weight

    async def update_limit(self):
        """Every update period updates allowed limit."""
        while True:
            await asyncio.sleep(self.update_period)
            self.current_weight = self.total_weight

    def __enter__(self) -> 'RateLimit':
        self._update_weight_task = asyncio.create_task(self.update_limit())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._update_weight_task.cancel()

    async def with_take_weight(self, task, weight:int):
        """Call function and take weight for this call.
        """

        await self.take_weight(weight)
        return await task
