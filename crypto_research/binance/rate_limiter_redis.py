# import asyncio
# import aioredis
#
# class RateLimit:
#     def __init__(self, redis_url, total_weight=6000, update_period=60):
#         self.redis_url = redis_url
#         self.total_weight = total_weight
#         self.update_period = update_period
#         self._update_weight_task = None
#         self.redis = None
#
#     async def initialize(self):
#         self.redis = await aioredis.create_redis_pool(self.redis_url)
#         await self.redis.set('current_weight', self.total_weight)
#
#     async def take_weight(self, weight: int):
#         """Take value from total weight in Redis atomically using Lua scripting."""
#         lua_script = """
#         local current_weight = redis.call('get', KEYS[1])
#         if not current_weight then
#             current_weight = ARGV[2]
#             redis.call('set', KEYS[1], current_weight)
#         else
#             current_weight = tonumber(current_weight)
#         end
#         local weight = tonumber(ARGV[1])
#         if current_weight >= weight then
#             current_weight = current_weight - weight
#             redis.call('set', KEYS[1], current_weight)
#             return current_weight
#         else
#             return -1
#         end
#         """
#         script = self.redis.register_script(lua_script)
#
#         while True:
#             result = await script(keys=['current_weight'], args=[weight, self.total_weight])
#             if result != -1:
#                 break
#             else:
#                 await asyncio.sleep(2)
#
#     async def update_limit(self):
#         """Every update period updates allowed limit in Redis."""
#         while True:
#             await asyncio.sleep(self.update_period)
#             await self.redis.set('current_weight', self.total_weight)
#
#     async def __aenter__(self) -> 'RateLimit':
#         await self.initialize()
#         self._update_weight_task = asyncio.create_task(self.update_limit())
#         return self
#
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         self._update_weight_task.cancel()
#         self.redis.close()
#         await self.redis.wait_closed()
#
#     async def with_take_weight(self, task, weight: int):
#         """Call function and take weight for this call."""
#         await self.take_weight(weight)
#         return await task
#
# # Usage example
# async def main():
#     async with RateLimit('redis://localhost', total_weight=6000, update_period=60) as rate_limiter:
#         # Use rate_limiter here
#         pass
#
# if __name__ == '__main__':
#     asyncio.run(main())
