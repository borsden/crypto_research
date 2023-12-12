# import contextlib
#
# from motor.motor_asyncio import AsyncIOMotorClient
#
#
# @contextlib.asynccontextmanager
# def mongo_client(timeout: int = 1_200) -> AsyncIOMotorClient:
#     async with AsyncIOMotorClient('mongodb://localhost:27017') as client:
#         yield client
#
#
#
# async def save_trading_pairs(client, db_name, collection_name, trading_pairs: list[str]):
#
#     db = client[db_name]
#     collection = db[collection_name]
#
    # collection = db[collection_name]
    #
    # operations = [
    #     UpdateOne(
    #         {'pair': pair},
    #         {'$setOnInsert': {'pair': pair}},
    #         upsert=True
    #     )
    #     for pair in trading_pairs
    # ]
    #
    # await collection.bulk_write(operations)