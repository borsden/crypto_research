from datetime import datetime, timedelta

from influxdb_client.domain.bucket import Bucket
from loguru import logger

from crypto_research.influx import influx_client


async def create_bucket(bucket: str, force: bool = False):
    """
    Create bucket.
    Bucket api is not provided in asyncclient, so use it in sync mode.
    Args:
        bucket: name of bucket
        force: if True, overwrite existing.
    """
    bucket_name = bucket
    async with influx_client(sync=True) as client:
        buckets_api = client.buckets_api()
        bucket = buckets_api.find_bucket_by_name(bucket)
        if bucket:
            if force:
                logger.info(f"Deleting bucket {bucket.name}")
                buckets_api.delete_bucket(bucket)
            else:
                logger.info(f"Bucket {bucket.name} already exists.")
                return

        logger.info(f"Creating bucket {bucket_name}")
        buckets_api.create_bucket(bucket_name=bucket_name)


async def delete_last_N_days(measurement: str, bucket: str, days: int):
    """
    Delete last N days in measurement.
    Args:
        measurement: measurement name
        bucket: bucket name
        number of days to delete

    """
    async with influx_client() as client:
        now = datetime.utcnow()
        delta = timedelta(days=days)
        predicate = f'_measurement="{measurement}"'
        # # TO DELETE ALL ROWS.
        await client.delete_api().delete(
            bucket=bucket, start=now - delta, stop=now + delta, predicate=predicate
        )


async def get_buckets() -> list[str]:
    """Get list of buckets."""
    async with influx_client(sync=True) as client:
        def bucket_names() -> str:
            buckets = client.buckets_api().find_buckets().buckets
            for bucket in buckets:
                bucket: Bucket
                yield bucket.name

        return list(bucket_names())
