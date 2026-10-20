import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional
from dotenv import load_dotenv
from pgvector.psycopg import register_vector_async
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
import redis.asyncio as aioredis

load_dotenv()

logger = logging.getLogger("uvicorn.error")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

db_pool: Optional[AsyncConnectionPool] = None
redis_client: Optional[aioredis.Redis] = None


async def configure_connection(conn: psycopg.AsyncConnection):
    """
    Callback executed every time a new connection is created in the pool.
    Registers the pgvector extension handlers for async serialization.
    """
    await register_vector_async(conn)


async def init_services():
    """
    Initializes the PostgreSQL connection pool and Redis client.
    Called inside FastAPI lifespan startup.
    """
    global db_pool, redis_client

    logger.info("Initializing PostgreSQL AsyncConnectionPool...")
    db_pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        min_size=2,
        max_size=20,
        open=False,
        configure=configure_connection,
        kwargs={"row_factory": dict_row},
    )
    await db_pool.open()
    logger.info("PostgreSQL pool ready.")

    logger.info("Connecting to Redis...")
    redis_client = aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        await redis_client.ping()
        logger.info("Redis client connected.")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}). Continuing with cache bypass.")


async def close_services():
    """
    Closes database and cache connections cleanly on shutdown.
    """
    global db_pool, redis_client

    if db_pool:
        logger.info("Closing PostgreSQL AsyncConnectionPool...")
        await db_pool.close()

    if redis_client:
        logger.info("Closing Redis connection...")
        await redis_client.aclose()


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Context manager yielding a pooled PostgreSQL async connection.
    Usage:
        async with get_db_connection() as conn:
            await conn.execute(...)
    """
    if db_pool is None:
        raise RuntimeError("Database connection pool is not initialized.")
    async with db_pool.connection() as conn:
        yield conn


async def get_db() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """FastAPI Dependency for route handlers."""
    async with get_db_connection() as conn:
        yield conn


async def get_redis() -> Optional[aioredis.Redis]:
    """FastAPI Dependency returning the Redis client."""
    return redis_client


# ---------------------------------------------------------------------------
# Caching Helpers
# -----------
async def get_cached_response(cache_key: str) -> Optional[dict]:
    """
    Retrieves and deserializes a JSON payload from Redis.
    Fails safely by returning None if Redis is unreachable.
    """
    if not redis_client:
        return None
    try:
        data = await redis_client.get(cache_key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.error(f"Cache lookup failed for key '{cache_key}': {e}")
    return None


async def set_cached_response(cache_key: str, data: Any, ttl_seconds: int = 3600) -> None:
    """
    Serializes and stores a response payload in Redis with an expiration TTL.
    """
    if not redis_client:
        return
    try:
        payload = json.dumps(data)
        await redis_client.setex(cache_key, ttl_seconds, payload)
    except Exception as e:
        logger.error(f"Cache write failed for key '{cache_key}': {e}")
