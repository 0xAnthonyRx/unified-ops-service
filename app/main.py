from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import agent, extract, rag
from app.schemas.contracts import HealthResponse
from app.services.database import close_services, get_db_connection, get_redis, init_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB connection pool and Redis
    await init_services()
    yield
    # Shutdown: Close all connections cleanly
    await close_services()


app = FastAPI(
    title="Unified Operations AI Engine",
    description="Enterprise API providing Deterministic Extraction, Hybrid RAG, and Stateful HITL Agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(extract.router)
app.include_router(rag.router)
app.include_router(agent.router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Checks live connectivity to PostgreSQL and Redis."""
    postgres_status = "unavailable"
    redis_status = "unavailable"

    try:
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1;")
                postgres_status = "healthy"
    except Exception:
        pass

    redis_client = await get_redis()
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = "healthy"
        except Exception:
            pass

    overall = "healthy" if postgres_status == "healthy" else "degraded"
    return HealthResponse(
        status=overall,
        postgres=postgres_status,
        redis=redis_status,
    )
