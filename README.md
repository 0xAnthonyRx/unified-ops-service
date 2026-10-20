# Unified Enterprise AI Operations Microservice

A production-grade, asynchronous AI microservice uniting **Deterministic Schema Defense** (Pydantic v2 + Instructor), **Two-Stage Hybrid Knowledge Retrieval** (PostgreSQL pgvector + FlashRank + Redis), and **Stateful Autonomous Agents with Human-in-the-Loop Breakpoints** (LangGraph).

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Endpoints & API Capabilities](#core-endpoints--api-capabilities)
- [Multi-Container Setup (Docker Compose)](#multi-container-setup-docker-compose)
- [Project Structure](#project-structure)
- [License](#license)

---

## Architecture Overview

```mermaid
flowchart TD
    Client["Client / HTTP Consumer"]

    subgraph GW["FastAPI Application Gateway"]
        direction TB
        A1["Async DB Connection Pool"]
        A2["Non-Blocking Background Tasks"]
        A3["SHA-256 Redis Response Cache"]
    end

    Client --> GW

    GW --> E1
    GW --> E2
    GW --> E3

    subgraph E1["POST /api/v1/extract · Deterministic Engine"]
        direction TB
        E1a["Schema-enforced validation"]
        E1b["Self-healing retry loops"]
        E1c["Guaranteed math consistency"]
        E1a --> E1b --> E1c
    end

    subgraph E2["POST /api/v1/rag/query · Hybrid Retrieval Engine"]
        direction TB
        E2a["Stage 1 · Dense + Sparse RRF"]
        E2b["Stage 2 · FlashRank rerank"]
        E2c["Sub-10ms Redis cache hit"]
        E2a --> E2b --> E2c
    end

    subgraph E3["POST /api/v1/agent/chat · Stateful Agent Engine"]
        direction TB
        E3a["LangGraph cyclical graph"]
        E3b["Multi-turn thread persistence"]
        E3c["HITL breakpoint authorization"]
        E3a --> E3b --> E3c
    end

    E1 --> DB
    E2 --> DB
    E3 --> DB

    subgraph DB["PostgreSQL (pgvector)"]
        direction TB
        D1["HNSW Vector Cosine Index"]
        D2["English tsvector GIN Index"]
        D3["JSONB Metadata Store"]
    end

    classDef engine fill:#0d1117,stroke:#58a6ff,color:#c9d1d9,stroke-width:1.5px;
    classDef gateway fill:#0d1117,stroke:#7ee787,color:#c9d1d9,stroke-width:1.5px;
    classDef store fill:#0d1117,stroke:#f778ba,color:#c9d1d9,stroke-width:1.5px;
    class E1,E2,E3 engine;
    class GW gateway;
    class DB store;
```

> If GitHub's Mermaid renderer looks cramped on your screen, the diagram below is a drop-in fallback.

<details>
<summary><strong>Fallback ASCII Diagram</strong></summary>

```text
                        [ Client / HTTP Consumer ]
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │        FastAPI Application Gateway       │
                │  - Async DB Connection Pool              │
                │  - Non-Blocking Background Tasks         │
                │  - SHA-256 Redis Response Cache          │
                └────────────────────┬─────────────────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     ▼                               ▼                               ▼
POST /api/v1/extract      POST /api/v1/rag/query        POST /api/v1/agent/chat
[Deterministic Engine]    [Hybrid Retrieval Engine]     [Stateful Agent Engine]

- Schema-enforced         - Stage 1: Dense + Sparse RRF  - LangGraph cyclical graph
  validation              - Stage 2: FlashRank rerank    - Multi-turn thread persistence
- Self-healing retry      - Sub-10ms Redis cache hit     - HITL breakpoint authorization
  loops
- Guaranteed math
  consistency
     │                               │                               │
     └───────────────────────────────┼───────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │           PostgreSQL (pgvector)          │
                    │  - HNSW Vector Cosine Index              │
                    │  - English tsvector GIN Index            │
                    │  - JSONB Metadata Store                  │
                    └──────────────────────────────────────────┘
```

</details>

---

## Core Endpoints & API Capabilities

| Route | Method | Engine | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Core | Checks live PostgreSQL connection pool and Redis cache reachability. |
| `/api/v1/extract` | `POST` | Phase 1 | Parses unstructured business text into strictly validated, mathematically verified schemas. |
| `/api/v1/rag/query` | `POST` | Phase 2 | Two-stage retrieval (SQL RRF + FlashRank cross-encoder) with Redis caching. |
| `/api/v1/rag/ingest` | `POST` | Phase 2 | Schedules asynchronous batch document embedding and indexing via background workers (`HTTP 202`). |
| `/api/v1/agent/chat` | `POST` | Phase 3 | Multi-turn conversational agent with autonomous tool calling and breakpoint interrupts. |
| `/api/v1/agent/decision` | `POST` | Phase 3 | Submits human operator authorization (`approve` / `reject`) to resume frozen agent threads. |

---

## Multi-Container Setup (Docker Compose)

### 1. Prerequisites

- Docker Engine & Docker Compose
- Valid API credentials for Google Gemini and Groq

### 2. Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=vectordb
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vectordb"
REDIS_URL="redis://localhost:6379/0"

GROQ_API_KEY="your-groq-key"
GEMINI_API_KEY="your-gemini-key"
EMBEDDING_MODEL="gemini/gemini-embedding-001"
GENERATION_MODEL="gemini/gemini-3.6-flash"
EXTRACTION_MODEL="gemini/gemini-3.6-flash"
FALLBACK_EXTRACTION_MODEL="groq/openai/gpt-oss-120b"
```

### 3. Launch Stack

```bash
docker compose up --build -d
```

### 4. Interactive API Documentation

Once the stack is healthy, open:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Project Structure

```text
.
├── app/                  # FastAPI application package (routes, engines, models)
├── init-db.sql           # pgvector extension + schema bootstrap
├── docker-compose.yml    # Multi-container orchestration
├── Dockerfile            # Service image build
├── .env.example          # Environment configuration template
├── .dockerignore
├── .gitignore
├── requirements.txt      # Frozen Python dependencies
└── README.md
```

---

## License

MIT
