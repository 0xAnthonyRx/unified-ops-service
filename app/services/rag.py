import asyncio
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional
from flashrank import Ranker, RerankRequest
import litellm
from pgvector import Vector
import psycopg
from psycopg.types.json import Jsonb
from app.schemas.contracts import DocumentIngestItem, RAGQueryResponse
from app.services.database import (
    get_cached_response,
    get_db_connection,
    set_cached_response,
)

logger = logging.getLogger("uvicorn.error")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini/gemini-3.8-flash")
FALLBACK_GENERATION_MODEL = os.getenv("FALLBACK_GENERATION_MODEL", "groq/openai/gpt-oss-120b")

# Concurrency bounding: prevents CPU exhaustion under heavy concurrent reranking
_RERANK_SEMAPHORE = asyncio.Semaphore(10)

# Lazy singleton for FlashRank: prevents import-time crashes if cache or weights fail to load
_ranker: Optional[Ranker] = None
_ranker_lock = asyncio.Lock()


async def get_ranker() -> Ranker:
    """Lazy-loads the cross-encoder model safely on first use."""
    global _ranker
    if _ranker is None:
        async with _ranker_lock:
            if _ranker is None:
                _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank_cache")
    return _ranker


def clean_query_text(query: str) -> str:
    """Strips punctuation and stop verbs to yield clean natural language tokens."""
    clean = re.sub(r"[^\w\s-]", " ", query)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
    stop_verbs = {"find", "show", "get", "search", "where", "what", "how", "who"}
    meaningful = [t for t in tokens if t.lower() not in stop_verbs]
    return " ".join(meaningful if meaningful else tokens)


async def get_embedding_async(text: str) -> Vector:
    """Generates a dense vector asynchronously with exponential backoff retries."""
    response = await litellm.aembedding(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIM,
        num_retries=3,
    )
    return Vector(response.data[0]["embedding"])


async def hybrid_search_async(query: str, limit: int = 5, candidates_limit: int = 20) -> List[Dict[str, Any]]:
    """Stage 1 RRF Search: dynamically scales candidate limits and avoids SQL string injection."""
    query_vector = await get_embedding_async(query)
    cleaned_query = clean_query_text(query)

    # Uses plainto_tsquery to enforce precision (AND semantics) and prevent hyphen syntax errors
    sql = """
    WITH vector_candidates AS (
        SELECT 
            id, 
            content, 
            metadata,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector ASC) AS rank
        FROM documents
        ORDER BY embedding <=> %(vec)s::vector ASC
        LIMIT %(candidates)s
    ),
    keyword_candidates AS (
        SELECT 
            id, 
            content, 
            metadata,
            ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, plainto_tsquery('english', %(clean_query)s)) DESC) AS rank
        FROM documents
        WHERE tsv @@ plainto_tsquery('english', %(clean_query)s)
        ORDER BY ts_rank(tsv, plainto_tsquery('english', %(clean_query)s)) DESC
        LIMIT %(candidates)s
    )
    SELECT 
        COALESCE(v.id, k.id) AS id,
        COALESCE(v.content, k.content) AS content,
        COALESCE(v.metadata, k.metadata) AS metadata,
        ROUND((
            COALESCE(1.0 / (60 + v.rank), 0.0) +
            COALESCE(1.0 / (60 + k.rank), 0.0)
        )::numeric, 5) AS rrf_score
    FROM vector_candidates v
    FULL OUTER JOIN keyword_candidates k ON v.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %(limit)s;
    """

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                sql,
                {
                    "vec": query_vector,
                    "clean_query": cleaned_query,
                    "candidates": candidates_limit,
                    "limit": limit,
                },
            )
            return await cur.fetchall()


def _sync_rerank(
    ranker_instance: Ranker,
    query: str, 
    candidates: List[Dict[str, Any]], 
    min_score: float, 
    top_k: int
) -> List[Dict[str, Any]]:
    """CPU-bound worker executing FlashRank cross-encoder inference."""
    if not candidates:
        return []

    # Unifies the document reference identifier: prefers metadata doc_id, falls back to DB id
    passages = [
        {
            "id": str(c["metadata"].get("doc_id") or c["id"]),
            "db_id": c["id"],
            "text": c["content"],
            "meta": c["metadata"],
        }
        for c in candidates
    ]

    rerank_req = RerankRequest(query=query, passages=passages)
    reranked = ranker_instance.rerank(rerank_req)
    valid = [p for p in reranked if p["score"] >= min_score]
    return valid[:top_k]


async def query_rag_pipeline(query: str, top_k: int = 3, min_score: float = 0.20) -> RAGQueryResponse:
    """
    End-to-End RAG execution pipeline:
    1. Check comprehensive Redis cache key (query + top_k + min_score).
    2. Stage 1: Async PostgreSQL Hybrid RRF Search.
    3. Stage 2: Concurrency-bounded local FlashRank cross-encoder rerank.
    4. Stage 3: Deterministic (temp=0.0) LLM generation with inline citations.
    5. Cache payload with 1-hour TTL.
    """
    # 1. Comprehensive Cache Key Hash
    cache_payload = f"{query.strip().lower()}:{top_k}:{min_score}"
    cache_key = f"rag:{hashlib.sha256(cache_payload.encode()).hexdigest()}"
    
    cached_data = await get_cached_response(cache_key)
    if cached_data:
        return RAGQueryResponse(**cached_data, cached=True)

    # 2. Stage 1: Hybrid Retrieval with dynamic candidate window
    candidates_limit = max(top_k * 4, 20)
    candidates = await hybrid_search_async(query, limit=top_k * 2, candidates_limit=candidates_limit)
    
    if not candidates:
        return RAGQueryResponse(
            query=query,
            answer="No relevant documentation was found in the knowledge base matching this query.",
            sources=[],
            cached=False,
        )

    # 3. Stage 2: Rerank with concurrency semaphore & lazy-loaded ranker
    ranker_instance = await get_ranker()
    async with _RERANK_SEMAPHORE:
        reranked = await asyncio.to_thread(_sync_rerank, ranker_instance, query, candidates, min_score, top_k)

    if not reranked:
        return RAGQueryResponse(
            query=query,
            answer="No high-confidence documents passed the relevance threshold for this query.",
            sources=[],
            cached=False,
        )

    # 4. Context Synthesis (consistent Document IDs)
    context_blocks = [f"Document [{r['id']}]:\n{r['text']}" for r in reranked]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user query using ONLY "
        "the provided context. Cite sources inline using [Document ID] exactly as provided. "
        "Do not invent facts or extrapolate beyond the provided text."
    )

    response = await litellm.acompletion(
        model=GENERATION_MODEL,
        fallbacks=[FALLBACK_GENERATION_MODEL],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
        ],
        temperature=0.0,  # Enforces reproducibility and citation determinism
        num_retries=3,
    )

    answer_text = response.choices[0].message.content
    sources = [r["id"] for r in reranked]

    result_payload = {
        "query": query,
        "answer": answer_text,
        "sources": sources,
    }

    # 5. Store validated response in Redis cache (1-hour TTL)
    await set_cached_response(cache_key, result_payload, ttl_seconds=3600)

    return RAGQueryResponse(**result_payload, cached=False)


async def ingest_documents_batch(documents: List[DocumentIngestItem]) -> int:
    """
    Background batch ingestion using nested savepoints:
    A failure on one document will roll back only that document's savepoint,
    preventing entire transaction block poisoning.
    """
    logger.info(f"Starting batch ingestion of {len(documents)} documents...")
    inserted_count = 0

    async with get_db_connection() as conn:
        # Outer transaction manages the overall batch connection
        async with conn.transaction():
            for idx, doc in enumerate(documents, start=1):
                try:
                    # Nested transaction creates an automatic SQL SAVEPOINT in Psycopg 3
                    async with conn.transaction():
                        embedding = await get_embedding_async(doc.content)
                        await conn.execute(
                            """
                            INSERT INTO documents (content, metadata, embedding)
                            VALUES (%s, %s, %s::vector);
                            """,
                            (doc.content, Jsonb(doc.metadata), embedding),
                        )
                        inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest document item #{idx}: {e}")
                    # Savepoint auto-rolls back on exception, keeping transaction healthy for doc idx+1

    logger.info(f"Batch ingestion complete. Successfully committed {inserted_count}/{len(documents)} documents.")
    return inserted_count
