import uuid
from fastapi import APIRouter, BackgroundTasks, status
from app.schemas.contracts import (
    DocumentIngestRequest,
    IngestJobResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.rag import ingest_documents_batch, query_rag_pipeline

router = APIRouter(prefix="/api/v1/rag", tags=["Hybrid RAG"])


@router.post("/query", response_model=RAGQueryResponse, status_code=status.HTTP_200_OK)
async def query_knowledge_base(request: RAGQueryRequest):
    """
    Executes Stage 1 Hybrid Search + Stage 2 FlashRank Rerank + Grounded Generation.
    Cached via Redis.
    """
    return await query_rag_pipeline(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )


@router.post("/ingest", response_model=IngestJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents(request: DocumentIngestRequest, background_tasks: BackgroundTasks):
    """
    Schedules asynchronous batch document embedding and indexing in the background.
    Immediately returns 202 Accepted with a job_id.
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(ingest_documents_batch, request.documents)
    return IngestJobResponse(
        job_id=job_id,
        status="processing",
        total_documents=len(request.documents),
        message="Document batch scheduled for embedding and indexing.",
    )
