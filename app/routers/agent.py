from fastapi import APIRouter, status
from app.schemas.contracts import (
    AgentChatRequest,
    AgentChatResponse,
    HITLDecisionRequest,
)
from app.services.agent import process_agent_message, process_hitl_decision

router = APIRouter(prefix="/api/v1/agent", tags=["Stateful Agent"])


@router.post("/chat", response_model=AgentChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(request: AgentChatRequest):
    """
    Sends a message to the operational agent. Returns either a completed reply
    or an 'interrupted' status if human authorization is required.
    """
    return await process_agent_message(
        message=request.message,
        thread_id=request.thread_id,
    )


@router.post("/decision", response_model=AgentChatResponse, status_code=status.HTTP_200_OK)
async def submit_hitl_decision(request: HITLDecisionRequest):
    """
    Submits an operator decision ('approve' or 'reject') for a paused thread.
    """
    return await process_hitl_decision(
        thread_id=request.thread_id,
        action=request.action,
        feedback=request.feedback,
    )
