import os
import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from app.schemas.contracts import AgentChatResponse, PendingAction
from app.services.rag import query_rag_pipeline

load_dotenv()
logger = logging.getLogger("uvicorn.error")

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

# --------------------
# 1. Operational Tools
# ---------------------------------------------------------------------------
@tool
def calculate_feed_consumption(animal_type: str, head_count: int, days: int) -> str:
    """Calculates feed requirements based on animal species, head count, and duration."""
    rates = {"rabbit": 0.12, "broiler": 0.15, "layer": 0.11}
    species = animal_type.lower().strip()
    if species not in rates:
        return f"Unknown animal type '{animal_type}'."
    daily_rate = rates[species]
    total_kg = round(head_count * daily_rate * days, 2)
    bags = round(total_kg / 50.0, 1)
    return f"{head_count} {species}s for {days} days require {total_kg} kg (~{bags} 50kg bags)."


@tool
def check_equipment_telemetry(device_id: str) -> str:
    """Retrieves current operating hours, oil level, and alerts for machinery."""
    telemetry_db = {
        "PUMP-TX8": {
            "status": "OPERATIONAL",
            "operating_hours": 248,
            "oil_level": "WARNING_LOW",
            "notes": "Approaching 250h threshold. Service required in 2 hours."
        }
    }
    dev = device_id.upper().strip()
    return str(telemetry_db.get(dev, f"Device {device_id} not registered in telemetry database."))


@tool
async def query_knowledge_base_tool(query: str) -> str:
    """
    Searches the official farm knowledge base, SOPs, and equipment manuals.
    Returns cited, high-confidence documentation.
    """
    res = await query_rag_pipeline(query, top_k=2, min_score=0.20)
    if not res.sources:
        return "No relevant documentation found."
    return f"Retrieved Context (Sources: {res.sources}):\n{res.answer}"


@tool
def dispatch_maintenance_ticket(equipment_id: str, priority: str, parts_required: str) -> str:
    """
    DISPATCHES AN OFFICIAL WORK ORDER to the maintenance team.
    High-stakes action requiring human authorization.
    """
    return (
        f"WORK ORDER CREATED [TICKET-9921]:\n"
        f"- Target Asset: {equipment_id}\n"
        f"- Priority: {priority.upper()}\n"
        f"- Allocated Parts/Specs: {parts_required}\n"
        f"- Status: DISPATCHED TO CENTRAL DEPOT CREW"
    )


tools = [
    calculate_feed_consumption,
    check_equipment_telemetry,
    query_knowledge_base_tool,
    dispatch_maintenance_ticket,
]

# --------------------------
# 2. Resilient Model Binding
# ---------------------------------------------------------------------------
# Primary: Groq (Low latency, verified key, no timeout drops)
primary_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.0,
)

# Backup: Gemini (if Groq ever encounters an issue)
backup_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=gemini_key,
    timeout=30.0,
)

primary_with_tools = primary_llm.bind_tools(tools)
backup_with_tools = backup_llm.bind_tools(tools)

llm_with_tools = primary_with_tools.with_fallbacks([backup_with_tools])

# ------------------------------------------
# 3. State Schema & Selective Interrupt Flow
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)


def human_approval_node(state: AgentState):
    """Breakpoint gate: Graph freezes right before this node executes."""
    return state


def route_with_selective_interrupt(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    sensitive_tools = {"dispatch_maintenance_ticket"}
    for call in last_message.tool_calls:
        if call["name"] in sensitive_tools:
            return "human_approval"
    return "tools"


# -----------------------------------
# 4. Graph Construction & Compilation
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("tools", tool_node)
builder.add_node("human_approval", human_approval_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_with_selective_interrupt,
    {
        "human_approval": "human_approval",
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("human_approval", "tools")
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
agent_graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"],
)

# -----------------------------------------
# 5. Service Interface for FastAPI Handlers
# ---------------------------------------------------------------------------
def _format_reply(message: Any) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


async def process_agent_message(message: str, thread_id: str) -> AgentChatResponse:
    """Executes or resumes an agent session for a given thread_id."""
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in agent_graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="values",
    ):
        pass

    snapshot = agent_graph.get_state(config)

    # Check if execution paused at the HITL gate
    if snapshot.next == ("human_approval",):
        last_message = snapshot.values["messages"][-1]
        pending = [
            PendingAction(tool=call["name"], args=call["args"])
            for call in last_message.tool_calls
        ]
        return AgentChatResponse(
            thread_id=thread_id,
            status="interrupted",
            pending_actions=pending,
            reply="High-stakes action detected. Operator authorization required.",
        )

    last_message = snapshot.values["messages"][-1]
    return AgentChatResponse(
        thread_id=thread_id,
        status="completed",
        reply=_format_reply(last_message),
    )


async def process_hitl_decision(thread_id: str, action: str, feedback: Optional[str] = None) -> AgentChatResponse:
    """Applies human approval or rejection to a paused thread and resumes execution."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)

    if snapshot.next != ("human_approval",):
        return AgentChatResponse(
            thread_id=thread_id,
            status="error",
            reply="Thread is not currently awaiting authorization.",
        )

    last_message = snapshot.values["messages"][-1]

    if action == "approve":
        async for _ in agent_graph.astream(None, config=config, stream_mode="values"):
            pass
    else:
        rejection_text = feedback or "Action rejected by operator."
        rejection_messages = [
            ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                content=f"ACTION REJECTED BY OPERATOR: {rejection_text}",
            )
            for call in last_message.tool_calls
        ]
        agent_graph.update_state(config, {"messages": rejection_messages}, as_node="tools")
        async for _ in agent_graph.astream(None, config=config, stream_mode="values"):
            pass

    updated_snapshot = agent_graph.get_state(config)
    final_message = updated_snapshot.values["messages"][-1]
    return AgentChatResponse(
        thread_id=thread_id,
        status="completed",
        reply=_format_reply(final_message),
    )
