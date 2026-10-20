from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator
from typing_extensions import Self


# ==============
# 1. Extraction Contracts (Phase 1 Engine)
# ===========================================================================
class PaymentStatus(str, Enum):
    PAID = "PAID"
    PENDING = "PENDING"
    OVERDUE = "OVERDUE"
    UNSPECIFIED = "UNSPECIFIED"


class LineItem(BaseModel):
    description: str = Field(
        ...,
        validation_alias=AliasChoices("description", "item_name", "product_name"),
        description="Standardized name or description of the product/service",
    )
    quantity: float = Field(..., gt=0, description="Quantity ordered")
    unit_price: float = Field(..., ge=0, description="Cost per unit")
    total_price: float = Field(..., ge=0, description="Total line cost (quantity * unit_price)")

    @model_validator(mode="after")
    def verify_line_math(self) -> Self:
        expected = round(self.quantity * self.unit_price, 2)
        actual = round(self.total_price, 2)
        if abs(expected - actual) > 0.05:
            raise ValueError(
                f"Math mismatch for '{self.description}': total_price ({actual}) != {self.quantity} * {self.unit_price} ({expected})"
            )
        return self


class PurchaseOrder(BaseModel):
    vendor_name: str = Field(..., description="Company or individual selling the goods")
    buyer_name: Optional[str] = Field(None, description="Purchaser name if specified")
    order_date: Optional[date] = Field(None, description="ISO date (YYYY-MM-DD) of transaction")
    currency: str = Field(default="USD", description="3-letter ISO currency code")
    items: List[LineItem] = Field(..., min_length=1, description="List of items purchased")
    payment_status: PaymentStatus = Field(
        default=PaymentStatus.UNSPECIFIED,
        description="PAID, PENDING, OVERDUE, or UNSPECIFIED",
    )
    grand_total: float = Field(..., description="Calculated sum of all line item totals")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        clean = v.strip().upper()
        if len(clean) != 3:
            raise ValueError(f"Currency must be a 3-letter ISO code, received: {clean}")
        return clean

    @model_validator(mode="after")
    def verify_grand_total(self) -> Self:
        calculated_sum = round(sum(item.total_price for item in self.items), 2)
        if abs(calculated_sum - round(self.grand_total, 2)) > 0.05:
            raise ValueError(
                f"Grand total mismatch: grand_total is {self.grand_total}, but line sum is {calculated_sum}"
            )
        return self


class ExtractionRequest(BaseModel):
    raw_text: str = Field(..., min_length=5, description="Unstructured invoice/order text to parse")


class ExtractionResponse(BaseModel):
    status: str = "success"
    data: PurchaseOrder


# ==================
# 2. Knowledge Base & RAG Contracts (Phase 2 Engine)
# ===========================================================================
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User search query or question")
    top_k: int = Field(default=3, ge=1, le=10, description="Max passages to retrieve")
    min_score: float = Field(default=0.20, ge=0.0, le=1.0, description="Relevance cutoff threshold")


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    cached: bool = False


class DocumentIngestItem(BaseModel):
    content: str = Field(..., min_length=10, description="Raw text of the document chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (doc_id, category, etc.)")


class DocumentIngestRequest(BaseModel):
    documents: List[DocumentIngestItem] = Field(..., min_length=1, description="Batch of documents to ingest")


class IngestJobResponse(BaseModel):
    job_id: str
    status: str = "processing"
    total_documents: int
    message: str


# ===========
# 3. Stateful Agent Contracts (Phase 3 Engine)
# ===========================================================================
class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message to the operational agent")
    thread_id: str = Field(default="session-default", description="Conversation session thread ID")


class PendingAction(BaseModel):
    tool: str
    args: Dict[str, Any]


class AgentChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" | "interrupted"
    reply: Optional[str] = None
    pending_actions: Optional[List[PendingAction]] = None


class HITLDecisionRequest(BaseModel):
    thread_id: str = Field(..., description="The conversation thread currently paused at a breakpoint")
    action: str = Field(..., pattern="^(approve|reject)$", description="'approve' or 'reject'")
    feedback: Optional[str] = Field(None, description="Optional explanation or instructions if rejected")


# ==========
# 4. Service Health Contract
# ===========================================================================
class HealthResponse(BaseModel):
    status: str
    postgres: str
    redis: str
