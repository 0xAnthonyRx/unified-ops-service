import os
import logging
from dotenv import load_dotenv
import instructor
import litellm
from litellm import acompletion
from app.schemas.contracts import PurchaseOrder

# Force fresh environment variables from .env
load_dotenv(override=True)
logger = logging.getLogger("uvicorn.error")

# Explicitly bind provider keys so LiteLLM fallbacks never default to stale shell memory
litellm.groq_key = os.getenv("GROQ_API_KEY")
litellm.gemini_key = os.getenv("GEMINI_API_KEY")

# Async Instructor client patched over LiteLLM acompletion
client = instructor.from_litellm(acompletion)

PRIMARY_MODEL = os.getenv("EXTRACTION_MODEL", "gemini/gemini-3.6-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_EXTRACTION_MODEL", "groq/openai/gpt-oss-120b")


async def extract_purchase_order_data(raw_text: str) -> PurchaseOrder:
    """
    Extracts structured, validated PurchaseOrder data from unstructured text.
    Enforces cross-field line math, grand total sum, and currency normalization.
    """
    try:
        response: PurchaseOrder = await client.chat.completions.create(
            model=PRIMARY_MODEL,
            fallbacks=[FALLBACK_MODEL],
            response_model=PurchaseOrder,
            max_retries=3,  # Schema validation self-healing attempts
            num_retries=2,  # Network & 503 retry attempts before fallback
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise extraction pipeline. "
                        "Extract purchase order details from the provided text. "
                        "Calculate item totals if they are missing or implied."
                    ),
                },
                {
                    "role": "user",
                    "content": raw_text,
                },
            ],
        )
        return response
    except Exception as e:
        logger.error(f"Deterministic extraction failed after retries: {e}")
        raise
