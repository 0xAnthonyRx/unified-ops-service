from fastapi import APIRouter, HTTPException, status
from app.schemas.contracts import ExtractionRequest, ExtractionResponse
from app.services.extractor import extract_purchase_order_data

router = APIRouter(prefix="/api/v1/extract", tags=["Deterministic Extraction"])


@router.post("", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_order(request: ExtractionRequest):
    """
    Parses unstructured text into a validated, mathematically consistent PurchaseOrder.
    """
    try:
        po = await extract_purchase_order_data(request.raw_text)
        return ExtractionResponse(status="success", data=po)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema extraction validation failed: {str(e)}",
        )
