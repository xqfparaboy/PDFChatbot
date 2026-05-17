from fastapi import APIRouter, HTTPException, status

from app.models import ChatRequest, ChatResponse
from app.services.rag_service import answer_question


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await answer_question(request.question, source=request.source)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
