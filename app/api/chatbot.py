from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.chatbot import ChatRequest, ChatResponse
from app.services.context_service import FinancialContextProvider, get_context_provider
from app.services.llm_service import LLMService, llm_service

router = APIRouter(prefix="", tags=["AI Chatbot"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Personal Finance AI Chatbot",
    description="Answer user questions regarding personal finances, spending, budgets, and savings.",
)
async def chat_endpoint(
    payload: ChatRequest,
    context_provider: FinancialContextProvider = Depends(get_context_provider),
    service: LLMService = Depends(lambda: llm_service),
) -> ChatResponse:
    """Processes a user chat message using retrieved financial context and the LLM service."""
    try:
        # Retrieve context for user to ground response in actual figures
        context = context_provider.get_user_context(payload.user_id)

        # Generate response
        response_text, source = service.generate_response(payload.message, context)

        return ChatResponse(response=response_text, source=source)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(exc)}",
        )

