from backend.api.chat import router as chat_router
from backend.api.documents import router as documents_router
from backend.api.knowledge_base import router as knowledge_base_router

__all__ = ["chat_router", "documents_router", "knowledge_base_router"]
