from fastapi import APIRouter

from vector_retrieval_service.embedding_retriever.models.model_factory import LanguageModels
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse
from vector_retrieval_service.service_api.services.retrival_service import get_text_embedding_service


retrival_service = APIRouter(prefix="/external_services")



@retrival_service.get("/get_text_embeddings", description="Get embeddings for a given text")
def get_text_embeddings(text: str,
                              requested_model: LanguageModels
    ) -> TextEmbeddingsResponse:
    embeddings = get_text_embedding_service(text, requested_model=requested_model),
    return embeddings
