from fastapi import APIRouter

from vector_retrieval_service.embedding_retriever.models.model_factory import LanguageModels
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse
from vector_retrieval_service.service_api.services.text_services import get_text_embedding_service, get_text_similarity_service


retrival_service = APIRouter(prefix="/text_processing", tags=['Text Processing'])



@retrival_service.get("/get_text_embeddings", description="Get embeddings for a given text")
async def get_text_embeddings(
    text: str,
    requested_model: LanguageModels
) -> TextEmbeddingsResponse:

    embeddings = await get_text_embedding_service(text, requested_model=requested_model),
    return embeddings



@retrival_service.post("/get_text_similarity", description="Compute text similiarity between a given query and a list of texts")
async def get_text_similarity(
    query: str,
    corpus: list[str],
    requested_model: LanguageModels
) -> dict[str, float]:

    return await get_text_similarity_service(query, corpus, requested_model=requested_model)
