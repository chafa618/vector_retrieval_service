
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse
#from vector_retrieval_service.embedding_retriever.models.llms import EmbeddingModel
from vector_retrieval_service.embedding_retriever.models.model_factory import LLMFactory, LanguageModels
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



def get_text_embedding_service(
    text: str,
    requested_model: LanguageModels
) -> TextEmbeddingsResponse:
    embedding_model = LLMFactory.get_model(
        requested_model
    )
    embeddings = embedding_model.get_embedding(text).tolist()

    return TextEmbeddingsResponse(
        embeddings=embeddings,
        engine_used=requested_model
    )

