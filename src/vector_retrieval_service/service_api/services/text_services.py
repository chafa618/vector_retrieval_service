
from ctypes import util
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse, TextSimilarityIndex, TextSimilarityResponse
from vector_retrieval_service.embedding_retriever.models.model_factory import LLMFactory, LanguageModels
from sklearn.metrics.pairwise import cosine_similarity

import asyncer
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


async def get_text_embedding_service(
    text: str,
    requested_model: LanguageModels
) -> TextEmbeddingsResponse:

    embedding_model = LLMFactory.get_model(
        requested_model
    )

    embeddings = await asyncer.asyncify(embedding_model.get_embedding)(text)

    return TextEmbeddingsResponse(
        embeddings=embeddings.tolist(),
        engine_used=requested_model
    )


async def get_text_similarity_service(
    query: str,
    corpus: list[str],
    requested_model: LanguageModels
) -> TextSimilarityResponse:

        embedding_model = LLMFactory.get_model(requested_model)
        texts = [query] + corpus
        embeddings = await asyncer.asyncify(embedding_model.get_embedding)(texts)
        query_embedding = embeddings[0].reshape(1, -1)
        corpus_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        doc_score_pairs = list(zip(corpus, similarities, range(len(corpus))))
        sorted_doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        res = [
                TextSimilarityIndex(
                    original_text_id=index_,
                    text=doc,
                    distance_to_query=score
                )
                for doc, score, index_ in sorted_doc_score_pairs]
        
        similarity_response = TextSimilarityResponse(
            query=query,
            results=res
        )

        return similarity_response
