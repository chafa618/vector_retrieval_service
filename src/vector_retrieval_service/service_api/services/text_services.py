
from ctypes import util
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse
from vector_retrieval_service.embedding_retriever.models.model_factory import LLMFactory, LanguageModels

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
) -> dict[str, float]:

    embedding_model = LLMFactory.get_model(
        requested_model
    )
    texts = [query] + corpus
    embeddings = await asyncer.asyncify(embedding_model.get_embedding)(texts)
    query_embeddings = embeddings[0]
    doc_embeddings = embeddings[1:]
    
    #scores = await asyncer.asyncify(util.dot_score)(query_embeddings, doc_embeddings)
    distances = {
        text_: float(np.linalg.norm(doc_embedding - query_embeddings)) # CHANGE THIS LINE
        for doc_embedding, text_ in zip(doc_embeddings, corpus)
    } 
    return dict(sorted(distances.items(), key=lambda x: x[1], reverse=False))
"""
    doc_score_pairs = list(zip(corpus, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return {
        "query": query,
        "results": [
            {
                "text": doc,
                "score": score
            }
            for doc, score in doc_score_pairs
        ]
    } """