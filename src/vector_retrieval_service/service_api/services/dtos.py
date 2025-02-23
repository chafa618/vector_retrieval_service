from typing import Any
from pydantic import BaseModel


class TextEmbeddingsResponse(BaseModel):
    embeddings: list[float]
    engine_used: str
    

class TextSimilarityIndex(BaseModel):
    original_text_id: int
    text: str
    distance_to_query: float

class TextSimilarityResponse(BaseModel):
    query: str
    results: list[TextSimilarityIndex]