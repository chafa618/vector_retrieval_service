from pydantic import BaseModel


class TextEmbeddingsResponse(BaseModel):
    embeddings: list[float]
    engine_used: str
    
    
class TextSimiliarityResponse(BaseModel):
    query: str
    results: list[dict[str, float]]