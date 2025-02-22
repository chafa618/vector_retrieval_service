from pydantic import BaseModel


class TextEmbeddingsResponse(BaseModel):
    embeddings: list[float]
    engine_used: str