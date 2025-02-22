

from enum import StrEnum, auto
from functools import lru_cache
from typing import Any

from vector_retrieval_service.embedding_retriever.models.llms import EmbeddingModel





class LanguageModels(StrEnum):
    mini_lm = auto()
    multi_qa = auto()


class LLMFactory:
    def __init__(self):
        pass

    @staticmethod
    @lru_cache
    def get_model(model_type: LanguageModels) -> Any:
        mini_lm = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
        match model_type:
            case LanguageModels.mini_lm:
                return mini_lm
            case LanguageModels.multi_qa:
                multi_qa = EmbeddingModel("sentence-transformers/multi-qa-mpnet-base-dot-v1")
                return multi_qa
            case _:
                return LanguageModels.mini_lm