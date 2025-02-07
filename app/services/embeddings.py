from abc import ABC, abstractmethod
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings


class BaseEmbeddings(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        """
        This method should be implemented as Base Embedding Model Provider
        :return: an instance of a specific EmbeddingModel class
        """
        pass


class HFEmbeddings(BaseEmbeddings):
    @classmethod
    def get_embeddings(cls) -> HuggingFaceEmbeddings:
        """
        This method should be Implementation as a HuggingFaceEmbedding Provider
        :return: an instance of HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings()
