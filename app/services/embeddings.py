from abc import ABC, abstractmethod, abstractproperty
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional, Any


class BaseEmbeddings(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        """
        This method should be implemented as Base Embedding Model Provider
        :return: an instance of a specific EmbeddingModel class
        """

        pass


class HFEmbeddings(BaseEmbeddings):
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        This method should be Implementation as a HUggingFaceEmbedding Provider
        :return: an instance of HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings()
