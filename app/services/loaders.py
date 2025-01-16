from abc import ABC, abstractmethod, abstractproperty
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional, Any


class BaseEmbeddings(ABC):

    @abstractmethod
    def get_embeddings(self) -> Any:
        """
        This method shoudl be implemented by Embedding Model Providers to return
        an instance of a specific EmbeddingModel class
        """

        pass


class HFEmbeddings(BaseEmbeddings):

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        This method return an instance of HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings()