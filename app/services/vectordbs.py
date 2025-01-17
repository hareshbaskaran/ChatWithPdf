from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from abc import ABC, abstractmethod
from typing import List


### Works only for Local Storage - VectorDB Indexes ###
class BaseVectorStore(ABC):
    def __init__(
        self, docs: List[Document], embeddings: Embeddings, vector_db_path: str
    ):
        self.docs = docs
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path

    # Instantiate vector_db_name, vector_db_path :
    # pass by vector_db_name

    @abstractmethod
    def get_vdb(self) -> FAISS:
        pass

    @abstractmethod
    def add_docs_to_vector_db(self) -> None:
        pass

    @abstractmethod
    def get_vdb_as_retriever(self) -> BaseRetriever:
        pass


class FAISSVectorStore(BaseVectorStore):
    def __init__(
        self, docs: List[Document], embeddings: Embeddings, vector_db_path: str
    ):
        super().__init__(
            docs=docs, embeddings=embeddings, vector_db_path=vector_db_path
        )

    def add_docs_to_vector_db(self):
        vector_store = self.get_vdb()

        vector_store.index()

        # Add documents to the vector store
        vector_store.add_documents(documents=self.docs, embedding=self.embeddings)

        # Save vector store if path is provided
        vector_store.save_local(self.vector_db_path)

        return None

    def get_vdb(self) -> FAISS:
        return FAISS.load_local(
            folder_path=self.vector_db_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def get_vdb_as_retriever(self):
        db = self.get_vdb()
        return db.as_retriever()
