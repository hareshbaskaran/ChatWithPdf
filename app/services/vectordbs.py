from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVectorStore(ABC):
    def __init__(self, vector_db_name: str):
        self.vector_db_name = vector_db_name

    @abstractmethod
    def add_docs_to_vector_db(
            self,
            docs: List[Document],
            embeddings: Embeddings,
            vector_db_path: Optional[str] = None,
    ):
        pass


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, vector_db_name: str = "faiss_index"):
        super().__init__(vector_db_name)
        self.vector_store = None

    def add_docs_to_vector_db(
            self,
            docs: List[Document],
            embeddings: Embeddings,
            vector_db_path: Optional[str] = None,
    ):
        if not docs:
            raise ValueError("Document list cannot be empty")

        # Load existing vector store or create a new one
        try:
            self.vector_store = FAISS.load_local(
                vector_db_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Loading vector store failed: {e}. Creating a new one.")
            self.vector_store = FAISS.from_documents(docs, embeddings)

        # Add documents to the vector store
        self.vector_store.add_documents(documents=docs, embedding=embeddings)

        # Save vector store if path is provided
        if vector_db_path:
            self.vector_store.save_local(vector_db_path)

        return self.vector_store

