from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from abc import ABC, abstractmethod
from typing import List, Optional

### Works only for Local Storage - VectorDB Indexes ###
#todo: comment and black style
class BaseVectorStore(ABC):

    # Instantiate vector_db_name, vector_db_path :
    # pass by vector_db_name

    @abstractmethod
    def add_docs_to_vector_db(
            self,
            docs: List[Document],
            embeddings: Embeddings,
            vector_db_path: str,
            vector_store: None
    ):
        pass


class FAISSVectorStore(BaseVectorStore):

    def add_docs_to_vector_db(
            self,
            docs: Optional[List[Document]],
            embeddings: Embeddings,
            vector_db_path: str,
            vector_store: None
    ):
        if not docs:
            return FAISS.load_local(
                vector_db_path, embeddings, allow_dangerous_deserialization=True
            )


        ### Handle Vector Store
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
        self.vector_store.save_local(vector_db_path)

        return self.vector_store



