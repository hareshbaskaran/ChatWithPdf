from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from abc import ABC, abstractmethod


### Works only for Local Storage - VectorDB Indexes ###
class BaseVectorStore(ABC):
    def __init__(
        self, embeddings: Embeddings, vector_db_path: str
    ):
        """
        Initialize the BaseVectorStore class
        :param embeddings: Embedding model to be used for document embeddings
        :param vector_db_path: Path where the vector database is stored
        """
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path

    @abstractmethod
    def get_vdb(self) -> FAISS:
        """
        Abstract method to retrieve the vector database instance
        :return: FAISS vector database instance
        """
        pass

    @abstractmethod
    def get_vdb_as_retriever(self) -> BaseRetriever:
        """
        Abstract method to retrieve the vector database as a retriever
        :return: A retriever based on the vector database
        """
        pass


class FAISSVectorStore(BaseVectorStore):
    def __init__(
        self, embeddings: Embeddings, vector_db_path: str
    ):
        """
        Initialize the FAISSVectorStore class
        :param embeddings: Embedding model to be used for document embeddings
        :param vector_db_path: Path where the vector database is stored
        """
        super().__init__(
            embeddings=embeddings, vector_db_path=vector_db_path
        )

    def add_docs_to_vector_db(self, docs):
        """
        Add documents to the FAISS vector database
        :param docs: List of documents to be added to the vector store
        :return: None
        """
        # Get the vector database instance
        vector_store = self.get_vdb()

        ### TODO: verify indexing with SQLRecordManager ###
        # Index the vector store
        vector_store.index()

        # Add documents to the vector store with the provided embedding
        vector_store.add_documents(documents=docs, embedding=self.embeddings)

        # Save the vector store locally at the provided path
        vector_store.save_local(self.vector_db_path)

        return None

    def get_vdb(self) -> FAISS:
        """
        Retrieve the FAISS vector database from the local path
        :return: FAISS vector database instance loaded from the local storage
        """
        return FAISS.load_local(
            folder_path=self.vector_db_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def get_vdb_as_retriever(self):
        """
        Retrieve the FAISS vector database as a retriever
        :return: A retriever object based on the vector database
        """
        db = self.get_vdb()
        return db.as_retriever()
