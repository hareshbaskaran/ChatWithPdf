import os
from abc import ABC, abstractmethod
from typing import Any

from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.utils.loggers import logger


### Works only for Local Storage - VectorDB Indexes ###
class BaseVectorStore(ABC):
    def __init__(self, embeddings: Embeddings, vector_db_path: str):
        """
        Initialize the BaseVectorStore class
        :param embeddings: Embedding model to be used for document embeddings
        :param vector_db_path: Path where the vector database is stored
        """
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path

    @abstractmethod
    def get_vdb(self) -> Any:
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
    def __init__(self, embeddings: Embeddings, vector_db_path: str):
        """
        Initialize the FAISSVectorStore class.
        :param embeddings: Embedding model to be used for document embeddings.
        :param vector_db_path: Path where the vector database is stored.
        """
        super().__init__(embeddings=embeddings, vector_db_path=vector_db_path)
        self.ensure_vector_db_exists()

    def ensure_vector_db_exists(self):
        """
        Ensure the vector database exists, otherwise create it with a dummy document.
        :return: None
        """
        if not self.vector_db_exists():
            logger.debug(
                "Vector database not found. Initializing with a dummy document."
            )
            self.initialize_with_dummy_document()

    def vector_db_exists(self) -> bool:
        """
        Check if the vector database exists at the given path.
        :return: True if the vector database exists, False otherwise.
        """
        return os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path)

    def initialize_with_dummy_document(self):
        """
        Initialize the vector database with a dummy document.
        :return: None
        """
        dummy_doc = [
            Document(
                page_content="This is a dummy document to initialize the vector store."
            )
        ]
        faiss_db = FAISS.from_documents(documents=dummy_doc, embedding=self.embeddings)
        faiss_db.save_local(self.vector_db_path)
        logger.debug(
            f"Vector database initialized at {self.vector_db_path} with a dummy document."
        )

    def get_vdb(self) -> FAISS:
        """
        Retrieve the FAISS vector database from the local path.
        If the vector database does not exist, initialize it with a dummy document.
        :return: FAISS vector database instance loaded from the local storage.
        """
        try:
            return FAISS.load_local(
                folder_path=self.vector_db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning(
                f"Error loading vector database: {e}. Reinitializing with a dummy document."
            )
            self.initialize_with_dummy_document()
            return FAISS.load_local(
                folder_path=self.vector_db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def add_docs_to_vector_db(self, docs):
        """
        Add documents to the FAISS vector database.
        :param docs: List of documents to be added to the vector store.
        :return: None
        """
        vector_store = self.get_vdb()
        vector_store.index()
        vector_store.add_documents(documents=docs, embedding=self.embeddings)
        vector_store.save_local(self.vector_db_path)
        logger.debug("Documents added to the vector database.")

    def get_vdb_as_retriever(self):
        """
        Retrieve the FAISS vector database as a retriever.
        :return: A retriever object based on the vector database.
        """
        return self.get_vdb().as_retriever()


class ChromaVectorStore:
    def __init__(
        self,
        embeddings: Embeddings,
        vector_db_path: str,
        collection_name: str = "default_collection",
    ):
        """
        Initialize the ChromaVectorStore class.
        :param embeddings: Embedding model to be used for document embeddings.
        :param vector_db_path: Path where the Chroma database is stored.
        :param collection_name: Name of the Chroma collection.
        """
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name

        # Ensure Chroma database exists
        self.ensure_vector_db_exists()

    def ensure_vector_db_exists(self):
        """
        Ensure the Chroma database exists, otherwise create it with a dummy document.
        """
        if not self.vector_db_exists():
            logger.debug(
                "Chroma database not found. Initializing with a dummy document."
            )
            self.initialize_with_dummy_document()

    def vector_db_exists(self) -> bool:
        """
        Check if the Chroma database exists at the given path.
        :return: True if the database exists, False otherwise.
        """
        # Chroma stores its data in the folder specified by `persist_directory`
        return os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path)

    def initialize_with_dummy_document(self):
        """
        Initialize the Chroma database with a dummy document.
        """
        dummy_doc = [
            Document(
                page_content="This is a dummy document to initialize the vector store."
            )
        ]
        chroma_db = Chroma.from_documents(
            documents=dummy_doc,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path,
            collection_name=self.collection_name,
        )
        chroma_db.persist()
        logger.debug(
            f"Chroma database initialized at {self.vector_db_path} with a dummy document."
        )

    def get_vdb(self) -> Chroma:
        """
        Retrieve the Chroma database. If the database does not exist, initialize it with a dummy document.
        :return: Chroma database instance.
        """
        try:
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path,
                collection_name=self.collection_name,
            )
        except Exception as e:
            logger.warning(
                f"Error loading Chroma database: {e}. Reinitializing with a dummy document."
            )
            self.initialize_with_dummy_document()
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path,
                collection_name=self.collection_name,
            )

    def add_docs_to_vector_db(self, docs):
        """
        Add documents to the Chroma database.
        :param docs: List of documents to be added to the vector store.
        """
        chroma_db = self.get_vdb()
        chroma_db.add_documents(documents=docs)
        chroma_db.persist()
        logger.debug("Documents added to the Chroma database.")

    def get_vdb_as_retriever(self):
        """
        Retrieve the Chroma database as a retriever.
        :return: A retriever object based on the database.
        """
        return self.get_vdb().as_retriever()
