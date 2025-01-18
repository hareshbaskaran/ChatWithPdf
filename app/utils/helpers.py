from typing import Any
from app.services.embeddings import HFEmbeddings
from app.services.loaders import PDFLoader
from app.services.chunkers import RTChunker
from app.services.vectordbs import FAISSVectorStore, ChromaVectorStore
from langchain.vectorstores import Chroma
from app.utils.variables import VECTOR_DB_PATH, SQL_MANAGER_NAMESPACE, SQLITE_DB_URL
from langchain.indexes import index, SQLRecordManager
import os


class PDFIngest:
    @staticmethod
    def instantiate_record_manager() -> SQLRecordManager:
        """
        Initializes and creates the schema for SQLRecordManager
        :return: Initialized SQLRecordManager instance
        """
        record_manager = SQLRecordManager(
            namespace=SQL_MANAGER_NAMESPACE, db_url=SQLITE_DB_URL
        )
        record_manager.create_schema()
        return record_manager

    async def handle_temp_dir(self, file) -> str:
        """
        Handles saving the uploaded file to a temporary directory.
        :return: The path where the file is saved
        """
        os.makedirs("docs", exist_ok=True)
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        with open(f"docs/{file.filename}", "wb") as temp_file:
            temp_file.write(await file.read())
        return f"docs/{file.filename}"

    @staticmethod
    def process_doc(doc_path: str):
        """
        Processes the PDF document: loads and chunks the document
        :param doc_path: The path to the PDF file
        :return: Chunked document list
        """
        docs = PDFLoader.get_docs(doc_path=doc_path)
        chunked_docs = RTChunker(docs=docs).split_docs()
        return chunked_docs

    @classmethod
    def get_vector_store(cls) -> ChromaVectorStore:
        """
        Returns a FAISSVectorStore object with embeddings
        :return: FAISSVectorStore instance
        """
        return ChromaVectorStore(
            embeddings=HFEmbeddings.get_embeddings(),
            vector_db_path=VECTOR_DB_PATH,
        )

    @staticmethod
    def is_duplicate(idx: index, docs: list) -> bool:
        """
        Custom logic to check if the document is a duplicate
        :param idx: The index object
        :param docs: List of documents
        :return: Boolean indicating if the document is a duplicate
        """
        if idx["num_skipped"] == len(docs):
            return True
        return False


### Retrieval Chain Methods
## todo : add pages as additional meta data


def parse_to_pydantic(result) -> Any:
    sources = [
        doc.metadata.get("source") + ",  page no : " + str(doc.metadata.get("page"))
        for doc in result["source_documents"]
    ]
    parsed_result = {"response": result["result"], "citations": list(set(sources))}
    return parsed_result
