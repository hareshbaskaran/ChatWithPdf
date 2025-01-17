from abc import ABC, abstractmethod
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Optional


class BaseLoader(ABC):
    def __init__(self, doc_path: Optional[str] = None):
        self.doc_path = doc_path

    @abstractmethod
    def get_docs(cls, doc_path: Optional[str] = None) -> List[Document]:
        """
        This Method should be Implemented as Base Document Loader
        :param: Document(s) | DocumentPath
        :return: List of Documents
        """
        pass


class PDFLoader(BaseLoader):
    @classmethod
    def get_docs(cls, doc_path: Optional[str] = None) -> List[Document]:
        """
        This method should be an Implementation of BaseLoader for PDF documents
        :param doc_path:
        :return: Document Loader of that instance
        """

        # Use doc_path from instance or method call
        if not doc_path:
            raise ValueError(
                "doc_path must be provided either during initialization or method call"
            )

        # return PDF loader of this instance
        return PyPDFLoader(doc_path).load()
