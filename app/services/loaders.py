from abc import ABC, abstractmethod
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Optional


class BaseLoader(ABC):
    @abstractmethod
    def get_docs(self, doc_path: Optional[str] = None) -> List[Document]:
        pass


class PDFLoader(BaseLoader):

    def get_docs(self, doc_path: Optional[str] = None) -> List[Document]:
        final_path = doc_path

        # Ensure we have a path to work with
        if not final_path:
            raise ValueError(
                "doc_path must be provided either during initialization or method call"
            )

        # Load PDF file from the given doc_path
        loader = PyPDFLoader(final_path)
        return loader.load()
