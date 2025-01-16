from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    @abstractmethod
    def split_docs(self, docs: List[Document]) -> List[Document]:
        """
        This method should be implemented as a Base Chunker Interface
        :returns Split documents into chunks.
        """
        pass


class RTChunker(BaseChunker):
    def split_docs(self, docs: List[Document]) -> List[Document]:
        """
        This method should be implemented as a Recursive Character Text Splitter
        :param docs:
        :return: Split Documents into Chunks
        """
        if not docs:
            raise ValueError("docs parameter cannot be empty")

        if not isinstance(docs, list) or not all(
            isinstance(doc, Document) for doc in docs
        ):
            raise TypeError("docs must be a list of Document objects")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        return splitter.split_documents(docs)
