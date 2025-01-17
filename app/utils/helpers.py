from fastapi import File
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.services.embeddings import HFEmbeddings
from app.services.loaders import PDFLoader
from app.services.chunkers import RTChunker
from app.services.vectordbs import FAISSVectorStore
from app.utils.variables import VECTOR_DB_PATH, SQL_MANAGER_NAMESPACE, SQLITE_DB_URL
from langchain.indexes import index, SQLRecordManager
import os

from langchain.indexes import index, SQLRecordManager


def instantiate_record_manager(namespace:str,db_url:str):
    record_manager = SQLRecordManager(
        namespace, db_url=db_url
    )
    record_manager.create_schema()
    return record_manager


async def handle_temp_dir(file):
    os.makedirs("docs", exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    with open(f"docs/{file.filename}", "wb") as temp_file:
        temp_file.write(await file.read())

    return f"docs/{file.filename}"

def process_doc(doc_path):
    docs = PDFLoader(doc_path).get_docs()
    chunked_docs = RTChunker(
        docs=docs
    ).split_docs()
    return chunked_docs

def get_vdb(docs):
    return FAISSVectorStore(
        docs=docs,
        embeddings=HFEmbeddings().get_embeddings(),
        vector_db_path=VECTOR_DB_PATH
    ).get_vdb()

def is_duplicate(idx: index,docs):
    """
    custom logic to find duplicates
    :param idx:
    :param docs:
    :return:
    """
    if idx['num_skipped'] == len(docs):
        return True
    return False








