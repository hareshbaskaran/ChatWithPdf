from services.embeddings import HFEmbeddings
from services.llms import GeminiLLMProvider
from services.loaders import PDFLoader
from services.chunkers import RTChunker
from services.vectordbs import FAISSVectorStore
from utils.variables import VECTOR_DB_PATH

from langchain.indexes import index, SQLRecordManager
#### indexes in lancghain
########### run pdf - ingestion ###############
def main():
    collection_name = "test_index"
    namespace = f"FAISS/{collection_name}"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()
    pdf_loader = PDFLoader(doc_path="docs/test_data/test_index2.pdf")
    doc_chunker = RTChunker(docs=pdf_loader.get_docs())
    docs = doc_chunker.split_docs()
    doc_len = len(docs)
    db = FAISSVectorStore(
        docs=docs,
        embeddings=HFEmbeddings().get_embeddings(),
        vector_db_path=VECTOR_DB_PATH,
    )
    db.add_docs_to_vector_db()
    vdb = db.get_vdb()

    idx = index(
        docs,
        record_manager,
        vdb,
        cleanup=None,
        source_id_key="source"
    )
    print(idx)
    if idx['num_skipped'] == doc_len:
        print("duplicate document")



"""    pdf_loader = PDFLoader(doc_path="docs/test_data/new_test.pdf")
    doc_chunker = RTChunker(docs=pdf_loader.get_docs())
    docs = doc_chunker.split_docs()

    db = FAISSVectorStore(
        docs=docs,
        embeddings=HFEmbeddings().get_embeddings(),
        vector_db_path=VECTOR_DB_PATH,
    )
    get_db = db.get_vdb()

    print(get_db.similarity_search("vectors", k=5))"""


if __name__ == "__main__":
    main()
