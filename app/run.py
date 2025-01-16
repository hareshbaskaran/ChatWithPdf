from services.embeddings import HFEmbeddings
from services.llms import GeminiLLMProvider
from services.loaders import PDFLoader
from services.chunkers import RTChunker
from services.vectordbs import FAISSVectorStore
from utils.variables import VECTOR_DB_PATH


########### run pdf - ingestion ###############
def main():
    pdf_loader = PDFLoader(doc_path="docs/test_data/test.pdf")
    doc_chunker = RTChunker(docs=pdf_loader.get_docs())
    docs = doc_chunker.split_docs()

    db = FAISSVectorStore(
        docs=docs,
        embeddings=HFEmbeddings().get_embeddings(),
        vector_db_path=VECTOR_DB_PATH,
    )
    get_db = db.get_vdb()

    print(get_db.similarity_search("vectors", k=5))


if __name__ == "__main__":
    main()
