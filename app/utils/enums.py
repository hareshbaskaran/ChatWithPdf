from enum import Enum

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma

# define custom services
from app.services.embeddings import HFEmbeddings
from app.services.llms import GeminiLLMProvider
from app.services.vectordbs import ChromaVectorStore, FAISSVectorStore

class LLMModels(Enum):
    GEMINI = GeminiLLMProvider
class VectorStores(Enum):
    CHROMA = Chroma
    FAISS = FAISS
class Embeddings(Enum):
    HUGGINGFACE = HuggingFaceEmbeddings

settings = {
    "LLM" : LLMModels.GEMINI,
    "VECTOR_STORE" : VectorStores.CHROMA,
    "EMBEDDINGS" : Embeddings.HUGGINGFACE,
}
