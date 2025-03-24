from enum import Enum

# define custom services
from services.embeddings import HFEmbeddings
from services.llms import GeminiLLMProvider
from services.vectordbs import ChromaVectorStore, FAISSVectorStore


class LLMModels(Enum):
    """LLM Model Providers"""

    GEMINI = GeminiLLMProvider


class VectorStores(Enum):
    """Vector Store Providers"""

    CHROMA = ChromaVectorStore
    FAISS = FAISSVectorStore


class Embeddings(Enum):
    """Embedding Model Providers"""

    HUGGINGFACE = HFEmbeddings
