from enum import Enum

# define custom services
from app.services.embeddings import HFEmbeddings
from app.services.llms import GeminiLLMProvider
from app.services.vectordbs import ChromaVectorStore, FAISSVectorStore


class LLMModels(Enum):
    GEMINI = GeminiLLMProvider


class VectorStores(Enum):
    CHROMA = ChromaVectorStore
    FAISS = FAISSVectorStore


class Embeddings(Enum):
    HUGGINGFACE = HFEmbeddings


settings = {
    "LLM": LLMModels.GEMINI,
    "VECTOR_STORE": VectorStores.CHROMA,
    "EMBEDDINGS": Embeddings.HUGGINGFACE,
}

"""## todo : add more modularization -> hook with config file / .env file
provider_store = {
    ### LLM MODELS 
    "GEMINI" : LLMModels.GEMINI,
    "OPENAI" : None,
    
    ### Vector Stores
    "CHROMA" : VectorStores.CHROMA,
    "FAISS" : VectorStores.FAISS,
    
    ### Embeddings 
    "HUGGINGFACE" : Embeddings.HUGGINGFACE
}

## VectorStores object is not callable error """
