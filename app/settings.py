from pydantic_settings import BaseSettings
from typing import Optional, Any, Annotated
class Settings(BaseSettings):
    app_name: str = "Chat With Pdf"
    llm_provider: str = "Gemini"
    llm_model : str = "gemini-1.5-flash"
    vector_db_store: str = "Chroma"
    embeddings_provider: str = "HuggingFaceEmbeddings"
    embeddings_model: Optional[str] = None

    # Design Level Configurations
    chunker_type: str = "RTChunker"
    loader_type: str = "PDFLoader"

    class Config:
        env_file = ".env"






