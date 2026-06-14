import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # File Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
    
    # Text Chunking Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Model Configurations
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = "llama3"

    class Config:
        env_file = ".env"

settings = Settings()