from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    app_name: str = "PDF Chatbot"
    upload_dir: Path = Path("uploaded_pdfs")

    qdrant_url: str | None = None
    qdrant_path: Path = Path("qdrant_storage")
    qdrant_collection: str = "pdf_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 500
    chunk_overlap: int = 100
    search_top_k: int = 5

    llm_provider: str = Field(default="ollama", pattern="^(ollama|openai)$")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "OPEN_AI_KEY",
            "OPEN_API_KEY",
            "openai_api_key",
        ),
    )
    openai_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
