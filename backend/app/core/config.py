"""
Application configuration
"""
import os
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config=SettingsConfigDict(
        env_file = ".env",
        case_sensitive = True,
        extra="ignore",
    )

    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Multimodal Document Chat"
    
    # Database
    DATABASE_URL: str = "postgresql://docuser:docpass@localhost:5432/docdb"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Gemini
    USE_GEMINI: bool = False
    GEMINI_MODEL: str = "gemini-2.5-pro"
    GEMINI_EMBEDDING_MODEL: str = "text-multilingual-embedding-002"
    GEMINI_MODEL_LOCATION: str = "us-central1"
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_APPLICATION_CREDENTIALS: str
    
    # Upload Settings
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    
    # Vector Store Settings
    EMBEDDING_DIMENSION: int = 384  # Shouldn't be change after database created, because embeddings(vector) column of the corresponding table cannot be changed in between executions.  Also shouldn't be more than 384, because Huggingface embedding model does only support up to 384.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5

    @model_validator(mode="after")
    def ensure_gcp_variables(self):
        if self.USE_GEMINI:
            required_vars = [
                "GOOGLE_CLOUD_PROJECT",
                "GEMINI_MODEL_LOCATION",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ]
            for var in required_vars:
                if not getattr(self, var):
                    raise ValueError(f"{var} must be set when USE_GEMINI is True")

            # Set GCP default credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_APPLICATION_CREDENTIALS
        return self

    @field_validator("EMBEDDING_DIMENSION")
    @classmethod
    def validate_embedding_dimension(cls, v):
        if v > 384:
            raise ValueError("EMBEDDING_DIMENSION cannot be more than 384.")
        return v

settings = Settings()
