from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCCITE_")

    data_dir: Path = Path(".doccite")
    chunk_size: int = Field(600, ge=64)
    chunk_overlap: int = Field(100, ge=0)
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    enable_dense: bool = True
    claim_rerank: bool = False
    quote_threshold: float = Field(0.80, ge=0, le=1)
    claim_partial_threshold: float = Field(0.50, ge=0, le=1)
    claim_supported_threshold: float = Field(0.90, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.claim_partial_threshold > self.claim_supported_threshold:
            raise ValueError("claim_partial_threshold must not exceed claim_supported_threshold")
        return self
