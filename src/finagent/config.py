"""Central configuration. Values come from environment variables or a local .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Every field can be overridden with a ``FINAGENT_`` env var."""

    model_config = SettingsConfigDict(
        env_prefix="FINAGENT_", env_file=".env", extra="ignore", frozen=True
    )

    google_api_key: SecretStr | None = None

    # Model names change often; keep them in config, never hard-code them in logic.
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    # Cost and rate control (free-tier limits are low; tune to your quota).
    requests_per_minute: int = 10
    max_retries: int = 5
    retry_base_delay_s: float = 2.0

    cache_dir: Path = Path(".cache/llm")
    data_dir: Path = Path("data/raw")
    runs_dir: Path = Path("runs")
    pdf_dir: Path = Path("data/raw/financebench_pdfs")
    index_dir: Path = Path(".cache/index")

    # Ingestion and retrieval defaults (tuned by experiment, not by guesswork).
    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 200
    retrieval_k: int = 5


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings()
