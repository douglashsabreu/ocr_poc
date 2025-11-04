from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    datalab_api_key: str = Field(..., alias="DATALAB_API_KEY")
    datalab_api_base: str = Field(
        "https://www.datalab.to/api/v1", alias="DATALAB_API_BASE"
    )
    datalab_model_name: str = Field("chandra", alias="DATALAB_MODEL_NAME")

    inference_method: Literal["vllm", "hf"] = Field(
        "vllm", alias="CHANDRA_INFERENCE_METHOD"
    )
    pipeline_mode: Literal["chandra", "datalab_api", "openai_api", "gdocai"] = Field(
        "datalab_api", alias="PIPELINE_MODE"
    )
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5-mini", alias="OPENAI_MODEL")
    openai_max_tokens: int = Field(2048, alias="OPENAI_MAX_TOKENS")
    images_dir: Path = Field(Path("images_example"), alias="IMAGES_DIR")
    output_dir: Path = Field(Path("outputs"), alias="OUTPUT_DIR")

    include_images: bool = Field(True, alias="INCLUDE_IMAGES")
    include_headers_footers: bool = Field(False, alias="INCLUDE_HEADERS_FOOTERS")
    max_output_tokens: int | None = Field(None, alias="MAX_OUTPUT_TOKENS")
    max_workers: int | None = Field(None, alias="MAX_WORKERS")
    max_retries: int | None = Field(None, alias="MAX_RETRIES")
    api_page_range: str | None = Field(None, alias="API_PAGE_RANGE")
    api_max_pages: int | None = Field(None, alias="API_MAX_PAGES")
    api_skip_cache: bool = Field(False, alias="API_SKIP_CACHE")
    api_langs: str | None = Field(None, alias="API_LANGS")
    api_poll_interval_seconds: float = Field(2.0, alias="API_POLL_INTERVAL_SECONDS")
    api_max_poll_attempts: int = Field(60, alias="API_MAX_POLL_ATTEMPTS")
    api_http_timeout_seconds: float = Field(60.0, alias="API_HTTP_TIMEOUT_SECONDS")
    api_endpoint: str = Field("ocr", alias="API_ENDPOINT")
    gdoc_project_id: str | None = Field(None, alias="GDOC_PROJECT_ID")
    gdoc_location: str | None = Field(None, alias="GDOC_LOCATION")
    gdoc_processor_id: str | None = Field(None, alias="GDOC_PROCESSOR_ID")
    gdoc_extractor_processor_id: str | None = Field(
        None, alias="GDOC_EXTRACTOR_PROCESSOR_ID"
    )
    quality_min_score: float = Field(0.55, alias="QUALITY_MIN_SCORE")
    field_min_confidence: float = Field(0.75, alias="FIELD_MIN_CONFIDENCE")
    use_gdoc_ai_gate: bool = Field(False, alias="USE_GDOC_AI_GATE")

    @model_validator(mode="after")
    def _ensure_directories(self) -> "AppSettings":
        """Normalize directories and ensure the output directory exists."""
        self.images_dir = self.images_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self
