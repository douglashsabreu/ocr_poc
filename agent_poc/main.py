"""Command line interface for image transcription agent."""

import argparse
import imghdr
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from .azure_blob_saver import upload_bytes_get_address
from .image_transcription import transcribe_uploaded_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_IMAGE_TYPES = {"jpeg", "png", "gif", "webp"}


@dataclass(frozen=True)
class EnvironmentConfig:
    """Holds environment configuration required for transcription."""

    storage_account_name: str
    storage_account_key: str
    container_name: str
    openai_api_key: str


def _build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description="Transcribe images using OpenAI Vision."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to an image file or a directory containing image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transcriptions"),
        help="Directory where transcription outputs will be stored.",
    )
    parser.add_argument(
        "--ttl-minutes",
        type=int,
        default=60,
        help="Validity window in minutes for the temporary blob.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the CLI execution.",
    )
    return parser


def _configure_logging(level: str) -> None:
    """Configure logging for the CLI."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _load_environment() -> EnvironmentConfig:
    """Load required environment variables."""

    storage_account_name = _require_env("AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key = _require_env("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = _require_env("AZURE_STORAGE_CONTAINER_NAME")
    openai_api_key = _require_env("OPENAI_API_KEY")
    return EnvironmentConfig(
        storage_account_name=storage_account_name,
        storage_account_key=storage_account_key,
        container_name=container_name,
        openai_api_key=openai_api_key,
    )


def _require_env(name: str) -> str:
    """Return environment variable value or raise an error."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Environment variable {name} is required")
    return value


def _collect_targets(target: Path) -> List[Path]:
    """Collect supported image files from a file or directory target."""

    resolved = target.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Target path not found: {resolved}")
    if resolved.is_file():
        _validate_image_file(resolved)
        return [resolved]
    if resolved.is_dir():
        return _collect_from_directory(resolved)
    raise ValueError(f"Unsupported target path: {resolved}")


def _collect_from_directory(directory: Path) -> List[Path]:
    """Collect supported image files from a directory."""

    logger = logging.getLogger("agent_poc.transcription")
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            _validate_image_file(path)
        except ValueError as error:
            logger.warning("Skipping %s: %s", path.name, error)
            continue
        files.append(path)
    if not files:
        raise ValueError(
            "No supported image files were found in the provided directory"
        )
    return files


def _validate_image_file(path: Path) -> None:
    """Validate path is a supported image file."""

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported file extension")
    detected_type = imghdr.what(path)
    if detected_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("file content is not a supported image format")


def _ensure_output_dir(output_dir: Path) -> Path:
    """Ensure the output directory exists and return its absolute path."""

    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _process_images(
    image_paths: Iterable[Path],
    config: EnvironmentConfig,
    output_dir: Path,
    ttl_minutes: int,
) -> None:
    """Process each image and persist transcription results."""

    logger = logging.getLogger("agent_poc.transcription")
    for image_path in image_paths:
        logger.info("Processing %s", image_path.name)
        output_path = _transcribe_path(image_path, config, output_dir, ttl_minutes)
        logger.info("Saved transcription to %s", output_path)


def _transcribe_path(
    image_path: Path,
    config: EnvironmentConfig,
    output_dir: Path,
    ttl_minutes: int,
) -> Path:
    """Transcribe a single image and return the output path."""

    blob_url = _upload_image_to_blob(image_path, config, ttl_minutes)
    transcription = transcribe_uploaded_image(blob_url, config.openai_api_key)
    output_path = output_dir / f"{image_path.stem}.txt"
    output_path.write_text(transcription, encoding="utf-8")
    return output_path


def _upload_image_to_blob(
    image_path: Path,
    config: EnvironmentConfig,
    ttl_minutes: int,
) -> str:
    """Upload image bytes to Azure Blob Storage and return access URL."""

    blob_name = _generate_blob_name(image_path)
    ttl_days = _calculate_ttl_days(ttl_minutes)
    content_type = _guess_content_type(image_path)
    data_bytes = image_path.read_bytes()
    return upload_bytes_get_address(
        data_bytes=data_bytes,
        blob_name=blob_name,
        container_name=config.container_name,
        storage_account_name=config.storage_account_name,
        storage_account_key=config.storage_account_key,
        ttl_days=ttl_days,
        content_type=content_type,
    )


def _generate_blob_name(image_path: Path) -> str:
    """Generate a unique blob name for the uploaded image."""

    timestamp = int(time.time() * 1000)
    return f"agent_{image_path.stem}_{timestamp}{image_path.suffix.lower()}"


def _calculate_ttl_days(ttl_minutes: int) -> float:
    """Convert minutes to fractional days for SAS token validity."""

    minutes = max(ttl_minutes, 1)
    return minutes / (24 * 60)


def _guess_content_type(image_path: Path) -> str | None:
    """Guess MIME type for the provided image path."""

    mime_type, _ = mimetypes.guess_type(image_path.name)
    return mime_type


def main() -> None:
    """Entry point for the agent transcription CLI."""

    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv()
    _configure_logging(args.log_level)
    try:
        config = _load_environment()
    except ValueError as error:
        parser.error(str(error))
        return
    try:
        images = _collect_targets(args.target)
    except (FileNotFoundError, ValueError) as error:
        logging.getLogger("agent_poc.transcription").error(str(error))
        raise SystemExit(1) from error
    output_dir = _ensure_output_dir(args.output_dir)
    try:
        _process_images(images, config, output_dir, args.ttl_minutes)
    except Exception as error:
        logging.getLogger("agent_poc.transcription").exception(
            "Processing failed: %s", error
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
