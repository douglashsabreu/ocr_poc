"""Main entry point for OCR proof of concept pipeline execution."""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ocr_poc.config import AppSettings
from ocr_poc.document_pipeline import DocumentPipeline
from ocr_poc.document_writer import DocumentResultWriter
from ocr_poc.image_repository import ImageRepository


def configure_logging() -> None:
    """Configure logging settings for the application.
    
    Sets up basic logging configuration with INFO level and structured format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for OCR pipeline execution.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing mode, 
            gate settings, and directory paths.
    """
    parser = argparse.ArgumentParser(description="Executa pipelines de OCR da POC.")
    parser.add_argument(
        "--mode",
        choices=["datalab_api", "gdocai", "chandra", "openai_api"],
        help="Override do modo configurado via PIPELINE_MODE",
    )
    parser.add_argument(
        "--use-gate",
        action=argparse.BooleanOptionalAction,
        help="Força o uso (ou não) do quality gate do Google Document AI",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Diretório com as imagens a processar (override IMAGES_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Diretório destino para os artefatos (override OUTPUT_DIR)",
    )
    return parser.parse_args()


def run_chandra_pipeline(settings: AppSettings, repository: ImageRepository) -> None:
    """Execute OCR pipeline using Chandra OCR client.
    
    Args:
        settings: Application configuration settings.
        repository: Image repository containing files to process.
    """
    from ocr_poc.image_loader import ImageLoader
    from ocr_poc.ocr_client import ChandraOCRClient
    from ocr_poc.pipeline import OCRPipeline
    from ocr_poc.result_writer import ResultWriter

    loader = ImageLoader()
    client = ChandraOCRClient(settings)
    writer = ResultWriter(settings.output_dir, save_images=settings.include_images)

    pipeline = OCRPipeline(
        repository,
        loader,
        client,
        writer,
        logger=logging.getLogger("ocr_poc.chandra"),
    )
    pipeline.run()


def run_openai_pipeline(settings: AppSettings, repository: ImageRepository) -> None:
    """Execute OCR pipeline using OpenAI multimodal API.
    
    Args:
        settings: Application configuration settings.
        repository: Image repository containing files to process.
    """
    from ocr_poc.api_pipeline import DatalabApiPipeline
    from ocr_poc.datalab_writer import DatalabApiResultWriter
    from ocr_poc.openai_client import OpenAIOCRClient

    api_client = OpenAIOCRClient(settings)
    api_writer = DatalabApiResultWriter(settings.output_dir)
    pipeline = DatalabApiPipeline(
        repository,
        api_client,
        api_writer,
        provider_name="API OpenAI",
        logger=logging.getLogger("ocr_poc.openai"),
    )
    pipeline.run()


def run_document_mode_pipeline(
    settings: AppSettings, repository: ImageRepository
) -> None:
    """Execute document processing pipeline with validation and quality gating.
    
    Processes documents through OCR providers (Datalab or Google Document AI),
    applies quality gates, performs field extraction and validation, and 
    generates structured output artifacts with detailed logging.
    
    Args:
        settings: Application configuration settings.
        repository: Image repository containing files to process.
    """
    pipeline = DocumentPipeline(
        settings,
        repository,
        logger=logging.getLogger("ocr_poc.document"),
    )
    writer = DocumentResultWriter(
        settings.output_dir,
        field_min_confidence=settings.field_min_confidence,
        quality_min_score=settings.quality_min_score,
    )
    try:
        for outcome in pipeline.run():
            if outcome is None:
                continue
            saved = writer.write(outcome)
            validation = saved.get("validation_data")
            decision = validation.decision if validation else "-"
            decision_score = validation.decision_score if validation else -1.0
            quality_min = (
                validation.quality.get("score_min") if validation else None
            )
            quality_avg = (
                validation.quality.get("score_avg") if validation else None
            )
            logging.getLogger("ocr_poc.document").info(
                "run_summary file=%s mode=%s decision=%s decision_score=%.2f quality_min=%s quality_avg=%s latencies=%s",
                outcome.source_path.name,
                outcome.mode,
                decision,
                decision_score,
                quality_min,
                quality_avg,
                outcome.latencies,
            )
            if validation and validation.issues:
                logging.getLogger("ocr_poc.document").warning(
                    "decision_issues file=%s issues=%s",
                    outcome.source_path.name,
                    "; ".join(validation.issues),
                )
    finally:
        pipeline.close()


def main() -> None:
    """Main application entry point.
    
    Loads configuration, parses arguments, initializes the appropriate pipeline
    based on the selected mode, and executes OCR processing.
    """
    args = parse_args()
    load_dotenv()
    overrides = {}
    if args.mode:
        overrides["PIPELINE_MODE"] = args.mode
    if args.use_gate is not None:
        overrides["USE_GDOC_AI_GATE"] = args.use_gate
    if args.images_dir:
        overrides["IMAGES_DIR"] = str(args.images_dir)
    if args.output_dir:
        overrides["OUTPUT_DIR"] = str(args.output_dir)

    settings = AppSettings(**overrides)

    configure_logging()
    logger = logging.getLogger("ocr_poc")
    logger.info(
        "Inicializando pipeline de OCR (%s). images_dir=%s output_dir=%s gate=%s thresholds={quality: %.2f, field: %.2f}",
        settings.pipeline_mode,
        settings.images_dir,
        settings.output_dir,
        settings.use_gdoc_ai_gate,
        settings.quality_min_score,
        settings.field_min_confidence,
    )

    repository = ImageRepository(settings.images_dir)

    if settings.pipeline_mode == "chandra":
        run_chandra_pipeline(settings, repository)
    elif settings.pipeline_mode in {"datalab_api", "gdocai"}:
        run_document_mode_pipeline(settings, repository)
    elif settings.pipeline_mode == "openai_api":
        run_openai_pipeline(settings, repository)
    else:
        logger.error("Modo de pipeline desconhecido: %s", settings.pipeline_mode)


if __name__ == "__main__":
    main()
