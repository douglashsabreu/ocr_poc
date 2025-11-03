import logging

from dotenv import load_dotenv

from ocr_poc.config import AppSettings
from ocr_poc.image_repository import ImageRepository


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_chandra_pipeline(settings: AppSettings, repository: ImageRepository) -> None:
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


def run_datalab_api_pipeline(
    settings: AppSettings, repository: ImageRepository
) -> None:
    from ocr_poc.api_pipeline import DatalabApiPipeline
    from ocr_poc.datalab_client import DatalabApiClient
    from ocr_poc.datalab_writer import DatalabApiResultWriter

    api_client = DatalabApiClient(settings)
    try:
        api_writer = DatalabApiResultWriter(settings.output_dir)
        pipeline = DatalabApiPipeline(
            repository,
            api_client,
            api_writer,
            provider_name="API Datalab",
            logger=logging.getLogger("ocr_poc.api"),
        )
        pipeline.run()
    finally:
        api_client.close()


def run_openai_pipeline(settings: AppSettings, repository: ImageRepository) -> None:
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


def main() -> None:
    load_dotenv()
    settings = AppSettings()

    configure_logging()
    logger = logging.getLogger("ocr_poc")
    logger.info("Inicializando pipeline de OCR (%s).", settings.pipeline_mode)

    repository = ImageRepository(settings.images_dir)

    if settings.pipeline_mode == "chandra":
        run_chandra_pipeline(settings, repository)
    elif settings.pipeline_mode == "openai_api":
        run_openai_pipeline(settings, repository)
    else:
        run_datalab_api_pipeline(settings, repository)


if __name__ == "__main__":
    main()
