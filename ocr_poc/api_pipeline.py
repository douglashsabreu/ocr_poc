from __future__ import annotations

import logging

from ocr_poc.datalab_client import DatalabApiClient
from ocr_poc.datalab_writer import DatalabApiResultWriter
from ocr_poc.image_repository import ImageRepository


class DatalabApiPipeline:
    """Pipeline that sends files to the Datalab OCR API."""

    def __init__(
        self,
        repository: ImageRepository,
        client: DatalabApiClient,
        writer: DatalabApiResultWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._client = client
        self._writer = writer
        self._logger = logger or logging.getLogger(__name__)

    def run(self) -> None:
        files = self._repository.list_files()
        if not files:
            self._logger.warning("Nenhum arquivo suportado encontrado para OCR.")
            return

        for path in files:
            self._logger.info("Processando arquivo via API: %s", path.name)
            try:
                result = self._client.process_file(path)
                saved_paths = self._writer.write(path, result)
                self._logger.info(
                    "OCR concluído para %s. Resultado JSON em %s",
                    path.name,
                    saved_paths["json"],
                )
            except Exception:  # noqa: BLE001
                self._logger.exception(
                    "Falha ao processar o arquivo %s utilizando a API da Datalab.",
                    path,
                )

