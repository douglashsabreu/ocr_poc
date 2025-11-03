from __future__ import annotations

import logging
from typing import Iterable

from ocr_poc.image_loader import ImageLoader, ImagePayload
from ocr_poc.image_repository import ImageRepository
from ocr_poc.ocr_client import OCRClient
from ocr_poc.result_writer import ResultWriter


class OCRPipeline:
    """Co-ordinates loading, processing and persisting OCR results."""

    def __init__(
        self,
        repository: ImageRepository,
        loader: ImageLoader,
        client: OCRClient,
        writer: ResultWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._loader = loader
        self._client = client
        self._writer = writer
        self._logger = logger or logging.getLogger(__name__)

    def run(self) -> None:
        files = self._repository.list_files()
        if not files:
            self._logger.warning("Nenhum arquivo suportado encontrado para OCR.")
            return

        for path in files:
            self._logger.info("Processando arquivo: %s", path.name)
            try:
                for payload in self._loader.load(path):
                    self._process_payload(payload)
            except Exception:  # noqa: BLE001 - log full stack trace for visibility
                self._logger.exception("Falha ao processar o arquivo %s", path)

    def _process_payload(self, payload: ImagePayload) -> None:
        self._logger.debug(
            "Executando OCR para %s (página %s)",
            payload.source.name,
            payload.page_index + 1,
        )
        result = self._client.run(payload.image)
        if getattr(result, "error", False):
            self._logger.error(
                "OCR retornou erro para %s (página %s). Verifique as configurações da API.",
                payload.source.name,
                payload.page_index + 1,
            )
            return
        paths = self._writer.write(payload.source, payload.page_index, result)
        self._logger.info(
            "Página %s processada. Markdown salvo em %s",
            payload.page_index + 1,
            paths["markdown"],
        )
