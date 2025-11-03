from __future__ import annotations

import logging

from pathlib import Path
from typing import Protocol

from ocr_poc.datalab_writer import DatalabApiResultWriter
from ocr_poc.image_repository import ImageRepository
from ocr_poc.validation import DeliveryValidation


class OCRClientProtocol(Protocol):
    def process_file(self, path: Path):  # pragma: no cover - structural typing only
        ...


class DatalabApiPipeline:
    """Pipeline genérica para clientes OCR baseados em API."""

    def __init__(
        self,
        repository: ImageRepository,
        client: OCRClientProtocol,
        writer: DatalabApiResultWriter,
        provider_name: str = "API Datalab",
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._client = client
        self._writer = writer
        self._provider_name = provider_name
        self._logger = logger or logging.getLogger(__name__)

    def run(self) -> None:
        files = self._repository.list_files()
        if not files:
            self._logger.warning("Nenhum arquivo suportado encontrado para OCR.")
            return

        for path in files:
            self._logger.info("Processando arquivo via %s: %s", self._provider_name, path.name)
            try:
                result = self._client.process_file(path)
                saved = self._writer.write(path, result)
                validation = saved.get("validation_data")
                self._logger.info(
                    "OCR concluído para %s. Resultado JSON em %s",
                    path.name,
                    saved["json"],
                )
                if validation:
                    if isinstance(validation, DeliveryValidation) and validation.status == "ok":
                        self._logger.info(
                            "Validação concluída: mercadoria entregue a %s.",
                            validation.receiver or "(nome não identificado)",
                        )
                    else:
                        self._logger.warning(
                            "Validação incompleta (%s) para %s: %s",
                            validation.status,
                            path.name,
                            "; ".join(validation.issues) or "motivo não identificado",
                        )
            except Exception:  # noqa: BLE001
                self._logger.exception(
                    "Falha ao processar o arquivo %s utilizando %s.",
                    path,
                    self._provider_name,
                )
