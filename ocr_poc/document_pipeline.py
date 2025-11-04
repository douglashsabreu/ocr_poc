"""Unified pipeline orchestrator for Document AI and Datalab providers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ocr_poc.config import AppSettings
from ocr_poc.datalab_client import DatalabApiClient
from ocr_poc.image_repository import ImageRepository
from ocr_poc.normalization import normalize_to_lines_and_meta
from ocr_poc.providers import GoogleDocAiProvider, guess_mime_type
from ocr_poc.quality.gate import assess_quality


@dataclass
class PipelineOutcome:
    """Encapsulates the complete result of processing a single document.

    Attributes:
        source_path: Path to the original source file.
        mode: Processing mode used (gdocai, datalab_api, etc.).
        engine_used: Primary OCR engine that produced the final result.
        engine_chain: List of all engines invoked during processing.
        normalized: Normalized OCR data with lines, quality, and metadata.
        quality_gate: Quality assessment result with pass/fail and scores.
        artifacts: Additional processing artifacts (raw payloads, intermediate results).
        latencies: Processing time measurements for each stage.
        skipped_extraction: True if processing stopped early due to quality gate rejection.
    """

    source_path: Path
    mode: str
    engine_used: str
    engine_chain: List[str]
    normalized: Dict[str, Any]
    quality_gate: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    latencies: Dict[str, float] = field(default_factory=dict)
    skipped_extraction: bool = False


class DocumentPipeline:
    """Runs OCR processing according to the configured mode."""

    def __init__(
        self,
        settings: AppSettings,
        repository: ImageRepository,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the document processing pipeline.

        Args:
            settings: Application configuration containing API credentials and thresholds.
            repository: Image repository providing access to source files.
            logger: Optional logger instance for pipeline events.
        """
        self._settings = settings
        self._repository = repository
        self._logger = logger or logging.getLogger(__name__)
        self._datalab_client: Optional[DatalabApiClient] = None
        self._gdoc_provider: Optional[GoogleDocAiProvider] = None

        if settings.pipeline_mode in {"datalab_api", "openai_api"}:
            self._datalab_client = DatalabApiClient(settings)

        if (
            settings.gdoc_project_id
            and settings.gdoc_location
            and settings.gdoc_processor_id
        ):
            self._gdoc_provider = GoogleDocAiProvider(
                settings.gdoc_project_id,
                settings.gdoc_location,
                settings.gdoc_processor_id,
                settings.gdoc_extractor_processor_id,
            )

    def close(self) -> None:
        """Release resources held by OCR clients and providers."""
        if self._datalab_client:
            self._datalab_client.close()

    def run(self) -> Iterator[PipelineOutcome]:
        """Execute the pipeline over all files in the repository.

        Yields:
            PipelineOutcome: Processing result for each file, including OCR data,
                quality assessment, artifacts, and latency measurements.
        """
        files = self._repository.list_files()
        if not files:
            self._logger.warning(
                "Nenhum arquivo suportado encontrado para processamento."
            )
            return

        for path in files:
            self._logger.info(
                "Processando arquivo %s utilizando modo %s.",
                path.name,
                self._settings.pipeline_mode,
            )
            start_total = time.perf_counter()
            try:
                outcome = self._process_file(path)
                outcome.latencies["total"] = time.perf_counter() - start_total
                yield outcome
            except Exception:  # noqa: BLE001
                self._logger.exception("Falha ao processar o arquivo %s", path)

    def _process_file(self, path: Path) -> PipelineOutcome:
        """Route file processing to the appropriate provider based on configured mode.

        Args:
            path: Path to the file to process.

        Returns:
            PipelineOutcome containing normalized OCR data and quality metrics.

        Raises:
            RuntimeError: If the pipeline mode is not supported.
        """
        mode = self._settings.pipeline_mode
        if mode == "gdocai":
            return self._process_with_gdoc(path)
        if mode == "datalab_api":
            return self._process_with_datalab(path)
        raise RuntimeError(f"Modo de pipeline não suportado para o novo fluxo: {mode}")

    def _process_with_gdoc(self, path: Path) -> PipelineOutcome:
        """Process file using Google Document AI OCR processor.

        Args:
            path: Path to the file to process.

        Returns:
            PipelineOutcome with Google Document AI results and quality assessment.

        Raises:
            RuntimeError: If Google Document AI is not properly configured.
        """
        if not self._gdoc_provider:
            raise RuntimeError(
                "Google Document AI não configurado. Defina GDOC_PROJECT_ID, GDOC_LOCATION e GDOC_PROCESSOR_ID."
            )

        image_bytes = path.read_bytes()
        mime_type = guess_mime_type(path.name)

        start = time.perf_counter()
        result = self._gdoc_provider.process_bytes(image_bytes, mime_type)
        latency_gdoc = time.perf_counter() - start

        normalized = normalize_to_lines_and_meta("gdocai", result)
        quality = self._run_quality_gate(
            normalized.get("quality"), self._settings.quality_min_score
        )
        normalized["quality"] = quality

        artifacts = {
            "gdocai_raw": result.get("raw_payload"),
        }
        if self._settings.gdoc_extractor_processor_id:
            try:
                extractor_payload = self._gdoc_provider.try_wb_extractor(
                    image_bytes, mime_type
                )
                if extractor_payload:
                    artifacts["gdocai_extractor"] = extractor_payload
            except Exception:  # noqa: BLE001
                self._logger.warning(
                    "Falha ao acionar extractor do Workbench para %s.",
                    path.name,
                    exc_info=True,
                )
        artifacts["gdocai_quality"] = quality

        if not quality.get("pass", True):
            self._logger.warning(
                "Documento %s reprovado no quality gate (score_min=%s, threshold=%.2f).",
                path.name,
                quality.get("score_min"),
                self._settings.quality_min_score,
            )

        outcome = PipelineOutcome(
            source_path=path,
            mode="gdocai",
            engine_used="gdocai",
            engine_chain=["gdocai"],
            normalized=normalized,
            quality_gate=quality,
            artifacts=artifacts,
        )
        outcome.latencies["gdocai"] = latency_gdoc
        return outcome

    def _process_with_datalab(self, path: Path) -> PipelineOutcome:
        """Process file using Datalab API with optional Google quality gate pre-check.

        When quality gate is enabled, performs fast quality assessment via Google
        Document AI before calling Datalab. If quality fails threshold, returns
        early with quality rejection details. Otherwise proceeds with Datalab OCR.

        Args:
            path: Path to the file to process.

        Returns:
            PipelineOutcome with Datalab results, optional gate metrics, and quality assessment.

        Raises:
            RuntimeError: If Datalab client is not initialized.
        """
        if not self._datalab_client:
            raise RuntimeError("Cliente da API Datalab não inicializado.")

        mime_type = guess_mime_type(path.name)
        image_bytes: bytes | None = None
        quality_result: Dict[str, Any] | None = None
        artifacts: Dict[str, Any] = {}
        engine_chain: List[str] = []
        gate_latency: float | None = None

        if self._settings.use_gdoc_ai_gate:
            if not self._gdoc_provider:
                self._logger.warning(
                    "USE_GDOC_AI_GATE habilitado, porém credenciais do Document AI não estão completas."
                )
            else:
                start_gate = time.perf_counter()
                if image_bytes is None:
                    image_bytes = path.read_bytes()
                docai_result = self._gdoc_provider.process_bytes(image_bytes, mime_type)
                latency_gate = time.perf_counter() - start_gate

                normalized_gate = normalize_to_lines_and_meta("gdocai", docai_result)
                quality_result = self._run_quality_gate(
                    normalized_gate.get("quality"),
                    self._settings.quality_min_score,
                )
                artifacts["gdocai_raw"] = docai_result.get("raw_payload")
                artifacts["gdocai_gate_lines"] = normalized_gate.get("lines")
                engine_chain.append("gdocai_gate")
                if quality_result and not quality_result.get("pass"):
                    normalized_gate["quality"] = quality_result
                    outcome = PipelineOutcome(
                        source_path=path,
                        mode="datalab_api",
                        engine_used="gdocai_gate",
                        engine_chain=engine_chain,
                        normalized=normalized_gate,
                        quality_gate=quality_result,
                        artifacts=artifacts,
                        skipped_extraction=True,
                    )
                    outcome.latencies["gdocai_gate"] = latency_gate
                    self._logger.warning(
                        "Documento %s bloqueado no quality gate (score_min=%s, threshold=%.2f).",
                        path.name,
                        quality_result.get("score_min"),
                        self._settings.quality_min_score,
                    )
                    return outcome
                if quality_result:
                    artifacts["gdocai_quality"] = quality_result
                gate_latency = latency_gate

        start_datalab = time.perf_counter()
        datalab_result = self._datalab_client.process_file(path)
        latency_datalab = time.perf_counter() - start_datalab

        normalized = normalize_to_lines_and_meta(
            "datalab_api",
            datalab_result,
            quality=quality_result,
        )
        quality_gate = self._run_quality_gate(
            normalized.get("quality"),
            self._settings.quality_min_score,
        )
        normalized["quality"] = quality_gate

        artifacts["datalab_raw"] = datalab_result.raw
        artifacts["datalab_parsed"] = datalab_result.parsed.model_dump(mode="json")
        engine_chain.append("datalab_api")

        outcome = PipelineOutcome(
            source_path=path,
            mode="datalab_api",
            engine_used="datalab_api",
            engine_chain=engine_chain,
            normalized=normalized,
            quality_gate=quality_gate,
            artifacts=artifacts,
        )
        outcome.latencies["datalab_api"] = latency_datalab
        if gate_latency is not None:
            outcome.latencies["gdocai_gate"] = gate_latency
        if not quality_gate.get("pass", True):
            self._logger.warning(
                "Documento %s reprovado no quality gate pós-Datalab (score_min=%s, threshold=%.2f).",
                path.name,
                quality_gate.get("score_min"),
                self._settings.quality_min_score,
            )
        return outcome

    def _run_quality_gate(
        self,
        quality_metrics: Dict[str, Any] | None,
        threshold: float,
    ) -> Dict[str, Any]:
        """Evaluate quality metrics against configured threshold.

        Args:
            quality_metrics: Quality scores and reasons from OCR provider.
            threshold: Minimum acceptable quality score.

        Returns:
            Quality gate result with pass/fail status, scores, reasons, and hints.
        """
        quality_metrics = quality_metrics or {}
        if quality_metrics.get("score_min") is None:
            return {
                "score_min": quality_metrics.get("score_min"),
                "score_avg": quality_metrics.get("score_avg"),
                "reasons": quality_metrics.get("reasons") or [],
                "pass": True,
                "hints": [],
                "threshold": threshold,
            }
        assessment = assess_quality(quality_metrics, threshold)
        assessment["threshold"] = threshold
        return assessment
