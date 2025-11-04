"""Persists results produced by the unified document pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ocr_poc.document_pipeline import PipelineOutcome
from ocr_poc.extraction.fields import extract_fields
from ocr_poc.report import build_validation_report
from ocr_poc.validation.engine import ValidationOutcome, run_validation


class DocumentResultWriter:
    """Serialises the MVP artefacts for downstream consumption."""

    def __init__(
        self,
        output_dir: Path,
        *,
        field_min_confidence: float,
        quality_min_score: float,
    ) -> None:
        """Initialize the document result writer.
        
        Args:
            output_dir: Directory where output artifacts will be saved.
            field_min_confidence: Minimum confidence threshold for extracted fields.
            quality_min_score: Minimum quality score threshold for documents.
        """
        self._output_dir = output_dir
        self._field_min_confidence = field_min_confidence
        self._quality_min_score = quality_min_score

    def write(self, outcome: PipelineOutcome) -> Dict[str, object]:
        """Write pipeline outcome to structured output files.
        
        Creates JSON, text, validation JSON, PDF report, and optional raw payload
        files in a dedicated subdirectory for each processed document.
        
        Args:
            outcome: Complete pipeline processing result including OCR data,
                quality metrics, and artifacts.
                
        Returns:
            Dictionary mapping artifact names to their file paths, including
            validation data object.
        """
        target_dir = self._output_dir / outcome.source_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        normalized = outcome.normalized
        lines = normalized.get("lines") or []
        full_text = normalized.get("full_text") or ""
        quality = normalized.get("quality") or outcome.quality_gate

        extracted_fields = extract_fields(lines, full_text)
        validation = run_validation(
            extracted_fields,
            quality,
            field_min_confidence=self._field_min_confidence,
            quality_min_score=self._quality_min_score,
            engine_used=outcome.engine_used,
            engine_chain=outcome.engine_chain,
        )

        ocr_path = target_dir / f"{outcome.source_path.stem}_ocr.json"
        text_path = target_dir / f"{outcome.source_path.stem}_ocr.txt"
        validation_path = target_dir / f"{outcome.source_path.stem}_validation.json"
        report_path = target_dir / f"{outcome.source_path.stem}_validation.pdf"
        raw_path = None

        self._write_json(
            ocr_path,
            {
                "mode": outcome.mode,
                "engine_used": outcome.engine_used,
                "engine_chain": outcome.engine_chain,
                "latencies": outcome.latencies,
                "quality": validation.quality,
                "fields": {
                    name: field.as_dict() for name, field in extracted_fields.items()
                },
                "full_text": full_text,
                "lines": lines,
                "raw_payload": normalized.get("raw_payload"),
                "artifacts": outcome.artifacts,
            },
        )
        self._write_text(
            text_path,
            outcome,
            validation,
            full_text,
        )
        self._write_json(validation_path, validation.to_dict())
        build_validation_report(report_path, outcome.source_path, outcome, validation)

        raw_payload = normalized.get("raw_payload")
        if raw_payload:
            suffix = "gdocai_raw" if "gdoc" in outcome.engine_used else "raw"
            raw_path = target_dir / f"{outcome.source_path.stem}_{suffix}.json"
            self._write_json(raw_path, raw_payload)

        return {
            "json": ocr_path,
            "text": text_path,
            "validation": validation_path,
            "report": report_path,
            "raw": raw_path,
            "validation_data": validation,
        }

    def _write_json(self, path: Path, payload: object) -> None:
        """Write payload as formatted JSON file.
        
        Args:
            path: Destination file path.
            payload: Object to serialize to JSON.
        """
        path.write_text(
            json.dumps(self._to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_text(
        self,
        path: Path,
        outcome: PipelineOutcome,
        validation: ValidationOutcome,
        full_text: str,
    ) -> None:
        """Write human-readable summary of pipeline results.
        
        Args:
            path: Destination file path.
            outcome: Pipeline processing result.
            validation: Validation outcome with decision and extracted fields.
            full_text: Complete OCR text extracted from document.
        """
        sections = [
            f"Arquivo: {outcome.source_path.name}",
            f"Modo selecionado: {outcome.mode}",
            f"Engine final: {outcome.engine_used}",
            f"Cadeia de engines: {', '.join(outcome.engine_chain) or '-'}",
            "",
            "== Qualidade ==",
            f"score_min: {validation.quality.get('score_min')}",
            f"score_avg: {validation.quality.get('score_avg')}",
            f"Limiar aplicado: {self._quality_min_score}",
            f"Sugestões: {', '.join(validation.quality.get('hints', [])) or '-'}",
            "",
            "== Decisão ==",
            f"Estado: {validation.decision}",
            f"Decision score: {validation.decision_score:.4f}",
            f"Pendências: {', '.join(validation.issues) if validation.issues else 'Nenhuma'}",
            "",
            "== Campos extraídos ==",
        ]

        for name, field in validation.fields.items():
            sections.append(
                f"- {name}: valor={field.value!r}, confiança={field.confidence}, página={field.page}, bbox={field.bbox}"
            )

        sections.append("")
        sections.append("== Texto OCR ==")
        sections.append(full_text or "(sem conteúdo extraído)")

        path.write_text("\n".join(sections), encoding="utf-8")

    def _to_jsonable(self, payload):
        """Recursively convert payload to JSON-serializable types.
        
        Args:
            payload: Object to convert.
            
        Returns:
            JSON-serializable representation of the payload.
        """
        if isinstance(payload, (str, int, float, bool)) or payload is None:
            return payload
        if isinstance(payload, list):
            return [self._to_jsonable(item) for item in payload]
        if isinstance(payload, dict):
            return {
                str(key): self._to_jsonable(value) for key, value in payload.items()
            }
        if isinstance(payload, Path):
            return str(payload)
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        return str(payload)
