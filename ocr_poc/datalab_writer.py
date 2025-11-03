from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict

from ocr_poc.datalab_client import DatalabApiResult
from ocr_poc.parser import OCRContentFormatter
from ocr_poc.report import build_delivery_report
from ocr_poc.validation import validate_delivery


class DatalabApiResultWriter:
    """Persist OCR responses from the Datalab API."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(self, source_path: Path, result: DatalabApiResult) -> Dict[str, object]:
        target_dir = self._output_dir / source_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / f"{source_path.stem}_ocr.json"
        text_path = target_dir / f"{source_path.stem}_ocr.txt"
        validation_path = target_dir / f"{source_path.stem}_validation.json"
        report_path = target_dir / f"{source_path.stem}_validation.pdf"

        json_path.write_text(json.dumps(result.raw, indent=2), encoding="utf-8")
        text_content = self._format_text(result)
        text_path.write_text(text_content, encoding="utf-8")

        validation = validate_delivery(result.parsed)
        reference_id = str(uuid.uuid4())
        validation = validation.model_copy(update={"reference_id": reference_id})
        validation_path.write_text(
            json.dumps(validation.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        build_delivery_report(report_path, source_path, result, validation)

        return {
            "json": json_path,
            "text": text_path,
            "validation": validation_path,
            "report": report_path,
            "validation_data": validation,
        }

    def _format_text(self, result: DatalabApiResult) -> str:
        formatter = OCRContentFormatter(result.parsed)
        header = [
            f"Request ID: {result.request_id}",
            formatter.render_summary(),
        ]
        pages = formatter.render_pages()
        if pages:
            header.append("")
            header.append(pages)
        return "\n".join(part for part in header if part)
