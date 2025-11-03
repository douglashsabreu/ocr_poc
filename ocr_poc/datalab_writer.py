from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ocr_poc.datalab_client import DatalabApiResult


class DatalabApiResultWriter:
    """Persist OCR responses from the Datalab API."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(self, source_path: Path, result: DatalabApiResult) -> Dict[str, Path]:
        target_dir = self._output_dir / source_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        json_path = target_dir / f"{source_path.stem}_ocr.json"
        text_path = target_dir / f"{source_path.stem}_ocr.txt"

        json_path.write_text(json.dumps(result.raw, indent=2), encoding="utf-8")
        text_content = self._format_text(result)
        text_path.write_text(text_content, encoding="utf-8")

        return {"json": json_path, "text": text_path}

    def _format_text(self, result: DatalabApiResult) -> str:
        if not result.text_per_page:
            return ""

        pages = []
        for index, page_text in enumerate(result.text_per_page, start=1):
            header = f"# Página {index}"
            pages.append(f"{header}\n{page_text}".strip())

        return "\n\n".join(pages)

