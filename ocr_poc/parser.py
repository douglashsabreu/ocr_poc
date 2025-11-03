from __future__ import annotations

from typing import List

from ocr_poc.models import OCRFinalResponse, OCRPage


class OCRContentFormatter:
    """Helpers to convert OCR responses into human-readable text."""

    def __init__(self, response: OCRFinalResponse) -> None:
        self._response = response

    def render_summary(self) -> str:
        status = self._response.status_label() or "unknown"
        success = self._response.success
        success_label = "success" if success else "pending" if success is None else "failed"
        parts = [
            f"Status: {status}",
            f"Success: {success_label}",
            f"Pages: {self._response.page_count or len(self._response.pages)}",
        ]
        if self._response.error:
            parts.append(f"Error: {self._response.error}")
        return "\n".join(parts)

    def render_pages(self) -> str:
        sections: List[str] = []
        for index, page in enumerate(self._response.pages, start=1):
            sections.append(format_page(page, index))
        return "\n\n".join(sections).strip()

    def render_full_text(self) -> str:
        summary = self.render_summary()
        pages = self.render_pages()
        if not pages:
            return summary
        return f"{summary}\n\n{pages}"


def format_page(page: OCRPage, index: int) -> str:
    header = f"# Página {page.page or index}"
    lines = page.deduplicated_plain_lines()
    if not lines:
        return header
    formatted_lines = "\n".join(f"- {line}" for line in lines)
    return f"{header}\n{formatted_lines}"

