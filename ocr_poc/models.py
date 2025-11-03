from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OCRCharacter(BaseModel):
    """Represents a single character detected by the OCR API."""

    model_config = ConfigDict(extra="ignore")

    text: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    bbox_valid: Optional[bool] = None


class OCRTextLine(BaseModel):
    """Represents a textual line within a page."""

    model_config = ConfigDict(extra="ignore")

    text: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    chars: List[OCRCharacter] = Field(default_factory=list)

    def as_plain_text(self) -> str:
        """Return the line text stripped and safe for display."""
        return (self.text or "").strip()


class OCRPage(BaseModel):
    """Represents a page in the OCR response."""

    model_config = ConfigDict(extra="ignore")

    page: Optional[int] = None
    text_lines: List[OCRTextLine] = Field(default_factory=list)
    lines: List[OCRTextLine] = Field(default_factory=list)
    image_bbox: Optional[List[float]] = None
    page_box: Optional[List[float]] = None

    def iter_lines(self) -> List[OCRTextLine]:
        """Prefer `text_lines` but fall back to `lines` when necessary."""
        return self.text_lines or self.lines or []

    def deduplicated_plain_lines(self) -> List[str]:
        """Return cleaned, deduplicated line strings for this page."""
        plain_lines: List[str] = []
        previous: Optional[str] = None
        for line in self.iter_lines():
            text = line.as_plain_text()
            if not text:
                continue
            if text == previous:
                continue
            plain_lines.append(text)
            previous = text
        return plain_lines

    def as_single_block(self) -> str:
        """Join the deduplicated lines into a single block of text."""
        return "\n".join(self.deduplicated_plain_lines())


class OCRFinalResponse(BaseModel):
    """Top-level model for the OCR final response payload."""

    model_config = ConfigDict(extra="ignore")

    status: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    page_count: Optional[int] = None
    total_cost: Optional[int] = None
    versions: Optional[dict] = None
    pages: List[OCRPage] = Field(default_factory=list)

    def status_label(self) -> str:
        return (self.status or "").lower()

