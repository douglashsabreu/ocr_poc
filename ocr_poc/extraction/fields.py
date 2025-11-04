"""Field extraction heuristics for MVP validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)
TRACKING_PATTERN = re.compile(r"\b([A-Z]{2}\d{9}[A-Z]{2})\b")
LONG_NUMBER_PATTERN = re.compile(r"\b\d{10,}\b")
SIGNATURE_TRACES = ("____", "----", "_____", "------", "_______")
NAME_SEPARATORS = re.compile(r"[:\-–—]\s*")
KEYWORDS_RECIPIENT = (
    "recebedor",
    "recebido",
    "responsavel",
    "responsável",
    "assinatura",
    "assinante",
)
SIGNATURE_KEYWORDS = ("assinatura", "signature")


@dataclass
class ExtractedField:
    name: str
    value: str | bool | None
    confidence: float | None
    bbox: List[float] | None = None
    page: int | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": _round_confidence(self.confidence),
            "bbox": self.bbox,
            "page": self.page,
        }


def extract_fields(
    lines: List[Dict[str, object]], full_text: str
) -> Dict[str, ExtractedField]:
    """Return the MVP set of extracted fields."""
    indexed_lines = [_normalise_line(line, idx) for idx, line in enumerate(lines)]
    field_map: Dict[str, ExtractedField] = {}

    field_map["date"] = _extract_date(indexed_lines)
    field_map["recipient_name"] = _extract_recipient(indexed_lines)
    field_map["signature_present"] = _extract_signature(indexed_lines)
    field_map["tracking_code"] = _extract_tracking(indexed_lines, full_text)

    return field_map


def _extract_date(lines: List[Dict[str, object]]) -> ExtractedField:
    best: Optional[ExtractedField] = None
    for line in lines:
        text = line["text"]
        for match in DATE_PATTERN.finditer(text):
            value = _normalize_date(match.group(1))
            candidate = ExtractedField(
                name="date",
                value=value,
                confidence=line["confidence"],
                bbox=line["bbox"],
                page=line["page"],
            )
            best = _choose_best(best, candidate)
    if best:
        return best
    return ExtractedField(name="date", value=None, confidence=0.0)


def _extract_recipient(lines: List[Dict[str, object]]) -> ExtractedField:
    best: Optional[ExtractedField] = None
    for line in lines:
        lowered = line["text"].lower()
        if not any(keyword in lowered for keyword in KEYWORDS_RECIPIENT):
            continue
        value = _clean_name(_split_after_separator(line["text"]))
        if not value:
            continue
        candidate = ExtractedField(
            name="recipient_name",
            value=value,
            confidence=line["confidence"],
            bbox=line["bbox"],
            page=line["page"],
        )
        best = _choose_best(best, candidate)
    if best:
        return best
    return ExtractedField(name="recipient_name", value=None, confidence=0.0)


def _extract_signature(lines: List[Dict[str, object]]) -> ExtractedField:
    for line in lines:
        lowered = line["text"].lower()
        if not any(keyword in lowered for keyword in SIGNATURE_KEYWORDS):
            continue
        trace_found = any(marker in line["text"] for marker in SIGNATURE_TRACES)
        confidence = line["confidence"]
        if trace_found:
            confidence = max(confidence, 0.9 if confidence is not None else 0.9)
        else:
            confidence = max(confidence or 0.0, 0.6)
        return ExtractedField(
            name="signature_present",
            value=trace_found,
            confidence=confidence,
            bbox=line["bbox"],
            page=line["page"],
        )
    return ExtractedField(
        name="signature_present",
        value=False,
        confidence=0.5,
    )


def _extract_tracking(
    lines: List[Dict[str, object]], full_text: str
) -> ExtractedField:
    best: Optional[ExtractedField] = None
    for line in lines:
        text = line["text"]
        match = TRACKING_PATTERN.search(text)
        if match:
            candidate = ExtractedField(
                name="tracking_code",
                value=match.group(1),
                confidence=line["confidence"],
                bbox=line["bbox"],
                page=line["page"],
            )
            best = _choose_best(best, candidate)
            continue
        long_candidate = LONG_NUMBER_PATTERN.search(text)
        if long_candidate:
            candidate = ExtractedField(
                name="tracking_code",
                value=long_candidate.group(0),
                confidence=(line["confidence"] or 0.6),
                bbox=line["bbox"],
                page=line["page"],
            )
            best = _choose_best(best, candidate)
    if best:
        return best

    # Fallback to full text search
    fallback = TRACKING_PATTERN.search(full_text) or LONG_NUMBER_PATTERN.search(full_text)
    if fallback:
        return ExtractedField(
            name="tracking_code",
            value=fallback.group(1) if fallback.lastindex else fallback.group(0),
            confidence=0.4,
        )
    return ExtractedField(name="tracking_code", value=None, confidence=0.0)


# ---------------------------------------------------------------------------#
# Helpers


def _normalise_line(line: Dict[str, object], index: int) -> Dict[str, object]:
    return {
        "index": index,
        "text": str(line.get("text", "")),
        "confidence": _round_confidence(line.get("confidence")),
        "bbox": line.get("bbox"),
        "page": line.get("page"),
    }


def _round_confidence(value: object) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _choose_best(
    current: Optional[ExtractedField],
    candidate: ExtractedField,
) -> ExtractedField:
    if current is None:
        return candidate
    current_conf = current.confidence or 0.0
    new_conf = candidate.confidence or 0.0
    return candidate if new_conf >= current_conf else current


def _normalize_date(raw: str) -> str:
    tokens = re.split(r"[/-]", raw)
    if len(tokens) != 3:
        return raw

    if len(tokens[0]) == 4:
        year, month, day = tokens
    else:
        day, month, year = tokens
    if len(year) == 2:
        year = f"20{year}" if int(year) < 50 else f"19{year}"
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _split_after_separator(text: str) -> str:
    parts = NAME_SEPARATORS.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()


def _clean_name(name: str) -> str:
    cleaned = name.strip().strip(":.-–— ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.rstrip("0123456789")
    return cleaned.strip()
