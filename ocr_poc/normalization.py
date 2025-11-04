"""Helpers to align OCR outputs across providers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

from ocr_poc.datalab_client import DatalabApiResult


def normalize_to_lines_and_meta(
    mode: str,
    payload: Any,
    *,
    quality: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalise provider-specific payloads to a common structure."""
    if mode == "gdocai":
        return _normalise_gdocai(payload)
    if mode == "datalab_api":
        return _normalise_datalab(payload, quality)
    raise ValueError(f"Modo de OCR não suportado para normalização: {mode}")


def _normalise_gdocai(payload: Dict[str, Any]) -> Dict[str, Any]:
    lines = payload.get("lines", [])
    text = "\n".join(line.get("text", "") for line in lines if line.get("text"))
    return {
        "lines": [_ensure_line_defaults(line) for line in lines],
        "full_text": text,
        "quality": payload.get("quality") or {},
        "raw_payload": payload.get("raw_payload"),
    }


def _normalise_datalab(
    result: DatalabApiResult,
    quality: Dict[str, Any] | None,
) -> Dict[str, Any]:
    lines, text = _extract_datalab_lines(result)
    return {
        "lines": lines,
        "full_text": text,
        "quality": quality or {"score_min": None, "score_avg": None, "reasons": []},
        "raw_payload": result.raw,
        "parsed": result.parsed.model_dump(mode="json"),
        "request_id": result.request_id,
    }


def _extract_datalab_lines(result: DatalabApiResult) -> Tuple[List[Dict[str, Any]], str]:
    aggregated_lines: List[Dict[str, Any]] = []
    text_blocks: List[str] = []

    for page_index, page in enumerate(result.parsed.pages or [], start=1):
        block_lines: List[str] = []
        for line in page.iter_lines():
            text = line.as_plain_text()
            if not text:
                continue
            aggregated_lines.append(
                {
                    "text": text,
                    "confidence": float(line.confidence) if line.confidence is not None else None,
                    "bbox": _resolve_bbox(line),
                    "page": page.page or page_index,
                }
            )
            block_lines.append(text)
        if block_lines:
            text_blocks.append("\n".join(block_lines))

    full_text = "\n\n".join(text_blocks)
    return aggregated_lines, full_text


def _ensure_line_defaults(line: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": line.get("text", ""),
        "confidence": float(line.get("confidence")) if line.get("confidence") is not None else None,
        "bbox": line.get("bbox"),
        "page": line.get("page"),
    }


def _resolve_bbox(line: Any) -> List[float] | None:
    if line.bbox:
        return list(line.bbox)
    if line.polygon:
        xs: List[float] = []
        ys: List[float] = []
        for point in _iterate_points(line.polygon):
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _iterate_points(polygon: Iterable[Iterable[float]]) -> Iterable[Iterable[float]]:
    for point in polygon or []:
        if isinstance(point, Iterable):
            yield point
