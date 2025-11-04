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
    """Normalise provider-specific payloads to a common structure.
    
    Args:
        mode: OCR provider mode identifier (gdocai or datalab_api).
        payload: Raw provider-specific response payload.
        quality: Optional quality metrics to inject (used for datalab_api).
        
    Returns:
        Normalized dictionary with keys: lines, full_text, quality, raw_payload.
        
    Raises:
        ValueError: If the mode is not supported.
    """
    if mode == "gdocai":
        return _normalise_gdocai(payload)
    if mode == "datalab_api":
        return _normalise_datalab(payload, quality)
    raise ValueError(f"Modo de OCR não suportado para normalização: {mode}")


def _normalise_gdocai(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Google Document AI response to common structure.
    
    Args:
        payload: Google Document AI response with lines and quality data.
        
    Returns:
        Normalized dictionary with standardized line format.
    """
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
    """Normalize Datalab API response to common structure.
    
    Args:
        result: Datalab API result object with parsed pages.
        quality: Optional quality metrics from external quality gate.
        
    Returns:
        Normalized dictionary with lines, text, and quality data.
    """
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
    """Extract lines and full text from Datalab API result pages.
    
    Args:
        result: Datalab API result with parsed page data.
        
    Returns:
        Tuple of (list of line dictionaries, concatenated full text).
    """
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
    """Ensure line dictionary has all standard fields with proper types.
    
    Args:
        line: Raw line dictionary from provider.
        
    Returns:
        Line dictionary with standardized fields.
    """
    return {
        "text": line.get("text", ""),
        "confidence": float(line.get("confidence")) if line.get("confidence") is not None else None,
        "bbox": line.get("bbox"),
        "page": line.get("page"),
    }


def _resolve_bbox(line: Any) -> List[float] | None:
    """Resolve bounding box from line bbox or polygon coordinates.
    
    Args:
        line: Line object with optional bbox or polygon attributes.
        
    Returns:
        Bounding box as [x_min, y_min, x_max, y_max] or None.
    """
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
    """Safely iterate over polygon points.
    
    Args:
        polygon: Collection of coordinate points.
        
    Yields:
        Individual point coordinates as iterables.
    """
    for point in polygon or []:
        if isinstance(point, Iterable):
            yield point
