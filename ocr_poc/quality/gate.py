"""Quality gate utilities shared across OCR providers."""

from __future__ import annotations

from typing import Iterable, Mapping


_SUGGESTIONS = {
    "motion_blur": "Evite movimentar o dispositivo durante a captura.",
    "defocus_blur": "Aproxime a câmera e refaça o foco antes de capturar.",
    "insufficient_lighting": "Aumente a iluminação ambiente ou evite ambientes escuros.",
    "low_brightness": "Aumente a iluminação ambiente ou evite ambientes escuros.",
    "over_exposure": "Reduza reflexos ou ajuste o ângulo para evitar áreas estouradas.",
    "under_exposure": "Aproxime a câmera ou utilize um ambiente mais iluminado.",
    "specular_glare": "Evite reflexos posicionando o documento em outro ângulo.",
    "camera_shake": "Segure o dispositivo com firmeza ou use apoio na captura.",
}


def assess_quality(
    quality_metrics: Mapping[str, object] | None, min_score: float
) -> dict:
    """Evaluate whether the document passes the minimum quality threshold.
    
    Args:
        quality_metrics: Quality scores and reasons from OCR provider.
        min_score: Minimum acceptable quality score threshold.
        
    Returns:
        Dictionary with quality assessment including pass/fail status,
        scores, normalized reasons, and actionable hints.
    """
    quality_metrics = quality_metrics or {}
    score_min = _to_float(quality_metrics.get("score_min"))
    score_avg = _to_float(quality_metrics.get("score_avg"))
    reasons_raw = quality_metrics.get("reasons") or []
    reasons = list(_normalise_reasons(reasons_raw))
    passed = score_min is not None and score_min >= min_score

    hints = [_map_reason_to_hint(reason) for reason in reasons]
    hints = [hint for hint in hints if hint]
    return {
        "score_min": score_min,
        "score_avg": score_avg,
        "reasons": reasons,
        "pass": passed,
        "hints": hints,
    }


def _map_reason_to_hint(reason: str) -> str | None:
    """Translate Document AI quality reasons into actionable hints.
    
    Args:
        reason: Quality issue reason code from Document AI.
        
    Returns:
        User-friendly suggestion for improving capture quality, or None.
    """
    key = reason.split(" ", 1)[0]
    return _SUGGESTIONS.get(key)


def _normalise_reasons(reasons: Iterable[object]) -> Iterable[str]:
    """Normalize quality reasons from various provider formats to strings.
    
    Args:
        reasons: Quality reasons in various formats (strings, dicts, objects).
        
    Yields:
        Normalized reason strings.
    """
    for reason in reasons:
        if isinstance(reason, str):
            yield reason
        elif isinstance(reason, Mapping):
            raw = reason.get("type") or reason.get("reason")
            if raw:
                yield str(raw)
        else:
            yield str(reason)


def _to_float(value: object) -> float | None:
    """Safely convert value to float.
    
    Args:
        value: Value to convert.
        
    Returns:
        Float value or None if conversion fails.
    """
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
