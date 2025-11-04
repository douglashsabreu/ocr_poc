"""Validation engine for the MVP, independent from legacy delivery checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from ocr_poc.extraction.fields import ExtractedField


@dataclass
class ValidationOutcome:
    """Encapsulates the result of document validation logic.
    
    Attributes:
        decision: Final validation decision (OK, NEEDS_REVIEW, REPROVADO).
        decision_score: Minimum confidence across all evaluated fields.
        issues: List of validation issues found.
        fields: Extracted fields with metadata.
        quality: Quality assessment data.
        engine_used: Primary OCR engine identifier.
        engine_chain: Sequence of engines used in processing.
        thresholds: Applied threshold values.
    """
    decision: str
    decision_score: float
    issues: List[str] = field(default_factory=list)
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    quality: Dict[str, object] = field(default_factory=dict)
    engine_used: str = ""
    engine_chain: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert validation outcome to dictionary format.
        
        Returns:
            Dictionary representation with all validation data.
        """
        return {
            "decision": self.decision,
            "decision_score": round(self.decision_score, 4),
            "issues": self.issues,
            "fields": {name: field.as_dict() for name, field in self.fields.items()},
            "quality": self.quality,
            "engine_used": self.engine_used,
            "engine_chain": self.engine_chain,
            "thresholds": self.thresholds,
        }


def run_validation(
    fields: Dict[str, ExtractedField],
    quality: Dict[str, object],
    *,
    field_min_confidence: float,
    quality_min_score: float,
    engine_used: str,
    engine_chain: Iterable[str],
) -> ValidationOutcome:
    """Fuse extracted fields and quality scores into a high-level decision.
    
    Args:
        fields: Extracted document fields with confidence values.
        quality: Quality assessment metadata.
        field_min_confidence: Minimum acceptable confidence for fields.
        quality_min_score: Minimum acceptable quality score.
        engine_used: Primary engine identifier.
        engine_chain: Sequence of engines in processing chain.
        
    Returns:
        ValidationOutcome with decision, score, issues, and metadata.
    """
    issues: List[str] = []
    decision = "OK"
    scores: List[float] = []

    quality_passed, quality_score, quality_issues = _assess_quality_gate(
        quality, quality_min_score
    )
    scores.append(quality_score)
    issues.extend(quality_issues)
    if not quality_passed:
        decision = "REPROVADO"

    for name, field in fields.items():
        normalized_name = name.lower()
        confidence = field.confidence or 0.0
        value = field.value

        if normalized_name == "signature_present":
            if value is False:
                issues.append("Assinatura não detectada no comprovante.")
                if decision != "REPROVADO":
                    decision = "NEEDS_REVIEW"
            scores.append(confidence)
            continue

        if not value:
            issues.append(f"O campo obrigatório '{name}' não foi identificado.")
            decision = "REPROVADO"
            scores.append(confidence)
            continue

        scores.append(confidence)
        if confidence < 0.5:
            issues.append(
                f"O campo '{name}' apresentou baixa confiança ({confidence:.2f})."
            )
            decision = "REPROVADO"
        elif confidence < field_min_confidence and decision != "REPROVADO":
            issues.append(
                f"O campo '{name}' precisa de revisão (confiança {confidence:.2f})."
            )
            decision = "NEEDS_REVIEW"

    decision_score = min(scores) if scores else 0.0
    thresholds = {
        "field_min_confidence": field_min_confidence,
        "quality_min_score": quality_min_score,
    }
    return ValidationOutcome(
        decision=decision,
        decision_score=decision_score,
        issues=issues,
        fields=fields,
        quality=quality,
        engine_used=engine_used,
        engine_chain=list(engine_chain),
        thresholds=thresholds,
    )


def _assess_quality_gate(
    quality: Dict[str, object],
    quality_min_score: float,
) -> Tuple[bool, float, List[str]]:
    """Evaluate quality metrics against threshold.
    
    Args:
        quality: Quality metadata with scores and hints.
        quality_min_score: Minimum acceptable score threshold.
        
    Returns:
        Tuple of (passed, score, issues list).
    """
    score_min = _to_float(quality.get("score_min"))
    quality_passed = bool(quality.get("pass", True))

    issues: List[str] = []
    if score_min is not None:
        if score_min < quality_min_score:
            quality_passed = False
            issues.append(
                f"Qualidade abaixo do limiar ({score_min:.2f} < {quality_min_score:.2f})."
            )
            hints = quality.get("hints") or []
            issues.extend(hints)
    elif not quality_passed:
        issues.append("Qualidade do documento não atende ao limiar mínimo.")

    if score_min is None:
        baseline = quality_min_score if quality_passed else 0.0
        return quality_passed, baseline, issues
    return quality_passed, score_min, issues


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
