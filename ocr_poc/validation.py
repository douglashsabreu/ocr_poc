from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from ocr_poc.models import OCRFinalResponse, OCRPage


DATE_REGEX = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
TIME_REGEX = re.compile(r"(\d{1,2})(?:[:hH\.](\d{2}))(?:[:\.](\d{2}))?")

KEY_VALUE_REGEX = re.compile(
    r"^\s*(?P<key>[A-Za-zÀ-ÖØ-öø-ÿ0-9\s./º°-]{3,80}?)\s*[:\-–—\.]{1,3}\s*(?P<value>.+?)\s*$"
)

PENDING_KEY_HINTS = {
    "num nf",
    "num nf e",
    "numero nf",
    "numero nfe",
    "numero nota fiscal",
    "num nota fiscal",
    "num nf-e",
    "nro documento",
    "numero documento",
    "romaneio",
    "romaneio de carga",
}

RECEIVER_KEYWORDS = {
    "recebedor",
    "recebido",
    "recebemos",
    "recebi",
    "assinatura do recebedor",
    "assinatura recebedor",
    "assinatura recebedora",
    "assinatura do responsável",
}

DELIVERY_DATE_KEYWORDS = {
    "data recebimento",
    "data de recebimento",
    "data recebida",
    "hora recebimento",
    "hora recebida",
    "recebimento",
    "entrega",
    "data entrega",
    "termino da prestacao",
    "término da prestacao",
    "fim da prestacao",
    "entregue",
    "baixa",
}

SHIPMENT_DATE_KEYWORDS = {"data saida", "hora saida", "data saída", "hora saída"}

DOCUMENT_KEYWORDS = {
    "documento",
    "conhecimento",
    "cte",
    "romaneio",
    "romaneio de carga",
}

INVOICE_KEYWORDS = {
    "nf",
    "nfe",
    "nota fiscal",
    "nota-fiscal",
    "numero nf",
    "num nf",
    "num nf e",
    "número nf",
}

STOP_WORDS = {
    "data",
    "hora",
    "do",
    "da",
    "de",
    "del",
    "para",
    "no",
    "na",
}


@dataclass
class KeyValue:
    key: str
    value: str
    normalized_key: str
    index: int

    @property
    def base(self) -> str:
        tokens = [
            token
            for token in self.normalized_key.split()
            if token and token not in STOP_WORDS
        ]
        return " ".join(tokens).strip()


class DeliveryValidation(BaseModel):
    status: str
    issues: List[str] = Field(default_factory=list)
    receiver: Optional[str] = None
    received_at: Optional[datetime] = None
    shipment_at: Optional[datetime] = None
    invoice_numbers: List[str] = Field(default_factory=list)
    document_numbers: List[str] = Field(default_factory=list)
    raw_text_sample: List[str] = Field(default_factory=list)
    reference_id: Optional[str] = None


def validate_delivery(response: OCRFinalResponse) -> DeliveryValidation:
    lines = _aggregate_lines(response)
    total_chars = sum(len(line.strip()) for line in lines)

    if not lines or total_chars < 40:
        return DeliveryValidation(
            status="illegible",
            issues=["OCR output contained insufficient text to analyse."],
            raw_text_sample=lines[:10],
        )

    key_values = _extract_key_values(lines)
    receiver = _extract_receiver(lines, key_values)
    invoices = _extract_invoice_numbers(lines, key_values)
    documents = _extract_document_numbers(lines, key_values)
    received_at = _extract_datetime(key_values, lines, DELIVERY_DATE_KEYWORDS)
    shipment_at = _extract_datetime(key_values, lines, SHIPMENT_DATE_KEYWORDS)

    issues: List[str] = []
    if receiver is None:
        issues.append("Receiver name not identified.")
    if received_at is None:
        issues.append("Delivery timestamp not identified.")
    if not invoices:
        issues.append("Invoice number not detected.")

    status = "ok"
    if issues:
        status = "missing_data"

    return DeliveryValidation(
        status=status,
        issues=issues,
        receiver=receiver,
        received_at=received_at,
        shipment_at=shipment_at,
        invoice_numbers=invoices,
        document_numbers=documents,
        raw_text_sample=lines[:25],
    )


def _aggregate_lines(response: OCRFinalResponse) -> List[str]:
    aggregated: List[str] = []
    for page in response.pages:
        aggregated.extend(page.deduplicated_plain_lines())
    deduped: List[str] = []
    seen = set()
    for line in aggregated:
        normalized = line.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _extract_key_values(lines: List[str]) -> List[KeyValue]:
    results: List[KeyValue] = []
    pending_key: Optional[str] = None
    for idx, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue

        match = KEY_VALUE_REGEX.match(text)
        if match:
            key = match.group("key").strip()
            value = match.group("value").strip()
            if value and _contains_alpha(key):
                normalized_key = _canonical_key(key)
                results.append(KeyValue(key, value, normalized_key, idx))
            pending_key = None
            continue

        if pending_key and _looks_like_value(text):
            normalized_key = _canonical_key(pending_key)
            results.append(KeyValue(pending_key, text, normalized_key, idx))
            pending_key = None
            continue

        normalized_line = _canonical_key(text)
        if normalized_line in PENDING_KEY_HINTS and not _looks_like_value(text):
            pending_key = text
            continue

        pending_key = None

    return results


def _extract_receiver(lines: List[str], key_values: List[KeyValue]) -> Optional[str]:
    for kv in key_values:
        if any(keyword in kv.normalized_key for keyword in RECEIVER_KEYWORDS):
            cleaned = _clean_receiver_value(kv.value)
            if cleaned:
                return cleaned

    # fallback: free-form line with receiver info
    for line in lines:
        norm = _canonical_key(line)
        if any(keyword in norm for keyword in RECEIVER_KEYWORDS):
            cleaned = _clean_receiver_value(line)
            if cleaned:
                return cleaned
    return None


def _extract_invoice_numbers(lines: List[str], key_values: List[KeyValue]) -> List[str]:
    candidates: List[str] = []

    for kv in key_values:
        if any(keyword in kv.normalized_key for keyword in INVOICE_KEYWORDS):
            candidates.extend(_extract_numeric_tokens(kv.value))

    for idx, line in enumerate(lines):
        norm = _canonical_key(line)
        if any(keyword in norm for keyword in INVOICE_KEYWORDS):
            candidates.extend(_extract_numeric_tokens(line))
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if _looks_like_value(next_line):
                    candidates.extend(_extract_numeric_tokens(next_line))

    return _unique_preserving_order(filter(_valid_invoice, candidates))


def _extract_document_numbers(
    lines: List[str], key_values: List[KeyValue]
) -> List[str]:
    candidates: List[str] = []

    for kv in key_values:
        if any(keyword in kv.normalized_key for keyword in DOCUMENT_KEYWORDS):
            candidates.extend(_extract_numeric_tokens(kv.value))

    for idx, line in enumerate(lines):
        norm = _canonical_key(line)
        if any(keyword in norm for keyword in DOCUMENT_KEYWORDS):
            candidates.extend(_extract_numeric_tokens(line))
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if _looks_like_value(next_line):
                    candidates.extend(_extract_numeric_tokens(next_line))

    return _unique_preserving_order(filter(_valid_document, candidates))


def _extract_datetime(
    key_values: List[KeyValue],
    lines: List[str],
    target_keywords: Iterable[str],
) -> Optional[datetime]:
    target_keywords = tuple(target_keywords)

    date_entries: Dict[str, List[Tuple[datetime, KeyValue]]] = {}
    time_entries: Dict[str, List[Tuple[time, KeyValue]]] = {}
    combined_candidates: List[Tuple[int, datetime]] = []

    for kv in key_values:
        normalized_key = kv.normalized_key
        base = kv.base
        value = kv.value

        date = _parse_date(value)
        parsed_time = _parse_time(value)

        if date and parsed_time:
            score = _priority_score(normalized_key, target_keywords)
            combined_candidates.append((score, datetime.combine(date, parsed_time)))
            continue

        if date:
            date_entries.setdefault(base or normalized_key, []).append((date, kv))

        if parsed_time:
            time_entries.setdefault(base or normalized_key, []).append((parsed_time, kv))

    for base, date_list in date_entries.items():
        times = time_entries.get(base)
        for date_value, kv in date_list:
            if times:
                for time_value, time_kv in times:
                    score = max(
                        _priority_score(kv.normalized_key, target_keywords),
                        _priority_score(time_kv.normalized_key, target_keywords),
                    )
                    combined_candidates.append(
                        (score, datetime.combine(date_value, time_value))
                    )
            else:
                score = _priority_score(kv.normalized_key, target_keywords)
                combined_candidates.append((score, datetime.combine(date_value, time(0, 0))))

    # fallback scanning through lines for inline date/time
    for line in lines:
        dates = list(DATE_REGEX.finditer(line))
        times = list(TIME_REGEX.finditer(line))
        if dates and times:
            for date_match in dates:
                date_value = _parse_date_match(date_match)
                for time_match in times:
                    time_value = _parse_time_match(time_match)
                    if date_value and time_value:
                        score = 1
                        combined_candidates.append(
                            (score, datetime.combine(date_value, time_value))
                        )

    if not combined_candidates:
        return None

    combined_candidates.sort(key=lambda item: (-item[0], item[1]))
    return combined_candidates[0][1]


def _priority_score(normalized_key: str, target_keywords: Iterable[str]) -> int:
    normalized_key = normalized_key or ""
    score = 0
    for keyword in target_keywords:
        if keyword in normalized_key:
            score = max(score, 5)
    if "receb" in normalized_key:
        score = max(score, 5)
    if "entrega" in normalized_key or "entreg" in normalized_key:
        score = max(score, 4)
    if "termino" in normalized_key or "término" in normalized_key or "fim" in normalized_key:
        score = max(score, 3)
    if "saida" in normalized_key or "saída" in normalized_key:
        score = max(score, 2)
    return score


def _clean_receiver_value(value: str) -> Optional[str]:
    cleaned = re.sub(r"(?i)recebido(?:ra|r)?\\s*por[:,]*", "", value)
    cleaned = re.sub(r"(?i)assinatura(?: do| da)?(?: recebedor[ae])?[:,]*", "", cleaned)
    cleaned = cleaned.strip(" -:;")
    cleaned = cleaned.replace("Recebedor", "").strip(" -:;")
    return cleaned.strip() or None


def _extract_numeric_tokens(value: str) -> List[str]:
    normalized = value.replace("/", " ").replace("\\", " ")
    normalized = normalized.replace("|", " ")
    tokens = re.findall(r"[0-9]{4,}", normalized)
    if "/" in value:
        tokens.extend([token for token in value.split("/") if token.isdigit()])
    return tokens


def _valid_invoice(token: str) -> bool:
    return 5 <= len(token) <= 18


def _valid_document(token: str) -> bool:
    return 4 <= len(token) <= 25


def _unique_preserving_order(tokens: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _canonical_key(text: str) -> str:
    normalized = strip_accents(text.lower())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _looks_like_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(char.isdigit() for char in stripped):
        return True
    return len(stripped.split()) <= 4 and stripped.isupper()


def _parse_date(value: str) -> Optional[datetime.date]:
    match = DATE_REGEX.search(value)
    if not match:
        return None
    return _parse_date_match(match)


def _parse_time(value: str) -> Optional[time]:
    match = TIME_REGEX.search(value.replace(";", ":"))
    if not match:
        return None
    return _parse_time_match(match)


def _parse_date_match(match: re.Match) -> Optional[datetime.date]:
    day, month, year = match.groups()
    day_i = int(day)
    month_i = int(month)
    year_i = int(year)
    if year_i < 100:
        year_i += 2000 if year_i < 50 else 1900
    try:
        return datetime(year_i, month_i, day_i).date()
    except ValueError:
        return None


def _parse_time_match(match: re.Match) -> Optional[time]:
    hour, minute, second = match.groups()
    hour_i = int(hour)
    minute_i = int(minute)
    second_i = int(second or 0)
    if second_i >= 60 or minute_i >= 60 or hour_i >= 24:
        return None
    return time(hour_i, minute_i, second_i)


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
