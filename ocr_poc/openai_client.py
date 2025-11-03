from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image

from ocr_poc.config import AppSettings
from ocr_poc.datalab_client import DatalabApiResult
from ocr_poc.models import OCRFinalResponse, OCRPage, OCRTextLine

log = logging.getLogger(__name__)


class OpenAIOCRClient:
    """Use OpenAI Vision-capable models to extract text for validation."""

    def __init__(self, settings: AppSettings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be defined to use OpenAI pipeline.")

        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    def process_file(self, path: Path) -> DatalabApiResult:
        log.info(f"Processing file via OpenAI API: {path.name}")

        image_content = self._load_image(path)
        text_content = self._perform_ocr(image_content)

        parsed = self._convert_to_response(text_content)
        text_per_page = [page.as_single_block() for page in parsed.pages]

        return DatalabApiResult(
            request_id=f"openai-{path.stem}",
            raw={"content": text_content},
            parsed=parsed,
            text_per_page=text_per_page,
        )

    def _load_image(self, path: Path) -> str:
        try:
            with Image.open(path) as image:
                return _encode_image_to_base64(image.convert("RGB"))
        except Exception as e:
            log.error(f"Failed to load image {path}: {e}")
            raise ValueError(f"Cannot load image file: {path}")

    def _perform_ocr(self, image_base64: str) -> str:
        payload = self._build_vision_payload(image_base64)

        response = self._client.responses.create(**payload)
        result = self._extract_response_content(response)

        log.info(f"OCR completed: {result[:100]}...")
        return result

    def _build_vision_payload(self, image_base64: str) -> dict:
        text_prompt = """Você é um assistente que extrai dados de comprovantes de entrega.
Transcreva o conteúdo do canhoto, incluindo: nome do recebedor, data e hora do recebimento,
número da nota fiscal/documento e quaisquer outras informações relevantes.

Retorne os dados em formato estruturado JSON com os campos:
{
    "receiver": "Nome do recebedor (string)",
    "delivery_date": "Data de recebimento (string ISO-8601)",
    "delivery_time": "Hora de recebimento (string HH:MM:SS)",
    "invoice_numbers": ["Lista de números de notas fiscais"],
    "documents": ["Lista de outros documentos"],
    "extracted_text": "Todo o texto extraído via OCR",
    "confidence": "Nível de confiança (high, medium, low)"
}

Se alguma informação não estiver visível, use null ou uma string vazia. Não invente dados.
Seja conciso e claro na resposta, respondendo apenas em PORTUGUÊS.
"""

        return {
            "model": self._settings.openai_model,
            "temperature": 1.0,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        },
                    ],
                }
            ],
            "max_output_tokens": self._settings.openai_max_tokens,
        }

    def _extract_response_content(self, response: Any) -> str:
        if hasattr(response, "output_text"):
            return response.output_text or ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content or ""
        return ""

    def _convert_to_response(self, content: str) -> OCRFinalResponse:
        text_lines = _extract_structured_lines(content)

        if not text_lines:
            text_lines = self._extract_plain_lines(content)

        page = OCRPage(page=1, text_lines=text_lines)

        return OCRFinalResponse(
            status="complete",
            success=True,
            page_count=1,
            pages=[page],
        )

    def _extract_plain_lines(self, content: str) -> List[OCRTextLine]:
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                lines.append(OCRTextLine(text=line))
        return lines


def _encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _parse_json_content(content: str) -> Dict[str, Any] | None:
    try:
        return json.loads(content)
    except Exception:
        return None


def _format_extracted_data(data: Dict[str, Any]) -> List[str]:
    lines = []

    receiver = data.get("receiver")
    if receiver:
        lines.append(f"Recebedor: {receiver}")

    delivery_date = data.get("delivery_date")
    if delivery_date:
        lines.append(f"Data de Recebimento: {delivery_date}")

    delivery_time = data.get("delivery_time")
    if delivery_time:
        lines.append(f"Hora de Recebimento: {delivery_time}")

    invoices = data.get("invoice_numbers") or []
    if isinstance(invoices, list) and invoices:
        joined = ", ".join(str(item) for item in invoices)
        lines.append(f"Notas Fiscais: {joined}")

    documents = data.get("documents") or []
    if isinstance(documents, list) and documents:
        joined = ", ".join(str(item) for item in documents)
        lines.append(f"Documentos: {joined}")

    extracted_text = data.get("extracted_text")
    if extracted_text:
        lines.append(f"Texto Extraído: {extracted_text}")

    confidence = data.get("confidence")
    if confidence:
        lines.append(f"Confiança: {confidence}")

    return lines


def _extract_structured_lines(content: str) -> List[OCRTextLine]:
    data = _parse_json_content(content)
    if not isinstance(data, dict):
        return []

    lines = _format_extracted_data(data)
    return [OCRTextLine(text=line) for line in lines if line]
