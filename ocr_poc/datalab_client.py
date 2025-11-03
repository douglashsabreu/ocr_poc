from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import httpx

from ocr_poc.config import AppSettings
from ocr_poc.models import OCRFinalResponse


@dataclass(frozen=True)
class DatalabApiResult:
    """Holds the final OCR payload returned by the Datalab API."""

    request_id: str
    raw: Dict[str, Any]
    parsed: OCRFinalResponse
    text_per_page: List[str]

    @property
    def status(self) -> str:
        return str(self.raw.get("status", "")).lower()

    @property
    def success(self) -> bool:
        success = self.raw.get("success")
        return bool(success) if success is not None else self.status == "complete"


class DatalabApiClient:
    """Handles communication with the Datalab OCR API."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=settings.api_http_timeout_seconds)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def process_file(self, path: Path) -> DatalabApiResult:
        """Submit a file to the OCR endpoint and poll until completion."""
        initial = self._submit_request(path)
        request_id = initial["request_id"]
        check_url = initial["request_check_url"]
        final_payload = self._poll_request(check_url)
        parsed = OCRFinalResponse.model_validate(final_payload)
        text_per_page = self._extract_text(parsed)
        return DatalabApiResult(
            request_id=request_id,
            raw=final_payload,
            parsed=parsed,
            text_per_page=text_per_page,
        )

    def _submit_request(self, path: Path) -> Dict[str, Any]:
        mime_type, _ = mimetypes.guess_type(path.name)
        content_type = mime_type or "application/octet-stream"

        with path.open("rb") as file_handle:
            files = {"file": (path.name, file_handle, content_type)}
            form = self._build_form_payload()
            response = self._client.post(
                self._endpoint_url,
                files=files,
                data=form,
                headers=self._headers,
            )

        if response.is_error:
            message = response.text
            raise RuntimeError(
                f"Falha ao enviar OCR (status {response.status_code}): {message}"
            )

        payload = response.json()
        if not payload.get("request_id") or not payload.get("request_check_url"):
            raise RuntimeError(f"Resposta inesperada da API: {payload}")
        if payload.get("success") is False:
            raise RuntimeError(f"Falha ao enviar OCR: {payload.get('error')}")
        return payload

    def _poll_request(self, url: str) -> Dict[str, Any]:
        for attempt in range(self._settings.api_max_poll_attempts):
            response = self._client.get(url, headers=self._headers)
            if response.is_error:
                message = response.text
                raise RuntimeError(
                    f"Falha ao checar OCR (status {response.status_code}): {message}"
                )

            payload = response.json()
            status = str(payload.get("status", "")).lower()

            if status == "complete":
                if payload.get("success") is False:
                    raise RuntimeError(f"OCR retornou erro: {payload.get('error')}")
                return payload

            if status in {"failed", "error"}:
                raise RuntimeError(f"OCR falhou: {payload.get('error')}")

            time.sleep(self._settings.api_poll_interval_seconds)

        raise TimeoutError(
            f"OCR não concluiu após "
            f"{self._settings.api_max_poll_attempts} tentativas."
        )

    def _build_form_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self._settings.api_page_range:
            payload["page_range"] = self._settings.api_page_range
        if self._settings.api_max_pages is not None:
            payload["max_pages"] = str(self._settings.api_max_pages)
        if self._settings.api_skip_cache:
            payload["skip_cache"] = "true"
        if self._settings.api_langs:
            payload["langs"] = self._settings.api_langs
        return payload

    def _extract_text(self, response: OCRFinalResponse) -> List[str]:
        pages = response.pages or []
        text_per_page: List[str] = []
        for page in pages:
            text_per_page.append(page.as_single_block())

        return text_per_page

    @property
    def _endpoint_url(self) -> str:
        return self._join_url(self._settings.datalab_api_base, self._settings.api_endpoint)

    @staticmethod
    def _join_url(base: str, endpoint: str) -> str:
        base = base.rstrip("/")
        endpoint = endpoint.strip("/")
        return f"{base}/{endpoint}"

    @property
    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self._settings.datalab_api_key}
