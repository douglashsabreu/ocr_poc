"""
Image transcription utilities for logistics multi-agent system
Copyright (c) 2025 FreteFY. All rights reserved.
"""

import logging
import time
from typing import Optional, Tuple

import openai
import requests
from requests.auth import HTTPBasicAuth

from .azure_blob_saver import upload_bytes_get_address

log = logging.getLogger(__name__)


def _download_with_auth(image_url: str, auth: Tuple[str, str]) -> requests.Response:
    """
    Download image with authentication.

    Args:
        image_url: URL of image file
        auth: Authentication tuple (token/username, password)

    Returns:
        HTTP response object
    """
    if auth[1] == "":
        headers = {"Authorization": f"Bearer {auth[0]}"}
        return requests.get(image_url, headers=headers)
    else:
        http_auth = HTTPBasicAuth(auth[0], auth[1])
        return requests.get(image_url, auth=http_auth)


def _download_image(
    image_url: str, auth: Optional[Tuple[str, str]] = None, authenticate: bool = False
) -> bytes:
    """
    Download image from URL.

    Args:
        image_url: URL of the image file to download
        auth: Optional authentication tuple (token/username, password)
        authenticate: Whether authentication is required

    Returns:
        Image content as bytes

    Raises:
        requests.RequestException: If download fails
    """
    if authenticate and auth:
        response = _download_with_auth(image_url, auth)
    else:
        response = requests.get(image_url)

    response.raise_for_status()
    return response.content


def _upload_to_azure_blob(
    content: bytes,
    storage_account_name: str,
    storage_account_key: str,
    container_name: str,
) -> str:
    """
    Upload image to Azure Blob Storage and return SAS URL.

    Args:
        content: Image content as bytes
        storage_account_name: Azure storage account name
        storage_account_key: Azure storage account key
        container_name: Azure storage container name

    Returns:
        URL of uploaded image with SAS token

    Raises:
        Exception: If upload fails
    """
    blob_name = f"image_{int(time.time())}.jpg"
    blob_url = upload_bytes_get_address(
        data_bytes=content,
        blob_name=blob_name,
        container_name=container_name,
        storage_account_name=storage_account_name,
        storage_account_key=storage_account_key,
        ttl_days=1.0,
        content_type="image/jpeg",
    )
    log.debug(f"Image uploaded to Azure Blob Storage: {blob_url}")
    return blob_url


def _build_vision_payload(image_url: str) -> dict:
    """
    Build payload for OpenAI Vision API request.

    Args:
        image_url: URL of the image to process

    Returns:
        API request payload
    """
    text_prompt = """You are responsible for analyzing proof-of-delivery documents.
    Review the proof document and extract the relevant delivery information.
    Assess the image quality and the reliability of the extracted information.
    Issuance date is not the receipt/delivery date.

    The proof document must be analyzed and the extracted text must be returned in JSON format.
    The JSON must follow exactly the format below:

    {
        "data": {
            "receiver": {
                "value": "Receiver name (string)",
                "confidence": 0.0
            },
            "delivery_date": {
                "value": "Receipt date (string ISO-8601)",
                "confidence": 0.0
            },
            "delivery_time": {
                "value": "Receipt time (string HH:MM:SS)",
                "confidence": 0.0
            },
            "invoice_number": {
                "value": "Invoice or document number (string)",
                "confidence": 0.0
            },
            "signature": {
                "value": true,
                "confidence": 0.0
            }
        },
        "image_quality": {
            "value": 0.0,
            "confidence": 0.0
        },
        "validation": {
            "is_valid": true,
            "confidence": 0.0
        }
    }
    If any information is not visible, use null or an empty string. Do not fabricate data.
    Be concise and clear in the response, replying only in PORTUGUESE.
    """

    payload = {
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        # "effort": "low",
        # "verbosity": "medium",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 2000,
    }
    return payload


def _perform_ocr(image_url: str, api_key: str) -> str:
    """
    Perform OCR on image using OpenAI Vision API.

    Args:
        image_url: URL of the image (must be accessible)
        api_key: OpenAI API key

    Returns:
        Extracted text from image

    Raises:
        Exception: If OCR fails
    """
    client = openai.OpenAI(api_key=api_key)
    payload = _build_vision_payload(image_url)

    response = client.chat.completions.create(**payload)
    result = response.choices[0].message.content
    log.info(f"OCR completed: {result[:100]}...")
    return result


def transcribe_uploaded_image(image_url: str, api_key: str) -> str:
    """Transcribe image available through an accessible URL.

    Args:
        image_url: URL of the uploaded image
        api_key: OpenAI API key

    Returns:
        Extracted text from image

    Raises:
        Exception: If OCR fails
    """
    return _perform_ocr(image_url, api_key)


def transcribe_image(
    image_url: str,
    storage_account_name: str,
    storage_account_key: str,
    container_name: str,
    api_key: str,
    auth: Optional[Tuple[str, str]] = None,
    authenticate: bool = False,
) -> str:
    """
    Transcribe image from URL using OpenAI Vision API with OCR.

    Args:
        image_url: URL of the image file
        storage_account_name: Azure storage account name
        storage_account_key: Azure storage account key
        container_name: Azure storage container name
        api_key: OpenAI API key
        auth: Optional authentication tuple
        authenticate: Whether authentication is required

    Returns:
        Extracted text from image

    Raises:
        requests.RequestException: If download fails
        Exception: If transcription fails
    """
    log.info(f"Transcribing image from URL {image_url}")

    content = _download_image(image_url, auth, authenticate)
    azure_image_url = _upload_to_azure_blob(
        content, storage_account_name, storage_account_key, container_name
    )
    return _perform_ocr(azure_image_url, api_key)
