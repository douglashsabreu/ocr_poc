from __future__ import annotations

import importlib
import os
from typing import Protocol, TYPE_CHECKING

from PIL import Image

from ocr_poc.config import AppSettings

if TYPE_CHECKING:
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchOutputItem


class OCRClient(Protocol):
    """Protocol that defines the OCR interface."""

    def run(self, image: Image.Image) -> "BatchOutputItem":
        ...


class ChandraOCRClient:
    """Concrete OCR client built on top of the Chandra library."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._manager = self._create_manager()

    def run(self, image: Image.Image) -> "BatchOutputItem":
        from chandra.model.schema import BatchInputItem

        batch = BatchInputItem(image=image, prompt_type="ocr_layout")
        generate_kwargs = {
            "include_images": self._settings.include_images,
            "include_headers_footers": self._settings.include_headers_footers,
        }

        if self._settings.max_output_tokens is not None:
            generate_kwargs["max_output_tokens"] = self._settings.max_output_tokens

        if self._settings.max_workers is not None:
            generate_kwargs["max_workers"] = self._settings.max_workers

        if self._settings.max_retries is not None:
            generate_kwargs["max_retries"] = self._settings.max_retries

        result = self._manager.generate([batch], **generate_kwargs)[0]
        return result

    def _create_manager(self) -> "InferenceManager":
        self._apply_runtime_environment()
        from chandra.model import InferenceManager

        return InferenceManager(method=self._settings.inference_method)

    def _apply_runtime_environment(self) -> None:
        os.environ["VLLM_API_KEY"] = self._settings.datalab_api_key
        os.environ["VLLM_API_BASE"] = self._settings.datalab_api_base
        os.environ["VLLM_MODEL_NAME"] = self._settings.datalab_model_name

        import chandra.model.vllm as vllm_module
        import chandra.settings as settings_module

        importlib.reload(settings_module)
        importlib.reload(vllm_module)

