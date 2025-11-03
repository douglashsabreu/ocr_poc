from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

if False:  # pragma: no cover - only used for type checking
    from chandra.model.schema import BatchOutputItem


class ResultWriter:
    """Persists OCR outputs to disk."""

    def __init__(self, output_dir: Path, save_images: bool = True) -> None:
        self._output_dir = output_dir
        self._save_images = save_images

    def write(self, source_path: Path, page_index: int, result: "BatchOutputItem") -> Dict[str, Path]:
        target_dir = self._output_dir / source_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_page{page_index + 1}"
        markdown_path = target_dir / f"{source_path.stem}{suffix}.md"
        html_path = target_dir / f"{source_path.stem}{suffix}.html"
        raw_path = target_dir / f"{source_path.stem}{suffix}_raw.txt"
        metadata_path = target_dir / f"{source_path.stem}{suffix}_meta.json"

        markdown_path.write_text(result.markdown, encoding="utf-8")
        html_path.write_text(result.html, encoding="utf-8")
        raw_path.write_text(result.raw, encoding="utf-8")

        chunk_keys = None
        if hasattr(result.chunks, "keys"):
            chunk_keys = list(result.chunks.keys())
        elif isinstance(result.chunks, (list, tuple)):
            chunk_keys = [str(index) for index, _ in enumerate(result.chunks)]

        metadata = {
            "token_count": result.token_count,
            "page_box": result.page_box,
            "chunk_keys": chunk_keys,
            "error": result.error,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if self._save_images and result.images:
            images_dir = target_dir / f"{source_path.stem}{suffix}_images"
            images_dir.mkdir(exist_ok=True)
            for name, image in result.images.items():
                image_path = images_dir / name
                if image_path.suffix == "":
                    image_path = image_path.with_suffix(".png")
                self._save_image(image_path, image)

        return {
            "markdown": markdown_path,
            "html": html_path,
            "raw": raw_path,
            "metadata": metadata_path,
        }

    @staticmethod
    def _save_image(path: Path, image: Image.Image) -> None:
        image.save(path, format="PNG")
