from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from chandra.input import load_file


@dataclass(frozen=True)
class ImagePayload:
    """Represents a single page or image to be processed."""

    source: Path
    page_index: int
    image: Image.Image


class ImageLoader:
    """Loads images and PDF pages into in-memory payloads."""

    def __init__(self, page_range: str | None = None) -> None:
        self._config = {"page_range": page_range} if page_range else {}

    def load(self, path: Path) -> Iterator[ImagePayload]:
        images = load_file(str(path), self._config)
        for index, image in enumerate(images):
            yield ImagePayload(source=path, page_index=index, image=image)

