from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".bmp",
}


class ImageRepository:
    """Provides an iterable collection of image paths for OCR processing."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def iter_files(self) -> Iterator[Path]:
        """Yield all supported files inside the root directory."""
        if not self._root.exists():
            raise FileNotFoundError(f"Input path not found: {self._root}")

        if self._root.is_file():
            if self._is_supported(self._root):
                yield self._root
            return

        for extension in SUPPORTED_EXTENSIONS:
            yield from self._root.glob(f"*{extension}")
            yield from self._root.glob(f"*{extension.upper()}")

    def list_files(self) -> list[Path]:
        """Return a sorted list of supported files."""
        return sorted({path.resolve() for path in self.iter_files()})

    @staticmethod
    def _is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith(
            "."
        )

