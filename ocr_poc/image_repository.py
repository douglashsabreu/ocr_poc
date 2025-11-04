"""Repository for managing and accessing image files for OCR processing."""

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
        """Initialize repository with a root directory or file path.
        
        Args:
            root: Path to a directory containing images or a single image file.
        """
        self._root = root

    def iter_files(self) -> Iterator[Path]:
        """Yield all supported files inside the root directory.
        
        Yields:
            Path objects for each supported image file found.
            
        Raises:
            FileNotFoundError: If the root path does not exist.
        """
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
        """Return a sorted list of supported files.
        
        Returns:
            Sorted list of absolute paths to supported image files.
        """
        return sorted({path.resolve() for path in self.iter_files()})

    @staticmethod
    def _is_supported(path: Path) -> bool:
        """Check if a file has a supported extension and is not hidden.
        
        Args:
            path: Path to check.
            
        Returns:
            True if the file extension is supported and filename doesn't start with dot.
        """
        return path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith(
            "."
        )

