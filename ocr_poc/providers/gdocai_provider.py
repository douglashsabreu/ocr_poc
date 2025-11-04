"""Integration with Google Document AI for Enterprise OCR."""

from __future__ import annotations

import mimetypes
from typing import Iterable, List, Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud.documentai_v1.types import BoundingPoly, NormalizedVertex
from google.cloud.documentai_v1.types import document as document_types
from google.protobuf.json_format import MessageToDict


class GoogleDocAiProvider:
    """Wrapper around Google Document AI Enterprise OCR."""

    def __init__(
        self,
        project_id: str,
        location: str,
        processor_id: str,
        extractor_processor_id: str | None = None,
    ) -> None:
        """Initialize Google Document AI provider client.
        
        Args:
            project_id: GCP project ID.
            location: Processor location (e.g., us, eu).
            processor_id: Document AI processor identifier.
            extractor_processor_id: Optional custom extractor processor ID.
            
        Raises:
            ValueError: If required parameters are missing.
        """
        if not project_id or not location or not processor_id:
            raise ValueError("Projeto, localização e processor_id são obrigatórios.")

        endpoint = f"{location}-documentai.googleapis.com"
        client_options = ClientOptions(api_endpoint=endpoint)
        self._client = documentai.DocumentProcessorServiceClient(
            client_options=client_options
        )
        self._processor_name = self._client.processor_path(
            project=project_id,
            location=location,
            processor=processor_id,
        )
        self._extractor_processor_id = extractor_processor_id

    def process_bytes(self, image_bytes: bytes, mime_type: str | None = None) -> dict:
        """Submit raw image/PDF bytes to Document AI and normalise the response.
        
        Args:
            image_bytes: Raw image or PDF file content.
            mime_type: MIME type of the document.
            
        Returns:
            Dictionary with quality metrics, extracted lines, and raw payload.
            
        Raises:
            RuntimeError: If Document AI processing fails.
        """
        clean_mime = mime_type or "application/pdf"
        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type=clean_mime,
        )

        request = documentai.ProcessRequest(
            name=self._processor_name,
            raw_document=raw_document,
            process_options=documentai.ProcessOptions(
                ocr_config=documentai.OcrConfig(
                    enable_image_quality_scores=True,
                )
            ),
        )

        try:
            response = self._client.process_document(request=request)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Falha ao processar documento no Google Document AI."
            ) from exc

        document = response.document
        quality = _extract_quality(document.pages)
        lines = _extract_lines(document)
        payload = MessageToDict(
            document._pb, preserving_proto_field_name=True, use_integers_for_enums=False
        )

        return {
            "quality": quality,
            "lines": lines,
            "raw_payload": payload,
        }

    def try_wb_extractor(self, image_bytes: bytes, mime_type: str | None = None) -> None:
        """Hook for Workbench Custom Extractor (stub for post-MVP implementation).
        
        Args:
            image_bytes: Raw image or PDF file content.
            mime_type: MIME type of the document.
            
        Returns:
            None (implementation pending).
        """
        if not self._extractor_processor_id:
            return


def guess_mime_type(filename: str | None) -> str:
    """Infer a MIME type based on the filename.
    
    Args:
        filename: Name of the file to analyze.
        
    Returns:
        MIME type string, defaults to application/octet-stream.
    """
    if not filename:
        return "application/octet-stream"
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def _extract_lines(
    document: document_types.Document,
) -> List[dict]:
    """Normalise lines returned by Document AI into a flat list.
    
    Args:
        document: Processed Document AI document object.
        
    Returns:
        List of dictionaries with text, confidence, bbox, and page.
    """
    text = document.text or ""
    results: List[dict] = []
    for page_index, page in enumerate(document.pages, start=1):
        for line in page.lines:
            layout = line.layout
            line_text = _layout_to_text(layout, text)
            if not line_text:
                continue
            bbox = _bounding_box(layout.bounding_poly)
            results.append(
                {
                    "text": line_text,
                    "confidence": float(layout.confidence or 0.0),
                    "bbox": bbox,
                    "page": page_index,
                }
            )
    return results


def _layout_to_text(layout: document_types.Document.Page.Layout, text: str) -> str:
    """Convert a text anchor into readable string.
    
    Args:
        layout: Layout object containing text anchor.
        text: Full document text for extracting segments.
        
    Returns:
        Extracted and concatenated text string.
    """
    anchor = layout.text_anchor
    if not anchor.text_segments:
        return ""
    fragments: List[str] = []
    for segment in anchor.text_segments:
        start = int(segment.start_index or 0)
        end = int(segment.end_index or 0)
        fragments.append(text[start:end])
    return "".join(fragments).strip()


def _bounding_box(
    bounding_poly: BoundingPoly | None,
) -> List[float]:
    """Convert a bounding poly into a simplified [x0, y0, x1, y1] box.
    
    Args:
        bounding_poly: Polygon defining text element boundaries.
        
    Returns:
        Normalized bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    if not bounding_poly:
        return [0.0, 0.0, 0.0, 0.0]
    vertices = _resolve_vertices(bounding_poly)
    if not vertices:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [vertex.x for vertex in vertices]
    ys = [vertex.y for vertex in vertices]
    min_x = max(min(xs), 0.0)
    min_y = max(min(ys), 0.0)
    max_x = min(max(xs), 1.0)
    max_y = min(max(ys), 1.0)
    return [min_x, min_y, max_x, max_y]


def _resolve_vertices(
    bounding_poly: BoundingPoly,
) -> Sequence[NormalizedVertex]:
    """Return normalised vertices whenever possible.
    
    Args:
        bounding_poly: Polygon with vertices.
        
    Returns:
        Sequence of normalized vertices or fallback to absolute vertices.
    """
    if bounding_poly.normalized_vertices:
        return bounding_poly.normalized_vertices
    return bounding_poly.vertices


def _extract_quality(pages: Iterable[document_types.Document.Page]) -> dict:
    """Aggregate quality scores and reasons across pages.
    
    Args:
        pages: Document pages with quality metadata.
        
    Returns:
        Dictionary with score_min, score_avg, and detected defects.
    """
    scores: List[float] = []
    defects: set[str] = set()
    for page in pages:
        quality = page.image_quality_scores
        if not quality:
            continue
        if quality.quality_score:
            scores.append(float(quality.quality_score))
        for defect in quality.detected_defects:
            reason = defect.type_ or "unknown"
            if defect.confidence:
                defects.add(f"{reason} ({defect.confidence:.2f})")
            else:
                defects.add(reason)
    score_min = min(scores) if scores else None
    score_avg = sum(scores) / len(scores) if scores else None

    return {
        "score_min": score_min,
        "score_avg": score_avg,
        "reasons": sorted(defects),
    }
