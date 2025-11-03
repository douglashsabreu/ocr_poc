from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ocr_poc.datalab_client import DatalabApiResult
from ocr_poc.validation import DeliveryValidation


def build_delivery_report(
    output_path: Path,
    source_file: Path,
    result: DatalabApiResult,
    validation: DeliveryValidation,
) -> None:
    """Gera um comprovante de recebimento em PDF com layout aprimorado."""

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Comprovante de Recebimento",
        author="ocr-poc",
    )

    styles = _build_stylesheet()
    story = []

    # Página 1 - resumo textual
    story.append(Paragraph("Comprovante de Recebimento", styles["Title"]))
    story.append(Paragraph("Resumo da Validação", styles["Section"]))
    story.append(Spacer(1, 6))
    story.append(_build_status_badge(validation.status, styles))
    story.append(Spacer(1, 10))
    story.append(_build_summary_table(source_file, result, validation, styles))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detalhes Extraídos", styles["Section"]))
    story.extend(_render_info_lists(validation, styles))

    if validation.issues:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Pendências Encontradas", styles["Section"]))
        for issue in validation.issues:
            story.append(Paragraph(f"• {issue}", styles["Body"]))

    if validation.raw_text_sample:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Amostra do Texto OCR", styles["Section"]))
        sample_text = "<br/>".join(
            _escape_html(line) for line in validation.raw_text_sample[:15]
        )
        story.append(Paragraph(sample_text, styles["Small"]))

    # Página 2 - imagem
    story.append(PageBreak())
    story.append(Paragraph("Imagem do Canhoto", styles["Section"]))
    story.append(Paragraph("Visualização do documento analisado.", styles["Body"]))
    story.append(Spacer(1, 10))
    story.append(_build_image_frame(source_file, doc.width))

    doc.build(story)


def _build_stylesheet() -> StyleSheet1:
    base = getSampleStyleSheet()
    accent = colors.HexColor("#0F9D58")
    dark = colors.HexColor("#1F2933")

    base.add(
        ParagraphStyle(
            name="Section",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=dark,
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=dark,
        )
    )
    base.add(
        ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
    )
    base.add(
        ParagraphStyle(
            name="Badge",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.white,
            alignment=1,
            spaceBefore=2,
            spaceAfter=2,
            leading=14,
        )
    )
    title_style = base["Title"]
    title_style.fontName = "Helvetica-Bold"
    title_style.textColor = dark
    title_style.fontSize = 20
    title_style.spaceAfter = 6

    return base


def _build_status_badge(status: str, styles: StyleSheet1) -> Paragraph:
    lookup = {
        "ok": ("VALIDAÇÃO APROVADA", colors.HexColor("#0F9D58")),
        "missing_data": ("VALIDAÇÃO COM PENDÊNCIAS", colors.HexColor("#FB8C00")),
        "illegible": ("DOCUMENTO ILEGÍVEL", colors.HexColor("#E53935")),
    }
    label, color = lookup.get(status.lower(), ("STATUS DESCONHECIDO", colors.HexColor("#6366F1")))
    style = ParagraphStyle(name="DynamicBadge", parent=styles["Badge"], backColor=color)
    return Paragraph(label, style)


def _build_summary_table(
    source_file: Path,
    result: DatalabApiResult,
    validation: DeliveryValidation,
    styles: StyleSheet1,
) -> Table:
    entrega_data, entrega_hora = _split_datetime(validation.received_at)
    envio_data, envio_hora = _split_datetime(validation.shipment_at)

    data = [
        ["Arquivo origem", source_file.name],
        ["UUID do comprovante", validation.reference_id or "-"],
        ["Data de geração", _format_datetime(datetime.now())],
        ["ID da solicitação (API)", result.request_id or "-"],
        ["Recebedor", validation.receiver or "Não identificado"],
        ["Data da entrega", entrega_data or "Não identificada"],
        ["Hora da entrega", entrega_hora or "Não identificada"],
        ["Data da expedição", envio_data or "Não identificada"],
        ["Hora da expedição", envio_hora or "Não identificada"],
    ]

    table = Table(data, colWidths=[60 * mm, 95 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D1FAE5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F5132")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#99F6E4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#34D399")),
                ("GRID", (0, 1), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ]
        )
    )
    return table


def _render_info_lists(
    validation: DeliveryValidation,
    styles: StyleSheet1,
) -> list:
    elements = []
    elements.extend(_render_list_section("Notas fiscais / NF-e", validation.invoice_numbers, styles["Body"]))
    elements.append(Spacer(1, 4))
    elements.extend(_render_list_section("Documentos relacionados", validation.document_numbers, styles["Body"]))
    return elements


def _render_list_section(
    title: str,
    items: Sequence[str],
    style: ParagraphStyle,
) -> list:
    if not items:
        return [Paragraph(f"{title}: <i>Não disponível</i>", style)]
    return [Paragraph(f"{title}: {', '.join(items)}", style)]


def _build_image_frame(image_path: Path, available_width: float):
    frame_width = available_width
    frame_height = 180 * mm

    if image_path.exists():
        try:
            pil_img = PILImage.open(image_path)
            img_width, img_height = pil_img.size
            aspect = img_height / img_width if img_width else 1
            target_width, target_height = _fit_within(frame_width - 6 * mm, frame_height - 6 * mm, aspect)
            image_flowable = ReportImage(str(image_path), width=target_width, height=target_height)
            image_flowable.hAlign = "CENTER"
            cell_content = [Spacer(1, 6), image_flowable, Spacer(1, 6)]
        except Exception:
            cell_content = [Paragraph("Imagem indisponível para visualização.", getSampleStyleSheet()["BodyText"])]
    else:
        cell_content = [Paragraph("Arquivo de imagem não encontrado.", getSampleStyleSheet()["BodyText"])]

    table = Table([[cell_content]], colWidths=[frame_width], rowHeights=[frame_height])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ]
        )
    )
    return table


def _fit_within(max_width: float, max_height: float, aspect: float) -> Tuple[float, float]:
    """Calcula dimensões proporcionais para caber no quadro."""
    width = max_width
    height = width * aspect
    if height > max_height:
        height = max_height
        width = height / aspect if aspect else max_width
    return width, height


def _split_datetime(value: datetime | None) -> Tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y"), value.strftime("%H:%M:%S")
    try:
        return value.strftime("%d/%m/%Y"), ""
    except AttributeError:
        return str(value), ""


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y %H:%M:%S")
    try:
        return value.strftime("%d/%m/%Y")
    except AttributeError:
        return str(value)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
