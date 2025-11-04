"""Simple A/B runner between Datalab API and Google Document AI."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from ocr_poc.config import AppSettings
from ocr_poc.document_pipeline import DocumentPipeline
from ocr_poc.document_writer import DocumentResultWriter
from ocr_poc.image_repository import ImageRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda comparação A/B entre modos de OCR.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images_example"),
        help="Diretório com as imagens a processar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ab"),
        help="Diretório base onde os resultados e CSV serão persistidos.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="datalab_api,gdocai",
        help="Lista de modos separados por vírgula para comparar.",
    )
    parser.add_argument(
        "--use-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Força o uso do quality gate quando aplicável.",
    )
    return parser.parse_args()


def run_mode(mode: str, args: argparse.Namespace) -> List[Dict[str, object]]:
    overrides: Dict[str, object] = {
        "PIPELINE_MODE": mode,
        "IMAGES_DIR": str(args.images_dir),
        "OUTPUT_DIR": str(args.output_dir / mode),
    }
    if args.use_gate is not None:
        overrides["USE_GDOC_AI_GATE"] = args.use_gate

    settings = AppSettings(**overrides)
    logger = logging.getLogger(f"ocr_poc.ab.{mode}")
    repository = ImageRepository(settings.images_dir)
    pipeline = DocumentPipeline(settings, repository, logger=logger)
    writer = DocumentResultWriter(
        settings.output_dir,
        field_min_confidence=settings.field_min_confidence,
        quality_min_score=settings.quality_min_score,
    )

    outcomes: List[Dict[str, object]] = []
    try:
        for outcome in pipeline.run():
            saved = writer.write(outcome)
            validation = saved.get("validation_data")
            if not validation:
                continue
            outcomes.append(
                {
                    "file": outcome.source_path.name,
                    "mode": mode,
                    "decision": validation.decision,
                    "decision_score": validation.decision_score,
                    "quality_score_min": validation.quality.get("score_min"),
                    "quality_score_avg": validation.quality.get("score_avg"),
                    "latency_total": outcome.latencies.get("total"),
                    "latency_engine": outcome.latencies.get(mode) or outcome.latencies.get("gdocai"),
                }
            )
    finally:
        pipeline.close()
    return outcomes


def write_csv(output_dir: Path, rows: List[Dict[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ab_compare.csv"
    headers = [
        "file",
        "mode",
        "decision",
        "decision_score",
        "quality_score_min",
        "quality_score_avg",
        "latency_total",
        "latency_engine",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def print_summary(rows: List[Dict[str, object]]) -> None:
    summary: Dict[str, Dict[str, int]] = {}
    for row in rows:
        mode = str(row["mode"])
        decision = str(row["decision"])
        summary.setdefault(mode, {}).setdefault(decision, 0)
        summary[mode][decision] += 1

    for mode, decisions in summary.items():
        logging.info("Resumo %s: %s", mode, ", ".join(f"{k}={v}" for k, v in decisions.items()))


def main() -> None:
    args = parse_args()
    load_dotenv()
    configure_logging()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("Nenhum modo fornecido para comparação.")

    all_rows: List[Dict[str, object]] = []
    for mode in modes:
        all_rows.extend(run_mode(mode, args))

    csv_path = write_csv(args.output_dir, all_rows)
    logging.info("CSV gerado em %s", csv_path)
    print_summary(all_rows)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
