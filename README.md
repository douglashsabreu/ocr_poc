## Datalab OCR Proof of Concept

This project evaluates the OCR capabilities offered by **Datalab**, wrapping the API in a modular, SOLID-oriented codebase. Two execution paths are available:

- `datalab_api` (default): calls the REST endpoint `/api/v1/ocr` and parses the response into structured outputs.
- `chandra`: runs the open-source `chandra-ocr` package locally or against a compatible vLLM server when deeper control is required.

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) on your PATH
- Environment variables:
  - `DATALAB_API_KEY`: API key issued by Datalab.
  - `DATALAB_API_BASE`: API base URL (defaults to `https://www.datalab.to/api/v1`).

### Installation

```bash
uv sync
```

`uv` provisions/updates `.venv` and installs every dependency, including `chandra-ocr` from GitHub.

### Usage

```bash
uv run python main.py
```

What happens during a run:

- Files from `images_example` (override via `IMAGES_DIR`) are enumerated.
- Each file is submitted to the selected backend (REST API or Chandra pipeline).
- Results are saved under `outputs/` (override via `OUTPUT_DIR`). For the REST path, outputs include:
  - A JSON dump mirroring the API payload.
  - A `.txt` companion containing a summary (status, page count) and the extracted lines per page in a reader-friendly format.

### Configuration

Tune behaviour through environment variables:

| Variable | Description |
| --- | --- |
| `PIPELINE_MODE` | Select `datalab_api` (default) or `chandra`. |
| `API_PAGE_RANGE` / `API_MAX_PAGES` | Restrict pages submitted to `/ocr`. |
| `API_SKIP_CACHE` | Force re-processing when `true`. |
| `API_POLL_INTERVAL_SECONDS` | Polling cadence between status checks (default `2`). |
| `API_MAX_POLL_ATTEMPTS` | Maximum number of status checks (default `60`). |
| `CHANDRA_INFERENCE_METHOD` | `vllm` (default) or `hf` when using the Chandra path. |
| `INCLUDE_IMAGES` | Persist extracted images in Chandra mode. |
| `MAX_OUTPUT_TOKENS`, `MAX_WORKERS`, `MAX_RETRIES` | Fine-tune Chandra requests. |
| `OUTPUT_DIR` | Directory for generated artefacts. |

### Observations

- REST calls require the `X-API-Key` header; the client inserts it automatically from `.env`.
- The parser layer (backed by Pydantic models) normalises the API response, removes duplicate lines, and produces user-facing summaries.
- Logging runs at `INFO` level and reports progress, warnings, and failures.
