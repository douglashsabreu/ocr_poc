## Datalab OCR Proof of Concept

This project evaluates the OCR capabilities offered by **Datalab** and **Google Document AI**, wrapping the APIs in a modular, SOLID-oriented codebase. Two execution paths are available:

- `datalab_api` (default): calls the REST endpoint `/api/v1/ocr` and parses the response into structured outputs.
- `openai_api`: converts the image locally and sends it to an OpenAI multimodal model (e.g., `gpt-4o-mini`) for extraction.
- `chandra`: runs the open-source `chandra-ocr` package locally or against a compatible vLLM server when deeper control is required.
- `gdocai` (MVP): integra com o Google Document AI Enterprise Document OCR com quality score habilitado para gate e extração unificada.

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) on your PATH
- Environment variables:
  - `DATALAB_API_KEY`: API key issued by Datalab.
  - `DATALAB_API_BASE`: API base URL (defaults to `https://www.datalab.to/api/v1`).
  - `OPENAI_API_KEY`: required when `PIPELINE_MODE=openai_api`.
  - `GDOC_PROJECT_ID`, `GDOC_LOCATION`, `GDOC_PROCESSOR_ID`: identificam o processor do Google Document AI (Enterprise Document OCR). Consulte a [documentação oficial](https://cloud.google.com/document-ai/docs/create-processor#documentai_create_processor-python) para criar/gerir processors.
  - `QUALITY_MIN_SCORE`, `FIELD_MIN_CONFIDENCE`: limiares mínimos de qualidade e confiança aplicados pelo quality gate e validação.
  - `USE_GDOC_AI_GATE`: quando `true`, roda o quality gate do Google mesmo no modo `datalab_api` antes de acionar APIs “lentas”.

### Installation

```bash
uv sync
```

`uv` provisions/updates `.venv` and installs every dependency, including `chandra-ocr` from GitHub.

### Usage

```bash
uv run python main.py [--mode gdocai|datalab_api|chandra|openai_api] [--images-dir <dir>] [--output-dir <dir>] [--use-gate/--no-use-gate]
```

What happens during a run:

- Files from `images_example` (override via CLI `--images-dir` or env `IMAGES_DIR`) are enumerated.
- The selected engine roda a normalização unificada (`lines`, `quality`, `raw_payload`).
- Um quality gate avalia `QUALITY_MIN_SCORE`; se reprovado, devolve razões e orientações (reflexo, foco, etc.).
- A extração heurística (`date`, `recipient_name`, `signature_present`, `tracking_code`) roda sobre as linhas normalizadas e gera `_validation.json` com decisão (`OK`, `NEEDS_REVIEW`, `REPROVADO`).
- Artefatos são salvos em `outputs/<arquivo>/` (override via CLI `--output-dir` ou env `OUTPUT_DIR`):
  - `<arquivo>_ocr.json`: payload normalizado com latências, campos e raw payload.
  - `<arquivo>_ocr.txt`: resumo human readable (qualidade, decisão, campos, texto OCR).
  - `<arquivo>_validation.json`: resultado estruturado do engine de validação.
  - `<arquivo>_validation.pdf`: relatório em PDF com status, campos e qualidade.
  - `<arquivo>_gdocai_raw.json`: raw payload do Document AI (quando disponível) para depuração.

Structured logs com `run_summary file=... mode=... decision=... latencies=...` são emitidos para facilitar observabilidade.

### Como rodar com Google Document AI

1. Siga a [documentação oficial](https://cloud.google.com/document-ai/docs/create-processor#documentai_create_processor-python) para:
   - Habilitar a API `documentai.googleapis.com` no projeto.
   - Criar um processor `OCR_PROCESSOR` ou *Enterprise Document OCR* na região desejada.
   - Habilitar [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) apontando `GOOGLE_APPLICATION_CREDENTIALS` para o JSON da service account.
   - (Opcional) Liste os tipos de processor via `fetchProcessorTypes` para validar disponibilidade.
2. Preencha `.env` com `GDOC_PROJECT_ID`, `GDOC_LOCATION`, `GDOC_PROCESSOR_ID` e defina `PIPELINE_MODE=gdocai`.
3. Execute:

```bash
uv run python main.py --mode gdocai
```

Durante o processamento, o quality gate do Google retorna `score_min`, `score_avg` e sugestões de melhoria de captura. Limiar padrão `QUALITY_MIN_SCORE=0.55` (ajuste conforme SLA).

### Fast-path com quality gate + Datalab API

Para usar o quality gate do Google antes de enviar arquivos ao Datalab:

```bash
uv run python main.py --mode datalab_api --use-gate
```

Se `score_min < QUALITY_MIN_SCORE`, o pipeline retorna reprovação imediata com razões (ex.: `motion_blur (0.42)`) e orientações (`Evite movimentar o dispositivo...`). Quando aprovado, o pipeline executa o Datalab API e reaproveita a mesma estrutura de validação.

### Experimento A/B rápido

O script `scripts/ab_compare.py` roda os mesmos arquivos em múltiplos modos e gera um CSV com decisões, qualidade e latências:

```bash
uv run python scripts/ab_compare.py --images-dir images_example --modes datalab_api,gdocai --use-gate
```

O CSV é salvo em `outputs/ab/ab_compare.csv` e o console exibe um resumo agregado (`OK=`, `NEEDS_REVIEW=`, etc.).

### Configuration

Tune behaviour through environment variables:

| Variable | Description |
| --- | --- |
| `PIPELINE_MODE` | Select `datalab_api` (default), `openai_api`, or `chandra`. |
| `GDOC_PROJECT_ID`, `GDOC_LOCATION`, `GDOC_PROCESSOR_ID` | Identificadores do processor do Google Document AI. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Caminho local para o JSON da service account (Application Default Credentials). |
| `QUALITY_MIN_SCORE` | Limiar mínimo aceito pelo quality gate (0 a 1). |
| `FIELD_MIN_CONFIDENCE` | Confiança mínima por campo obrigatório. |
| `USE_GDOC_AI_GATE` | `true` para rodar o quality gate antes do envio ao Datalab. |
| `API_PAGE_RANGE` / `API_MAX_PAGES` | Restrict pages submitted to `/ocr`. |
| `API_SKIP_CACHE` | Force re-processing when `true`. |
| `API_POLL_INTERVAL_SECONDS` | Polling cadence between status checks (default `2`). |
| `API_MAX_POLL_ATTEMPTS` | Maximum number of status checks (default `60`). |
| `OPENAI_MODEL` | Model used when `PIPELINE_MODE=openai_api` (`gpt-4o-mini` by default). |
| `OPENAI_MAX_TOKENS` | Cap for tokens returned by the OpenAI response (`1024` por padrão). |
| `CHANDRA_INFERENCE_METHOD` | `vllm` (default) or `hf` when using the Chandra path. |
| `INCLUDE_IMAGES` | Persist extracted images in Chandra mode. |
| `MAX_OUTPUT_TOKENS`, `MAX_WORKERS`, `MAX_RETRIES` | Fine-tune Chandra requests. |
| `OUTPUT_DIR` | Directory for generated artefacts. |

### Observations

- REST calls require the `X-API-Key` header; the client inserts it automatically from `.env`.
- The parser layer (backed by Pydantic models) normalises the API response, removes duplicate lines, and produces user-facing summaries.
- Logging runs at `INFO` level, produzindo linhas estruturadas com arquivo, modo, decisão, latências e scores de qualidade.
- `gdocai` salva o payload bruto (`*_gdocai_raw.json`) quando disponível, facilitando depuração.
- A extração mínima (`fields.py`) é heurística: adapte com regras adicionais ou conecte um Custom Extractor do Workbench futuramente via `GDOC_EXTRACTOR_PROCESSOR_ID`.
- Para OpenAI, set `PIPELINE_MODE=openai_api` e exporte `OPENAI_API_KEY`.
