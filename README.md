## OCR Proof of Concept com Datalab

Este projeto executa um fluxo de OCR sobre arquivos de imagem e PDF utilizando os serviços da **Datalab**, mantendo uma arquitetura modular e aderente a princípios SOLID. Há dois modos de operação:

- `datalab_api` (padrão): utiliza o endpoint REST `/api/v1/ocr`.
- `chandra`: usa o pacote `chandra-ocr` executando inferência local ou via servidor vLLM compatível com OpenAI.

### Pré-requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) disponível no PATH
- Variáveis de ambiente:
  - `DATALAB_API_KEY`: chave fornecida pela Datalab.
  - `DATALAB_API_BASE`: endpoint base da API (padrão `https://www.datalab.to/api/v1`).

### Instalação

```bash
uv sync
```

O comando acima cria/atualiza o `.venv` local e instala todas as dependências (incluindo o pacote `chandra-ocr` a partir do repositório GitHub).

### Execução

```bash
uv run python main.py
```

O pipeline fará:

- Leitura dos arquivos suportados em `images_example` (padrão, configurável via `IMAGES_DIR`).
- Envio de cada arquivo para a API (ou modelo local, conforme modo).
- Escrita dos resultados em `outputs/` (configurável via `OUTPUT_DIR`), gerando JSON completo da resposta e um `.txt` com texto agregado por página.

### Configuração adicional

As seguintes variáveis opcionais permitem ajustar o comportamento:

| Variável | Descrição |
| --- | --- |
| `PIPELINE_MODE` | Define o modo (`datalab_api` ou `chandra`). |
| `API_PAGE_RANGE` / `API_MAX_PAGES` | Limita páginas enviadas ao endpoint `/ocr`. |
| `API_SKIP_CACHE` | Define se o cache deve ser ignorado (`false` padrão). |
| `API_POLL_INTERVAL_SECONDS` | Intervalo de polling do status (padrão 2s). |
| `API_MAX_POLL_ATTEMPTS` | Quantidade máxima de verificações (padrão 60). |
| `CHANDRA_INFERENCE_METHOD` | `"vllm"` (padrão) ou `"hf"` para o modo chandra. |
| `INCLUDE_IMAGES` | Salva imagens extraídas quando no modo chandra. |
| `MAX_OUTPUT_TOKENS`, `MAX_WORKERS`, `MAX_RETRIES` | Ajustes finos do modo chandra. |
| `OUTPUT_DIR` | Pasta onde os resultados serão persistidos. |

### Observações

- Os endpoints documentados em [documentation.datalab.to](https://documentation.datalab.to) exigem o header `X-API-Key` com a chave fornecida.
- O pipeline registra logs em nível `INFO` com o progresso de cada arquivo e sinaliza falhas de API.
