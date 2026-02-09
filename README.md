# evalbenchs

CLI utilities to run LLMs against domain benchmarks with OpenRouter.

## Benchmarks included

See `benchmarks.yaml` for the full list of benchmarks, sources, and the system prompt policy.
If a benchmark does not provide a system prompt explicitly, the base system prompt is used.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="..."
```

GigaChat (used for models with the `gigachat/` prefix) uses the official SDK and
credentials-based auth:

```bash
export GIGACHAT_CREDENTIALS="..."
# or legacy:
export GIGACHAT_API_KEY="..."
```

Optional:

```bash
export GIGACHAT_SCOPE="GIGACHAT_API_CORP"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

To disable SSL verification for GigaChat (e.g. custom certificate):

```bash
export GIGACHAT_VERIFY_SSL_CERTS="false"
```
## Inspect samples / system prompts

Inspect a couple of samples and check if a dataset embeds a system prompt:

```bash
inspect-benchmarks --bench 1 --samples 2
```

## Run benchmarks

Run all benchmarks with the default models (GigaChat 2 Max v.28 via GigaChat API and GPT-4o-mini via OpenRouter):

```bash
run-benchmarks --sample-size 300 --max-concurrency 4
```

`--max-concurrency` limits how many requests are in-flight to a provider for a single
benchmark+model run. It does not cap the number of benchmarks/models executed in
parallel.

Run a subset of benchmarks (by id), e.g. only benchmarks 1 and 3:

```bash
run-benchmarks --bench 1 3 --sample-size 300
```

Run a quick smoke test (2-3 samples) for a single benchmark:

```bash
run-benchmarks --bench 1 --sample-size 3
```

Output is written to `runs/<benchmark-id>_<model>.jsonl`. A consolidated
`runs/summary.csv` is also generated with accuracy statistics across all
benchmarks and models.
