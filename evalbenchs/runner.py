from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalbenchs.config import BenchmarkConfig, Config
from evalbenchs.data import load_benchmark, sample_dataset, LoadedBenchmark
from evalbenchs.gigachat import GigaChatClient
from evalbenchs.openrouter import OpenRouterClient
from evalbenchs.prompting import PromptBuilder, extract_choice_from_response, extract_gold_answer


@dataclass
class RunResult:
    benchmark: BenchmarkConfig
    model: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable form (e.g. numpy, Pydantic)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if not isinstance(obj, (str, int, float, bool, list, dict)):
        return str(obj)
    return obj


def _safe_text(response_json: dict[str, Any]) -> str:
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _score_response(
    example: dict[str, Any], response_text: str, benchmark: BenchmarkConfig | None = None
) -> bool:
    gold = extract_gold_answer(example, benchmark)
    if not gold:
        return False
    gold_str = str(gold).strip().lower()
    predicted = extract_choice_from_response(response_text)
    if predicted:
        predicted = predicted.strip().lower()
    else:
        resp = response_text.strip()
        # Да/Нет, True/False, Yes/No: compare first word or stated answer later in text
        if gold_str in ("да", "нет", "true", "false", "yes", "no"):
            first = resp.split(None, 1)[0].lower() if resp else ""
            first = first.rstrip(".,;:!?)\"\'")
            if first in ("да", "нет", "true", "false", "yes", "no"):
                return first == gold_str
            # Model may state answer in the middle/end: "ответом будет **да**", "ответ: да", "**да**"
            if gold_str in ("да", "нет"):
                m = re.search(r"(?:ответом будет|ответ:)\s*\**(да|нет)\b", resp, re.I)
                if m and m.group(1).lower() == gold_str:
                    return True
                m = re.search(r"\*\*(да|нет)\*\*", resp, re.I)
                if m and m.group(1).lower() == gold_str:
                    return True
        predicted = resp.lower()
    if "," in gold_str:
        # Multiple correct: predicted must be one of them
        return predicted in [s.strip() for s in gold_str.split(",")]
    return predicted == gold_str


async def _run_example(
    client: OpenRouterClient | GigaChatClient,
    model: str,
    prompt_builder: PromptBuilder,
    example: dict[str, Any],
    semaphore: asyncio.Semaphore,
    benchmark: BenchmarkConfig,
) -> dict[str, Any]:
    async with semaphore:
        messages = prompt_builder.build_messages(example)
        response_json = await client.chat(model, messages)
        response_text = _safe_text(response_json)
        return {
            "messages": messages,
            "response": response_text,
            "raw": response_json,
            "correct": _score_response(example, response_text, benchmark),
            "gold": extract_gold_answer(example, benchmark),
            "example": example,
        }


async def run_benchmark(
    config: Config,
    benchmark: BenchmarkConfig,
    model: str,
    output_dir: Path,
    sample_size: int,
    seed: int,
    max_concurrency: int,
    loaded: LoadedBenchmark | None = None,
    gigachat_max_concurrency: int | None = None,
    gigachat_serial_semaphore: asyncio.Semaphore | None = None,
) -> RunResult:
    # GigaChat: only one benchmark run at a time to avoid 429
    if model.startswith("gigachat/") and gigachat_serial_semaphore is not None:
        async with gigachat_serial_semaphore:
            return await _run_benchmark_impl(
                config, benchmark, model, output_dir, sample_size, seed,
                max_concurrency, loaded, gigachat_max_concurrency,
            )
    return await _run_benchmark_impl(
        config, benchmark, model, output_dir, sample_size, seed,
        max_concurrency, loaded, gigachat_max_concurrency,
    )


async def _run_benchmark_impl(
    config: Config,
    benchmark: BenchmarkConfig,
    model: str,
    output_dir: Path,
    sample_size: int,
    seed: int,
    max_concurrency: int,
    loaded: LoadedBenchmark | None = None,
    gigachat_max_concurrency: int | None = None,
) -> RunResult:
    if loaded is None:
        print(f"Loading benchmark [{benchmark.id}] {benchmark.name}...")
        loaded = load_benchmark(benchmark)
    dataset = sample_dataset(loaded.dataset, sample_size, seed)
    total = len(dataset)
    print(f"Running [{benchmark.id}] {benchmark.name} — {model} ({total} samples).")
    prompt_builder = PromptBuilder(
        config.base_system_prompt, benchmark.system_prompt, benchmark=benchmark
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{benchmark.id}_{model.replace('/', '_')}.jsonl"
    if model.startswith("gigachat/") and gigachat_max_concurrency is not None:
        concurrency = gigachat_max_concurrency
    else:
        concurrency = max_concurrency
    semaphore = asyncio.Semaphore(concurrency)
    if model.startswith("gigachat/"):
        model_name = model.split("/", 1)[1]
        client = GigaChatClient(model=model_name)
    else:
        client = OpenRouterClient()
        model_name = model

    try:
        tasks = [
            _run_example(client, model_name, prompt_builder, ex, semaphore, benchmark)
            for ex in dataset
        ]
        results: list[dict[str, Any]] = []
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{benchmark.name}] {model}: {done}/{total} done.")
    finally:
        if isinstance(client, GigaChatClient):
            await client.close()

    correct = sum(1 for item in results if item["correct"])
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(_json_serializable(item), ensure_ascii=False) + "\n")
    print(f"  [{benchmark.name}] {model}: {correct}/{total} correct. Results -> {output_path}")
    return RunResult(benchmark=benchmark, model=model, total=len(results), correct=correct)


async def run_all(
    config: Config,
    benchmarks: list[BenchmarkConfig],
    models: list[str],
    output_dir: Path,
    sample_size: int,
    seed: int,
    max_concurrency: int,
    gigachat_max_concurrency: int | None = None,
) -> list[RunResult]:
    # Load each benchmark once (not per model)
    print("Loading benchmarks (once each)...")
    loaded_by_id: dict[int, LoadedBenchmark] = {}
    for b in benchmarks:
        loaded_by_id[b.id] = load_benchmark(b)
        print(f"  [{b.id}] {b.name} loaded.")
    # When GigaChat is used and not overridden, limit concurrency to avoid 429
    giga_concurrency = gigachat_max_concurrency
    if giga_concurrency is None and any(m.startswith("gigachat/") for m in models):
        giga_concurrency = int(os.environ.get("GIGACHAT_MAX_CONCURRENCY", "1"))
    # Only one GigaChat benchmark at a time (sequential by benchmark for GigaChat)
    gigachat_serial = asyncio.Semaphore(1) if any(m.startswith("gigachat/") for m in models) else None
    tasks = [
        run_benchmark(
            config, b, m, output_dir, sample_size, seed, max_concurrency,
            loaded=loaded_by_id[b.id],
            gigachat_max_concurrency=giga_concurrency,
            gigachat_serial_semaphore=gigachat_serial,
        )
        for b in benchmarks
        for m in models
    ]
    return await asyncio.gather(*tasks)
