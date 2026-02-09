from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalbenchs.config import BenchmarkConfig, Config
from evalbenchs.data import load_benchmark, sample_dataset
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
    predicted = extract_choice_from_response(response_text) or response_text.strip()
    return predicted.lower().strip() == str(gold).lower().strip()


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
) -> RunResult:
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
    semaphore = asyncio.Semaphore(max_concurrency)
    if model.startswith("gigachat/"):
        model_name = model.split("/", 1)[1]
        client = GigaChatClient(model=model_name)
    else:
        client = OpenRouterClient()
        model_name = model

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

    correct = sum(1 for item in results if item["correct"])
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
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
) -> list[RunResult]:
    tasks = [
        run_benchmark(config, b, m, output_dir, sample_size, seed, max_concurrency)
        for b in benchmarks
        for m in models
    ]
    return await asyncio.gather(*tasks)
