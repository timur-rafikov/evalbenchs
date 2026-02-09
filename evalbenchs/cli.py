from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evalbenchs.config import get_benchmark_by_ids, load_config
from evalbenchs.runner import run_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLMs on domain benchmarks")
    parser.add_argument("--config", default="benchmarks.yaml")
    parser.add_argument("--bench", type=int, nargs="*", default=[])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-4o-mini", "gigachat/gigachat-2-max"],
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmarks = get_benchmark_by_ids(config, args.bench)
    results = asyncio.run(
        run_all(
            config,
            benchmarks,
            args.models,
            Path(args.output_dir),
            args.sample_size,
            args.seed,
            args.max_concurrency,
        )
    )
    print("\nSummary:")
    for result in results:
        print(
            f"[{result.benchmark.id}] {result.benchmark.name} - {result.model}: "
            f"{result.correct}/{result.total} ({result.accuracy:.2%})"
        )


if __name__ == "__main__":
    main()
