from __future__ import annotations

import argparse
import asyncio
import csv
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
        default=["openai/gpt-4o", "gigachat/gigachat-2-max"],
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
    output_dir = Path(args.output_dir)
    results = asyncio.run(
        run_all(
            config,
            benchmarks,
            args.models,
            output_dir,
            args.sample_size,
            args.seed,
            args.max_concurrency,
        )
    )
    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_id",
                "benchmark_name",
                "model",
                "correct",
                "total",
                "accuracy",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "benchmark_id": result.benchmark.id,
                    "benchmark_name": result.benchmark.name,
                    "model": result.model,
                    "correct": result.correct,
                    "total": result.total,
                    "accuracy": f"{result.accuracy:.4f}",
                }
            )

    print("\nSummary:")
    for result in results:
        print(
            f"[{result.benchmark.id}] {result.benchmark.name} - {result.model}: "
            f"{result.correct}/{result.total} ({result.accuracy:.2%})"
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
