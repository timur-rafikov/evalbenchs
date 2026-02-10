from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evalbenchs.config import load_config
from evalbenchs.data import load_benchmark
from evalbenchs.prompting import extract_gold_answer

SYSTEM_PROMPT_KEYS = {"system", "system_prompt", "systemMessage"}


def check_labels(bench, dataset, sample_size: int = 20) -> tuple[int, int]:
    """Return (with_gold, total) for first sample_size examples."""
    n = min(sample_size, len(dataset))
    with_gold = 0
    for i in range(n):
        ex = dataset[i]
        if extract_gold_answer(ex, bench):
            with_gold += 1
    return with_gold, n


def detect_system_prompt(example: dict[str, Any]) -> str | None:
    for key in SYSTEM_PROMPT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect benchmark samples")
    parser.add_argument("--config", default="benchmarks.yaml")
    parser.add_argument("--bench", type=int, nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--check-labels", action="store_true", help="Check how many examples have gold labels (split: train/test)")
    args = parser.parse_args()

    config = load_config(args.config)
    benchmarks = [
        bench
        for bench in config.benchmarks
        if not args.bench or bench.id in args.bench
    ]

    if args.check_labels:
        print("Split / labels check (first 20 examples per benchmark):")
        for bench in benchmarks:
            try:
                loaded = load_benchmark(bench)
                with_gold, total = check_labels(bench, loaded.dataset, 20)
                status = "OK" if with_gold == total and total else ("NO LABELS" if with_gold == 0 else f"PARTIAL ({with_gold}/{total})")
                print(f"  [{bench.id}] {bench.name}  split={bench.split}  labels: {with_gold}/{total}  {status}")
            except Exception as e:
                print(f"  [{bench.id}] {bench.name}  split={bench.split}  error: {e}")
        return

    for bench in benchmarks:
        loaded = load_benchmark(bench)
        dataset = loaded.dataset
        print(f"\n[{bench.id}] {bench.name} ({bench.domain})")
        for idx, example in enumerate(dataset.select(range(min(args.samples, len(dataset))))):
            prompt = detect_system_prompt(example)
            if prompt:
                print(f"System prompt detected: {prompt}")
            print(f"Sample {idx + 1}: {json.dumps(example, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
