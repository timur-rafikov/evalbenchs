from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evalbenchs.config import load_config
from evalbenchs.data import load_benchmark

SYSTEM_PROMPT_KEYS = {"system", "system_prompt", "systemMessage"}


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
    args = parser.parse_args()

    config = load_config(args.config)
    benchmarks = [
        bench
        for bench in config.benchmarks
        if not args.bench or bench.id in args.bench
    ]

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
