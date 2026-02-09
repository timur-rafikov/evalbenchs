from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    id: int
    name: str
    domain: str
    source: str
    split: str
    hf_repo: str | None = None
    hf_config: str | None = None
    repo: str | None = None
    system_prompt: str | None = None
    notes: str | None = None
    # Optional schema overrides for dataset columns (single key or first match wins)
    question_key: str | None = None
    choices_key: str | None = None
    answer_key: str | None = None


@dataclass
class Config:
    base_system_prompt: str
    benchmarks: list[BenchmarkConfig]


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    base_system_prompt = data["base_system_prompt"]
    benchmarks = [
        BenchmarkConfig(
            id=item["id"],
            name=item["name"],
            domain=item["domain"],
            source=item["source"],
            split=item.get("split", "test"),
            hf_repo=item.get("hf_repo"),
            hf_config=item.get("hf_config"),
            repo=item.get("repo"),
            system_prompt=item.get("system_prompt"),
            notes=item.get("notes"),
            question_key=item.get("question_key"),
            choices_key=item.get("choices_key"),
            answer_key=item.get("answer_key"),
        )
        for item in data["benchmarks"]
    ]
    return Config(base_system_prompt=base_system_prompt, benchmarks=benchmarks)


def get_benchmark_by_ids(config: Config, ids: list[int] | None) -> list[BenchmarkConfig]:
    if not ids:
        return config.benchmarks
    lookup = {bench.id: bench for bench in config.benchmarks}
    missing = [bench_id for bench_id in ids if bench_id not in lookup]
    if missing:
        missing_str = ", ".join(map(str, missing))
        raise ValueError(f"Unknown benchmark id(s): {missing_str}")
    return [lookup[bench_id] for bench_id in ids]


def as_dict(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "id": config.id,
        "name": config.name,
        "domain": config.domain,
        "source": config.source,
        "split": config.split,
        "hf_repo": config.hf_repo,
        "hf_config": config.hf_config,
        "repo": config.repo,
        "system_prompt": config.system_prompt,
        "notes": config.notes,
        "question_key": config.question_key,
        "choices_key": config.choices_key,
        "answer_key": config.answer_key,
    }
