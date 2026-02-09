from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import datasets
import requests
from datasets import Dataset

from evalbenchs.config import BenchmarkConfig

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv"}


@dataclass
class LoadedBenchmark:
    benchmark: BenchmarkConfig
    dataset: Dataset


def _download_github_repo(repo_url: str, destination: Path) -> Path:
    match = re.search(r"github.com/([^/]+/[^/]+)", repo_url)
    if not match:
        raise ValueError(f"Unsupported GitHub URL: {repo_url}")
    repo_path = match.group(1).replace(".git", "")
    zip_url = f"https://codeload.github.com/{repo_path}/zip/refs/heads/main"
    response = requests.get(zip_url, timeout=30)
    if response.status_code != 200:
        zip_url = f"https://codeload.github.com/{repo_path}/zip/refs/heads/master"
        response = requests.get(zip_url, timeout=30)
    response.raise_for_status()
    zip_path = destination / "repo.zip"
    zip_path.write_bytes(response.content)
    shutil.unpack_archive(str(zip_path), destination)
    extracted_dirs = [path for path in destination.iterdir() if path.is_dir()]
    if not extracted_dirs:
        raise RuntimeError("GitHub download did not unpack to a directory")
    return extracted_dirs[0]


def _find_data_files(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            candidates.append(path)
    return sorted(candidates)


def _load_dataset_from_files(files: Iterable[Path]) -> Dataset:
    file_list = list(files)
    if not file_list:
        raise RuntimeError("No dataset files found in GitHub repository")
    primary = file_list[0]
    if primary.suffix == ".jsonl":
        return datasets.load_dataset("json", data_files=str(primary), split="train")
    if primary.suffix == ".json":
        return datasets.load_dataset("json", data_files=str(primary), split="train")
    if primary.suffix in {".csv", ".tsv"}:
        delimiter = "\t" if primary.suffix == ".tsv" else ","
        return datasets.load_dataset(
            "csv", data_files=str(primary), split="train", delimiter=delimiter
        )
    raise RuntimeError(f"Unsupported dataset file extension: {primary.suffix}")


def load_benchmark(benchmark: BenchmarkConfig) -> LoadedBenchmark:
    if benchmark.source == "huggingface":
        dataset = datasets.load_dataset(
            benchmark.hf_repo,
            benchmark.hf_config,
            split=benchmark.split,
        )
        return LoadedBenchmark(benchmark=benchmark, dataset=dataset)
    if benchmark.source == "github":
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = _download_github_repo(benchmark.repo, Path(tmpdir))
            files = _find_data_files(repo_root)
            dataset = _load_dataset_from_files(files)
            return LoadedBenchmark(benchmark=benchmark, dataset=dataset)
    raise ValueError(f"Unknown source {benchmark.source}")


def sample_dataset(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def serialize_example(example: dict[str, Any]) -> str:
    return json.dumps(example, ensure_ascii=False)
