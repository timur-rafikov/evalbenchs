from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import datasets
import requests
from datasets import Dataset

from evalbenchs.config import BenchmarkConfig

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt"}

# Path segments that indicate non-dataset files (skip when searching in repos)
SKIP_PATH_SEGMENTS = {"license", "readme", "licenses", "assets", "fonts", "docs", ".github"}


@dataclass
class LoadedBenchmark:
    benchmark: BenchmarkConfig
    dataset: Dataset


def _download_github_repo(repo_url: str, destination: Path) -> Path:
    match = re.search(r"github.com/([^/]+/[^/]+)", repo_url)
    if not match:
        raise ValueError(f"Unsupported GitHub URL: {repo_url}")
    repo_path = match.group(1).replace(".git", "")
    default_branch = None
    repo_api = f"https://api.github.com/repos/{repo_path}"
    try:
        repo_response = requests.get(repo_api, timeout=30)
        repo_response.raise_for_status()
        default_branch = repo_response.json().get("default_branch")
    except requests.RequestException:
        default_branch = None
    zip_candidates = [branch for branch in [default_branch, "main", "master"] if branch]
    response = None
    zip_url = None
    for branch in zip_candidates:
        zip_url = f"https://codeload.github.com/{repo_path}/zip/refs/heads/{branch}"
        response = requests.get(zip_url, timeout=30)
        if response.status_code == 200:
            break
    if response is None:
        raise RuntimeError("Unable to download GitHub repository archive")
    response.raise_for_status()
    zip_path = destination / "repo.zip"
    zip_path.write_bytes(response.content)
    if not zipfile.is_zipfile(zip_path):
        detail = response.text.strip()
        raise RuntimeError(
            f"GitHub download did not return a zip archive from {zip_url}: "
            f"{detail or 'unknown error'}"
        )
    shutil.unpack_archive(str(zip_path), destination)
    extracted_dirs = [path for path in destination.iterdir() if path.is_dir()]
    if not extracted_dirs:
        raise RuntimeError("GitHub download did not unpack to a directory")
    return extracted_dirs[0]


def _is_likely_dataset_file(path: Path) -> bool:
    """Skip license/readme/assets and prefer structured data extensions."""
    parts_lower = [p.lower() for p in path.parts]
    if any(seg in SKIP_PATH_SEGMENTS for seg in parts_lower):
        return False
    if path.name.lower() in {"license", "readme", "license.txt", "readme.txt"}:
        return False
    suf = path.suffix.lower()
    if suf == ".txt":
        # Only accept .txt in data-like dirs (e.g. data/, benchmark/) to avoid LICENSE.txt
        return any(d in parts_lower for d in ("data", "benchmark", "dataset", "train", "test"))
    return True


def _collect_data_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            if not _is_likely_dataset_file(path):
                continue
            candidates.append(path)
    return candidates


def _find_data_files(
    repo_root: Path, split: str = "test", subset: str | None = None
) -> list[Path]:
    candidates = _collect_data_files(repo_root)
    if subset:
        subset_lower = subset.lower()
        candidates = [p for p in candidates if subset_lower in str(p).lower()]
    if not candidates and (repo_root / "data").is_dir():
        candidates = _collect_data_files(repo_root / "data")
    # TeleQnA etc.: dataset in a single .zip in repo root — extract and search inside
    if not candidates:
        zips = [p for p in repo_root.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
        if len(zips) == 1:
            zip_path = zips[0]
            extract_to = repo_root / "_zip_extract"
            zip_password = os.environ.get("REPO_ZIP_PASSWORD") or os.environ.get("GITHUB_REPO_ZIP_PASSWORD")
            try:
                extract_to.mkdir(exist_ok=True)
                extracted = False
                if zip_password:
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(extract_to, pwd=zip_password.encode("utf-8"))
                        extracted = True
                    except NotImplementedError:
                        pass
                    if not extracted:
                        # Python zipfile doesn't support this compression; try unzip CLI (e.g. Linux)
                        try:
                            r = subprocess.run(
                                ["unzip", "-P", zip_password, "-o", str(zip_path), "-d", str(extract_to)],
                                capture_output=True,
                                timeout=120,
                            )
                            extracted = r.returncode == 0
                        except (OSError, FileNotFoundError):
                            pass
                        if not extracted:
                            raise RuntimeError(
                                "Zip uses a compression method not supported by Python. "
                                "Install 'unzip' and ensure REPO_ZIP_PASSWORD is set, or extract manually: "
                                f'unzip -P "$REPO_ZIP_PASSWORD" "{zip_path}" -d <out_dir>'
                            ) from None
                else:
                    shutil.unpack_archive(str(zip_path), str(extract_to))
                    extracted = True
                if extracted:
                    subdirs = [p for p in extract_to.iterdir() if p.is_dir()]
                    search_root = subdirs[0] if len(subdirs) == 1 else extract_to
                    candidates = _collect_data_files(search_root)
            finally:
                if extract_to.exists():
                    shutil.rmtree(extract_to, ignore_errors=True)
    if not candidates:
        return []
    # Prefer file whose name contains the requested split (e.g. test_v1.jsonl for split "test")
    split_lower = split.lower()
    with_split = [p for p in candidates if split_lower in p.stem.lower()]
    chosen = sorted(with_split) if with_split else sorted(candidates)
    # Prefer .jsonl/.json over .txt/.csv so we don't pick LICENSE.txt or wrong format
    ext_order = (".jsonl", ".json", ".csv", ".tsv", ".txt")
    chosen.sort(key=lambda p: (ext_order.index(p.suffix.lower()) if p.suffix.lower() in ext_order else 99))
    return chosen


def _normalize_teleqna_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert 'option 1'/'option 2'/... and 'answer': 'option 2: ...' to question/choices/answer."""
    out: dict[str, Any] = dict(item)
    option_keys = [k for k in item if isinstance(k, str) and re.match(r"option\s*\d+", k, re.I)]
    option_keys.sort(key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else 0)
    if option_keys:
        choices = [item[k] for k in option_keys if isinstance(item.get(k), str)]
        if choices:
            out["choices"] = choices
        ans = item.get("answer")
        if isinstance(ans, str):
            m = re.search(r"option\s*(\d+)", ans, re.I)
            if m:
                idx = int(m.group(1))
                out["answer"] = "ABCDEFGHIJ"[idx - 1] if 1 <= idx <= 10 else str(idx)
    return out


def _load_jsonl_or_json_file(path: Path) -> Dataset:
    """Load .txt/.jsonl: try JSONL (one JSON per line), then single JSON array or object."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if rows:
        return Dataset.from_list(rows)
    # Fallback: whole file as JSON (array or object with list inside)
    with open(path, encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Could not parse {path} as JSONL (one JSON per line) or as single JSON. {e}"
            ) from e
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return Dataset.from_list(data)
    if isinstance(data, dict):
        for key in ("data", "questions", "examples", "items", "rows"):
            if key in data and isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                return Dataset.from_list(data[key])
        # TeleQnA-style: keys "question 0", "question 1", ... each value is an object
        if all(re.match(r"question\s*\d+", k, re.I) for k in data if isinstance(k, str)):
            items = list(data.values())
            if items and isinstance(items[0], dict):
                normalized = [_normalize_teleqna_item(d) for d in items]
                return Dataset.from_list(normalized)
        # Any dict whose values are all dicts -> use as list
        if data and all(isinstance(v, dict) for v in data.values()):
            items = list(data.values())
            if items and isinstance(items[0], dict) and "question" in items[0]:
                normalized = [_normalize_teleqna_item(d) for d in items]
                return Dataset.from_list(normalized)
    raise RuntimeError(f"No list of objects found in {path} (tried JSONL and JSON array/object)")


def _load_dataset_from_files(files: Iterable[Path], split: str = "test") -> Dataset:
    file_list = list(files)
    if not file_list:
        raise RuntimeError("No dataset files found in GitHub repository")
    primary = file_list[0]
    suf = primary.suffix.lower()
    if suf == ".jsonl":
        return _load_jsonl_or_json_file(primary)
    if suf == ".txt":
        return _load_jsonl_or_json_file(primary)
    if suf == ".json":
        return datasets.load_dataset("json", data_files=str(primary), split="train")
    if suf in {".csv", ".tsv"}:
        delimiter = "\t" if suf == ".tsv" else ","
        return datasets.load_dataset(
            "csv", data_files=str(primary), split="train", delimiter=delimiter
        )
    raise RuntimeError(f"Unsupported dataset file extension: {primary.suffix}")


def _hf_token() -> str | bool:
    """Token for Hugging Face Hub (gated datasets). From env or use cached from 'huggingface-cli login'."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True


def load_benchmark(benchmark: BenchmarkConfig) -> LoadedBenchmark:
    if benchmark.source == "huggingface":
        dataset = datasets.load_dataset(
            benchmark.hf_repo,
            benchmark.hf_config,
            split=benchmark.split,
            token=_hf_token(),
            verification_mode="no_checks",
        )
        if benchmark.hf_filter:
            filters = benchmark.hf_filter
            def keep(row):
                for col, value in filters.items():
                    if row.get(col) != value:
                        return False
                return True
            dataset = dataset.filter(keep)
        return LoadedBenchmark(benchmark=benchmark, dataset=dataset)
    if benchmark.source == "github":
        if benchmark.local_data_path:
            local_path = Path(benchmark.local_data_path).resolve()
            if not local_path.exists():
                raise FileNotFoundError(f"local_data_path not found: {local_path}")
            if local_path.is_file():
                files = [local_path]
            else:
                files = _find_data_files(
                    local_path, split=benchmark.split, subset=benchmark.github_data_subset
                )
            dataset = _load_dataset_from_files(files, split=benchmark.split)
            return LoadedBenchmark(benchmark=benchmark, dataset=dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = _download_github_repo(benchmark.repo, Path(tmpdir))
            files = _find_data_files(
                repo_root, split=benchmark.split, subset=benchmark.github_data_subset
            )
            dataset = _load_dataset_from_files(files, split=benchmark.split)
            return LoadedBenchmark(benchmark=benchmark, dataset=dataset)
    raise ValueError(f"Unknown source {benchmark.source}")


def sample_dataset(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def serialize_example(example: dict[str, Any]) -> str:
    return json.dumps(example, ensure_ascii=False)
