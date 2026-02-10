"""
Analyze benchmark runs: classify incorrect answers as parse error vs model error.
Parse error = model response actually indicated the correct answer but we extracted wrong.
Model error = model gave a wrong answer (or we cannot detect that it was parsing).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evalbenchs.prompting import CHOICE_LABELS


def _all_letters_in_order(text: str) -> list[str]:
    """Return list of A-J letters found in text in order (case-insensitive)."""
    return re.findall(r"\b([A-J])\b", (text or "").upper())


def _gold_index(gold: str) -> int | None:
    """0-based index for gold letter (A=0, B=1, ...)."""
    if not gold or len(gold) != 1:
        return None
    g = gold.upper()
    if g in CHOICE_LABELS:
        return CHOICE_LABELS.index(g)
    return None


def _get_gold_choice_text(example: dict, gold: str) -> str | None:
    """Return the choice text for the gold answer (e.g. full text of option B)."""
    idx = _gold_index(gold)
    if idx is None:
        return None
    choices = example.get("choices") or example.get("options") or example.get("options_list")
    if isinstance(choices, (list, tuple)) and 0 <= idx < len(choices):
        c = choices[idx]
        if isinstance(c, dict):
            return (c.get("text") or c.get("content") or c.get("label") or str(c))[:200]
        return str(c)[:200]
    option_keys = [k for k in example if isinstance(k, str) and re.fullmatch(r"[A-J]", k)]
    if option_keys:
        sorted_keys = sorted(option_keys)
        if idx < len(sorted_keys):
            return str(example.get(sorted_keys[idx], ""))[:200]
    return None


def classify_incorrect(response: str, gold: str, example: dict) -> str:
    """
    Classify an incorrect item: 'parse_error' or 'model_error'.
    Parse error = we could have extracted the right answer (e.g. model said it differently).
    """
    if not response or not gold:
        return "model_error"
    resp_strip = response.strip()
    resp_lower = resp_strip.lower()
    gold_strip = str(gold).strip()
    gold_lower = gold_strip.lower()

    # 0) True/False: we only extract A-J, so "True." / "False." are never extracted
    if gold_lower in ("true", "false"):
        if resp_lower.startswith("true") or resp_lower.startswith("false"):
            first_word = resp_lower.split(None, 1)[0] if resp_lower else ""
            if first_word in ("true", "false") and first_word == gold_lower:
                return "parse_error"
    # Yes/No (Да/Нет) — same idea
    if gold_lower in ("yes", "no", "да", "нет"):
        for w in ("yes", "no", "да", "нет"):
            if resp_lower.startswith(w) and gold_lower == w:
                return "parse_error"

    gold_upper = gold_strip.upper()
    if len(gold_upper) != 1 or gold_upper not in CHOICE_LABELS:
        # Already handled True/False, Yes/No; other golds (e.g. numeric) -> model_error
        return "model_error"

    # 1) Multiple letters: we take first A-J; if gold appears later, it's a parse error
    letters = _all_letters_in_order(response)
    if len(letters) > 1 and gold_upper in letters:
        first = letters[0]
        if first != gold_upper:
            return "parse_error"  # model said correct letter but not first

    # 2) Model answered with option number (e.g. "option 2", "2.", "2)") for gold B (index 1)
    idx = _gold_index(gold)
    if idx is not None:
        num = idx + 1
        # "option 2", "option 2:", "answer: 2", "2.", "2)"
        if re.search(rf"(?:option|answer|choice)\s*[:\s]*{num}\b", resp_lower):
            return "parse_error"
        if re.search(rf"\b{num}\s*[.)]\s", response):
            return "parse_error"
        if re.search(rf"^\s*{num}\s*[.)]", resp_strip, re.MULTILINE):
            return "parse_error"

    # 3) Response contains the exact choice text of the gold option (model said right content, no letter)
    gold_text = _get_gold_choice_text(example, gold)
    if gold_text and gold_text.strip():
        needle = gold_text.strip()[:50].lower()
        if len(needle) >= 10 and needle in response.lower():
            return "parse_error"

    return "model_error"


def load_results(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify incorrect GPT-4o (or other model) answers: parse error vs model error."
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        nargs="?",
        default=Path("runs"),
        help="Directory with result JSONL files (e.g. runs/)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model substring to match in filenames (e.g. gpt-4o or openai_gpt-4o)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample parse_error / model_error to print (0 = none)",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        print(f"Not a directory: {runs_dir}")
        return

    # Collect all *model*.jsonl files
    pattern = f"*{args.model}*.jsonl"
    files = sorted(runs_dir.glob(pattern))
    if not files:
        print(f"No files matching {pattern} in {runs_dir}")
        return

    total_incorrect = 0
    parse_errors = 0
    model_errors = 0
    parse_samples: list[dict] = []
    model_samples: list[dict] = []

    for path in files:
        items = load_results(path)
        for item in items:
            if item.get("correct") is True:
                continue
            total_incorrect += 1
            response = item.get("response") or ""
            gold = item.get("gold")
            example = item.get("example") or {}
            kind = classify_incorrect(response, gold, example)
            if kind == "parse_error":
                parse_errors += 1
                if args.show_samples and len(parse_samples) < args.show_samples:
                    parse_samples.append(
                        {"file": path.name, "gold": gold, "response_preview": response[:300], "example_id": example.get("id")}
                    )
            else:
                model_errors += 1
                if args.show_samples and len(model_samples) < args.show_samples:
                    model_samples.append(
                        {"file": path.name, "gold": gold, "response_preview": response[:300], "example_id": example.get("id")}
                    )

    print("=" * 60)
    print("Error analysis (incorrect answers only)")
    print(f"  Files: {[p.name for p in files]}")
    print(f"  Total incorrect: {total_incorrect}")
    if total_incorrect:
        print(f"  Parse errors (answer was in response, we extracted wrong): {parse_errors} ({100 * parse_errors / total_incorrect:.1f}%)")
        print(f"  Model errors (model gave wrong answer): {model_errors} ({100 * model_errors / total_incorrect:.1f}%)")
    print("=" * 60)

    if args.show_samples and parse_samples:
        print("\nSample PARSE ERRORS (model likely right, we parsed wrong):")
        for i, s in enumerate(parse_samples, 1):
            print(f"  [{i}] {s['file']} gold={s['gold']} id={s.get('example_id')}")
            print(f"      response: {s['response_preview']!r}")

    if args.show_samples and model_samples:
        print("\nSample MODEL ERRORS (model gave wrong answer):")
        for i, s in enumerate(model_samples, 1):
            print(f"  [{i}] {s['file']} gold={s['gold']} id={s.get('example_id')}")
            print(f"      response: {s['response_preview']!r}")


if __name__ == "__main__":
    main()
