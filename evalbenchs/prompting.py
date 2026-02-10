from __future__ import annotations

import re
from typing import Any

from evalbenchs.config import BenchmarkConfig

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Keys to try for question / choices / answer when not overridden by benchmark config
QUESTION_KEYS = ("question", "prompt", "query", "instruction", "stem", "text", "input", "context")
CHOICES_KEYS = ("choices", "options", "options_list", "alternatives")
ANSWER_KEYS = ("answer", "label", "correct", "gold", "output", "outputs", "target", "answer_index", "correct_answer", "Solution")

# MERA-style: example has "inputs": {"question", "option_a", "option_b", ...} — use these, ignore "instruction" template
INPUTS_OPTION_KEYS = ["option_a", "option_b", "option_c", "option_d", "option_e", "option_f", "option_g", "option_h"]


def _get_question(example: dict[str, Any], benchmark: BenchmarkConfig | None) -> str:
    # Prefer actual question from inputs (no template substitution; avoid "Докажи..." etc.)
    inputs = example.get("inputs")
    if isinstance(inputs, dict):
        q = inputs.get("question")
        if q is not None and str(q).strip():
            return str(q).strip()
    if benchmark and benchmark.question_key and benchmark.question_key in example:
        val = example[benchmark.question_key]
        if isinstance(val, str) and val.strip():
            return val.strip()
    # TransportBench etc.: "Problem" field
    if example.get("Problem") and isinstance(example["Problem"], str) and example["Problem"].strip():
        return example["Problem"].strip()
    for key in (QUESTION_KEYS if not benchmark or not benchmark.question_key else (benchmark.question_key,)):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            # Don't use instruction if it's a template with placeholders (e.g. {question})
            if key == "instruction" and ("{question}" in val or "{domain}" in val):
                continue
            return val.strip()
    return str(example)


def _get_choices(example: dict[str, Any], benchmark: BenchmarkConfig | None) -> list[str]:
    # Prefer options from inputs.option_a, option_b, ... (order preserved, skip null/empty)
    inputs = example.get("inputs")
    if isinstance(inputs, dict):
        opts = []
        for i, key in enumerate(INPUTS_OPTION_KEYS):
            v = inputs.get(key)
            if v is None or not str(v).strip():
                break
            label = CHOICE_LABELS[i] if i < len(CHOICE_LABELS) else str(i + 1)
            opts.append(f"{label}. {str(v).strip()}")
        if opts:
            return opts
    if benchmark and benchmark.choices_key and benchmark.choices_key in example:
        raw = example[benchmark.choices_key]
        if isinstance(raw, (list, tuple)):
            return [_format_choice(i, c) for i, c in enumerate(raw)]
        if isinstance(raw, dict):
            return [f"{k}. {v}" for k, v in sorted(raw.items())]
    for key in (CHOICES_KEYS if not benchmark or not benchmark.choices_key else (benchmark.choices_key,)):
        raw = example.get(key)
        if isinstance(raw, (list, tuple)):
            return [_format_choice(i, c) for i, c in enumerate(raw)]
        if isinstance(raw, dict):
            return [f"{k}. {v}" for k, v in sorted(raw.items())]
    option_keys = [k for k in example if isinstance(k, str) and re.fullmatch(r"[A-J]", k)]
    if option_keys:
        return [f"{k}. {example[k]}" for k in sorted(option_keys)]
    return []


def _format_choice(idx: int, choice: Any) -> str:
    label = CHOICE_LABELS[idx] if idx < len(CHOICE_LABELS) else str(idx + 1)
    if isinstance(choice, dict):
        text = choice.get("text") or choice.get("content") or choice.get("label") or str(choice)
    else:
        text = str(choice)
    return f"{label}. {text}"


class PromptBuilder:
    def __init__(
        self,
        base_system_prompt: str,
        override_system_prompt: str | None = None,
        benchmark: BenchmarkConfig | None = None,
    ):
        self.system_prompt = override_system_prompt or base_system_prompt
        self.benchmark = benchmark

    def build_messages(self, example: dict[str, Any]) -> list[dict[str, str]]:
        question = _get_question(example, self.benchmark)
        choices = _get_choices(example, self.benchmark)
        prompt = question
        if choices:
            prompt += "\n\nOptions:\n" + "\n".join(choices)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]


def extract_gold_answer(example: dict[str, Any], benchmark: BenchmarkConfig | None = None) -> str | None:
    keys = list(ANSWER_KEYS)
    if benchmark and benchmark.answer_key:
        keys = [benchmark.answer_key] + [k for k in keys if k != benchmark.answer_key]
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            idx = int(value)
            if 0 <= idx < len(CHOICE_LABELS):
                return CHOICE_LABELS[idx]
            return str(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                continue
            # "Solution": "False. The optimal..." -> extract "False"
            if key == "Solution" or s.lower().startswith("true") or s.lower().startswith("false"):
                low = s.lower()
                if low.startswith("true"):
                    return "True"
                if low.startswith("false"):
                    return "False"
            s_upper = s.upper()
            if len(s_upper) == 1 and s_upper in CHOICE_LABELS:
                return s_upper
            return s  # "Да", "Нет", "A, B", etc.
        if isinstance(value, (list, tuple)) and value:
            # e.g. outputs = ["A", "B"] -> "A, B"
            return ", ".join(str(v).strip() for v in value if v is not None and str(v).strip())
    return None


def extract_choice_from_response(response: str) -> str | None:
    if not response or not isinstance(response, str):
        return None
    match = re.search(r"\b([A-J])\b", response.upper())
    if match:
        return match.group(1)
    return None
