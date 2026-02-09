from __future__ import annotations

import re
from typing import Any

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


class PromptBuilder:
    def __init__(self, base_system_prompt: str, override_system_prompt: str | None = None):
        self.system_prompt = override_system_prompt or base_system_prompt

    def build_messages(self, example: dict[str, Any]) -> list[dict[str, str]]:
        question = self._extract_question(example)
        choices = self._extract_choices(example)
        prompt = question
        if choices:
            prompt += "\n\nOptions:\n" + "\n".join(choices)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _extract_question(self, example: dict[str, Any]) -> str:
        for key in ("question", "prompt", "query", "instruction", "stem"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(example)

    def _extract_choices(self, example: dict[str, Any]) -> list[str]:
        if "choices" in example and isinstance(example["choices"], (list, tuple)):
            return [self._format_choice(idx, choice) for idx, choice in enumerate(example["choices"])]
        if "options" in example and isinstance(example["options"], (list, tuple)):
            return [self._format_choice(idx, choice) for idx, choice in enumerate(example["options"])]
        option_keys = [key for key in example if re.fullmatch(r"[A-H]", str(key))]
        if option_keys:
            return [f"{key}. {example[key]}" for key in sorted(option_keys)]
        return []

    def _format_choice(self, idx: int, choice: Any) -> str:
        label = CHOICE_LABELS[idx] if idx < len(CHOICE_LABELS) else str(idx + 1)
        return f"{label}. {choice}"


def extract_gold_answer(example: dict[str, Any]) -> str | None:
    for key in ("answer", "label", "correct", "gold", "output"):
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip()
    return None


def extract_choice_from_response(response: str) -> str | None:
    match = re.search(r"\b([A-H])\b", response.upper())
    if match:
        return match.group(1)
    return None
