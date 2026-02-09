from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI


class OpenRouterClient:
    """Client for OpenRouter using OpenAI-compatible API (AsyncOpenAI with base_url)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = (msg.content or "").strip() if getattr(msg, "content", None) else ""
        return {
            "choices": [{"message": {"content": content}, "finish_reason": response.choices[0].finish_reason}],
            "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
        }
