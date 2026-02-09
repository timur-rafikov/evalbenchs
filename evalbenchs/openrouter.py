from __future__ import annotations

import json
import os
from typing import Any

import aiohttp


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    async def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=120) as response:
                    if response.status >= 400:
                        detail = await response.text()
                        raise RuntimeError(
                            f"OpenRouter error {response.status}: {detail.strip() or 'unknown error'}"
                        )
                    return await response.json()
            except aiohttp.ClientError as exc:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError("OpenRouter returned invalid JSON") from exc
