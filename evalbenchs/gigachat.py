from __future__ import annotations

import os
from typing import Any

from gigachat import GigaChat

try:
    from gigachat.models import Chat, Messages, MessagesRole
    _HAS_CHAT_MODELS = True
except ImportError:
    _HAS_CHAT_MODELS = False


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"Инструкция: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts) if parts else ""


def _messages_to_chat(messages: list[dict[str, str]]) -> Any:
    if not _HAS_CHAT_MODELS:
        return _messages_to_prompt(messages)
    role_map = {
        "system": MessagesRole.SYSTEM,
        "user": MessagesRole.USER,
        "assistant": MessagesRole.ASSISTANT,
    }
    gigachat_messages = []
    for msg in messages:
        role = role_map.get((msg.get("role") or "").lower(), MessagesRole.USER)
        content = (msg.get("content") or "").strip()
        if content:
            gigachat_messages.append(Messages(role=role, content=content))
    if not gigachat_messages:
        gigachat_messages = [Messages(role=MessagesRole.USER, content="")]
    return Chat(messages=gigachat_messages)


class GigaChatClient:
    """Client for GigaChat using the official gigachat SDK (credentials, scope, model)."""

    def __init__(
        self,
        model: str,
        credentials: str | None = None,
        scope: str | None = None,
        verify_ssl_certs: bool | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        creds = credentials or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_API_KEY")
        if not creds:
            raise RuntimeError(
                "GigaChat credentials required. Set GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY."
            )
        verify_env = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "").strip().lower()
        if verify_ssl_certs is not None:
            verify = verify_ssl_certs
        elif verify_env in ("1", "true", "yes"):
            verify = True
        else:
            # Default False: GigaChat often uses certs that fail standard verification
            verify = False
        self._client_kwargs = {
            "credentials": creds,
            "scope": scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
            "model": self._model_id(model),
            "verify_ssl_certs": verify,
            "temperature": 0,
            "profanity_check": False,
        }
        if base_url:
            self._client_kwargs["base_url"] = base_url

    @staticmethod
    def _model_id(name: str) -> str:
        # Map CLI model name to GigaChat model id (e.g. gigachat-2-max -> GigaChat-2-Max)
        if not name:
            return "GigaChat"
        lower = name.lower().strip()
        if "gigachat-2-max" in lower or "gigachat_2_max" in lower:
            return "GigaChat-2-Max"
        if "gigachat-2" in lower or "gigachat_2" in lower:
            return "GigaChat-2"
        if "gigachat" in lower:
            return "GigaChat"
        return name

    async def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        """Send chat request. `model` is ignored; instance model is used."""
        payload = _messages_to_chat(messages)
        async with GigaChat(**self._client_kwargs) as client:
            response = await client.achat(payload)
        content = ""
        if response.choices:
            msg = response.choices[0].message
            if hasattr(msg, "content") and msg.content:
                content = (msg.content or "").strip()
        return {
            "choices": [{"message": {"content": content}, "finish_reason": getattr(response.choices[0], "finish_reason", None)}],
            "usage": getattr(response, "usage", None),
        }
