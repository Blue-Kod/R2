from __future__ import annotations

import json
import http.client
from abc import ABC, abstractmethod
from typing import Any, Iterator
from urllib.parse import urlparse

from ..types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChoiceChunk,
    Delta,
    Message,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from ..exceptions import ProviderError, RateLimitError


class BaseProvider(ABC):
    name: str
    needs_key: bool = False
    key_name: str | None = None
    supports_tools: bool = False
    model_map: dict[str, str] = {}

    @property
    def supported_models(self) -> list[str]:
        return list(self.model_map.keys())

    def get_actual_model(self, model: str) -> str:
        return self.model_map.get(model, model)

    @abstractmethod
    def _call(
        self, messages: list[dict], model: str, max_tokens: int, stream: bool, **kwargs
    ) -> Any:
        pass

    def chat_completions_create(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 100,
        stream: bool = False,
        **kwargs,
    ):
        actual_model = self.get_actual_model(model)
        return self._call(messages, actual_model, max_tokens, stream, **kwargs)


def http_stream_request(
    url: str, payload: dict, headers: dict, timeout: int = 30
) -> Iterator[dict]:
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=timeout
    )
    body = json.dumps(payload).encode("utf-8")
    conn.request("POST", parsed.path, body=body, headers=headers)
    resp = conn.getresponse()

    if resp.status == 429:
        conn.close()
        raise RateLimitError("http", "Rate limited (429)")
    if resp.status != 200:
        err = resp.read().decode("utf-8", errors="replace")[:200]
        conn.close()
        raise ProviderError("http", f"HTTP {resp.status}: {err}")

    try:
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("data: "):
                continue
            data = text[6:]
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
    finally:
        conn.close()


def http_request(
    url: str, payload: dict, headers: dict, timeout: int = 30
) -> dict:
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=timeout
    )
    body = json.dumps(payload).encode("utf-8")
    conn.request("POST", parsed.path, body=body, headers=headers)
    resp = conn.getresponse()

    if resp.status == 429:
        conn.close()
        raise RateLimitError("http", "Rate limited (429)")
    if resp.status != 200:
        err = resp.read().decode("utf-8", errors="replace")[:200]
        conn.close()
        raise ProviderError("http", f"HTTP {resp.status}: {err}")

    try:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)
    finally:
        conn.close()


def parse_completion(data: dict, fallback_model: str) -> ChatCompletion:
    choices = []
    for c in data.get("choices", []):
        msg = c.get("message", {})
        rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=ToolCallFunction(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", ""),
                ),
            ))
        choices.append(
            Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content") or "",
                    reasoning_content=rc,
                    tool_calls=tool_calls,
                ),
                finish_reason=c.get("finish_reason"),
            )
        )
    u = data.get("usage") or {}
    return ChatCompletion(
        id=data.get("id", ""),
        created=data.get("created", 0),
        model=data.get("model", fallback_model),
        choices=choices,
        usage=Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        ),
    )


def parse_chunk(obj: dict, fallback_model: str) -> ChatCompletionChunk | None:
    raw_choices = obj.get("choices", [])
    if not raw_choices:
        return None
    choices = []
    for c in raw_choices:
        d = c.get("delta", {})
        rc = d.get("reasoning_content") or d.get("reasoning") or ""
        choices.append(
            ChoiceChunk(
                index=c.get("index", 0),
                delta=Delta(
                    role=d.get("role"),
                    content=d.get("content") or "",
                    reasoning_content=rc,
                ),
                finish_reason=c.get("finish_reason"),
            )
        )
    return ChatCompletionChunk(
        id=obj.get("id", ""),
        created=obj.get("created", 0),
        model=obj.get("model", fallback_model),
        choices=choices,
    )
