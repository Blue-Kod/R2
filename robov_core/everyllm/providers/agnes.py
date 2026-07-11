from __future__ import annotations

from .base import (
    BaseProvider,
    http_request,
    http_stream_request,
    parse_chunk,
    parse_completion,
)
from ..types import ChatCompletion, ChatCompletionChunk


class AgnesProvider(BaseProvider):
    name = "agnes"
    needs_key = True
    key_name = "agnes"
    supports_tools = True
    _url = "https://apihub.agnes-ai.com/v1/chat/completions"
    model_map = {
        "agnes-2.0-flash": "agnes-2.0-flash",
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        if stream:
            return self._stream(messages, model, max_tokens, **kwargs)
        return self._non_stream(messages, model, max_tokens, **kwargs)

    def _non_stream(self, messages, model, max_tokens, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        data = http_request(self._url, payload, self._headers())
        return parse_completion(data, model)

    def _stream(self, messages, model, max_tokens, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        for obj in http_stream_request(self._url, payload, self._headers()):
            chunk = parse_chunk(obj, model)
            if chunk is not None:
                yield chunk
