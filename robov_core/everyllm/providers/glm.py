from __future__ import annotations

from .base import (
    BaseProvider,
    http_request,
    http_stream_request,
    parse_chunk,
    parse_completion,
)
from ..types import ChatCompletion, ChatCompletionChunk


class GlmProvider(BaseProvider):
    name = "glm"
    needs_key = True
    key_name = "glm"
    supports_tools = True
    _url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model_map = {
        "glm-4.7-flash": "GLM-4.7-Flash",
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
            return self._stream(messages, model, max_tokens)
        return self._non_stream(messages, model, max_tokens)

    def _glm_payload(self, model, messages, max_tokens, stream, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "thinking": {"type": "disabled"},
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        return payload

    def _non_stream(self, messages, model, max_tokens, **kwargs):
        payload = self._glm_payload(model, messages, max_tokens, stream=False, **kwargs)
        data = http_request(self._url, payload, self._headers())
        return parse_completion(data, model)

    def _stream(self, messages, model, max_tokens, **kwargs):
        payload = self._glm_payload(model, messages, max_tokens, stream=True, **kwargs)
        for obj in http_stream_request(self._url, payload, self._headers()):
            chunk = parse_chunk(obj, model)
            if chunk is not None:
                yield chunk
