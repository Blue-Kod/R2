from __future__ import annotations

from .base import (
    BaseProvider,
    http_request,
    http_stream_request,
    parse_chunk,
    parse_completion,
)
from ..types import ChatCompletion, ChatCompletionChunk


class G4fSpaceProvider(BaseProvider):
    name = "g4f.space"
    needs_key = False
    _url = "https://g4f.space/v1/chat/completions"
    model_map = {
        "g4f.space/auto": "auto",
    }

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        if stream:
            return self._stream(messages, model, max_tokens)
        return self._non_stream(messages, model, max_tokens)

    def _non_stream(self, messages, model, max_tokens):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = http_request(self._url, payload, self._headers())
        return parse_completion(data, model)

    def _stream(self, messages, model, max_tokens):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        for obj in http_stream_request(self._url, payload, self._headers()):
            chunk = parse_chunk(obj, model)
            if chunk is not None:
                yield chunk
