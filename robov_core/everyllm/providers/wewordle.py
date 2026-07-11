from __future__ import annotations

from ..types import ChatCompletion, ChatCompletionChunk
from ..exceptions import ProviderError
from .base import BaseProvider


class WeWordleProvider(BaseProvider):
    name = "wewordle"
    needs_key = False
    model_map = {
        "wewordle/v3": "v3",
        "wewordle/gpt-4o": "gpt-4o",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        try:
            import g4f.client
            import g4f.Provider as P
        except ImportError:
            raise ProviderError(self.name, "g4f not installed")

        client = g4f.client.Client(provider=P.WeWordle)
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=stream,
        )
