from __future__ import annotations

from ..types import ChatCompletion, ChatCompletionChunk
from ..exceptions import ProviderError
from .base import BaseProvider


class CohereProvider(BaseProvider):
    name = "cohere"
    needs_key = False
    model_map = {
        "cohere/command-r-plus": "command-r-plus",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        try:
            import g4f.client
            import g4f.Provider as P
        except ImportError:
            raise ProviderError(self.name, "g4f not installed")

        client = g4f.client.Client(provider=P.CohereForAI_C4AI_Command)
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=stream,
        )
