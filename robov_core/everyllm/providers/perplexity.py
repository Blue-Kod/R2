from __future__ import annotations

from ..types import ChatCompletion, ChatCompletionChunk
from ..exceptions import ProviderError
from .base import BaseProvider


class PerplexityProvider(BaseProvider):
    name = "perplexity"
    needs_key = False
    model_map = {
        "perplexity/auto": "auto",
        "perplexity/turbo": "turbo",
        "perplexity/gpt41": "gpt41",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        try:
            import g4f.client
            import g4f.Provider as P
        except ImportError:
            raise ProviderError(
                self.name,
                "g4f package not installed. Run: pip install g4f",
            )

        client = g4f.client.Client(provider=P.Perplexity)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
        )
