from __future__ import annotations

from ..exceptions import ProviderError
from .base import BaseProvider


class AnyProvider(BaseProvider):
    name = "anyprovider"
    needs_key = False
    model_map = {
        "anyprovider/openai": "openai",
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

        client = g4f.client.Client(provider=P.AnyProvider)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
        )
