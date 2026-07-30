from __future__ import annotations

import json
import time
from typing import Any, Iterator

from .base import BaseProvider
from ..types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChoiceChunk,
    Delta,
    Message,
    Usage,
)
from ..exceptions import ProviderError

GRADIO_TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 3.0


def _get_gradio_client(space_id: str):
    try:
        from gradio_client import Client
    except ImportError:
        raise ProviderError(
            "hf_spaces",
            "gradio_client not installed. Run: pip install gradio_client",
        )
    return Client(space_id, verbose=False)


def _call_with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_err


def _fake_stream(result_text: str, model: str) -> Iterator[ChatCompletionChunk]:
    yield ChatCompletionChunk(
        id="",
        model=model,
        model_used=model,
        choices=[ChoiceChunk(
            index=0,
            delta=Delta(role="assistant", content=result_text),
            finish_reason=None,
        )],
    )
    yield ChatCompletionChunk(
        id="",
        model=model,
        model_used=model,
        choices=[ChoiceChunk(
            index=0,
            delta=Delta(),
            finish_reason="stop",
        )],
    )


def _to_completion(text: str, model: str) -> ChatCompletion:
    return ChatCompletion(
        id="",
        model=model,
        model_used=model,
        choices=[Choice(
            index=0,
            message=Message(role="assistant", content=text),
            finish_reason="stop",
        )],
        usage=Usage(),
    )


def _messages_to_user_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


# ── MiniMax-Text-01 ────────────────────────────────────────────────
class MiniMaxTextProvider(BaseProvider):
    name = "minimax"
    needs_key = False
    supports_tools = False
    _space = "MiniMaxAI/MiniMax-Text-01"
    model_map = {
        "minimax-text-01": "minimax-text-01",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        user_text = _messages_to_user_text(messages)
        client = _get_gradio_client(self._space)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        result = _call_with_retry(
            client.predict,
            message=user_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_name="/chat",
        )
        text = str(result) if result else ""
        if stream:
            return _fake_stream(text, model)
        return _to_completion(text, model)


# ── MiniMax-VL-01 (multimodal) ─────────────────────────────────────
class MiniMaxVLProvider(BaseProvider):
    name = "minimax_vl"
    needs_key = False
    supports_tools = False
    _space = "MiniMaxAI/MiniMax-VL-01"
    model_map = {
        "minimax-vl-01": "minimax-vl-01",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        user_text = _messages_to_user_text(messages)
        client = _get_gradio_client(self._space)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        result = _call_with_retry(
            client.predict,
            message={"text": user_text, "files": []},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_name="/chat",
        )
        text = str(result) if result else ""
        if stream:
            return _fake_stream(text, model)
        return _to_completion(text, model)


# ── Step-3.7-Flash ─────────────────────────────────────────────────
class StepFlashProvider(BaseProvider):
    name = "stepfun"
    needs_key = False
    supports_tools = False
    _space = "stepfun-ai/Step-3.7-Flash-dev"
    model_map = {
        "step-3.7-flash": "step-3.7-flash",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        client = _get_gradio_client(self._space)
        temperature = kwargs.get("temperature", 0.7)
        reasoning_effort = kwargs.get("reasoning_effort", "medium")
        messages_json = json.dumps(messages, ensure_ascii=False)
        raw = _call_with_retry(
            client.predict,
            messages_json=messages_json,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_name="/chat_with_step",
        )
        try:
            data = json.loads(raw)
            content = data.get("content", "")
            reasoning = data.get("reasoning_content", "")
        except (json.JSONDecodeError, TypeError):
            content = str(raw) if raw else ""
            reasoning = ""

        if not content.strip() and reasoning.strip():
            content = reasoning

        if stream:
            return _fake_stream(content, model)

        return ChatCompletion(
            id="",
            model=model,
            model_used=model,
            choices=[Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                ),
                finish_reason="stop",
            )],
            usage=Usage(),
        )


# ── Qwen3-Omni ─────────────────────────────────────────────────────
class QwenOmniProvider(BaseProvider):
    name = "qwen"
    needs_key = False
    supports_tools = False
    _space = "Qwen/Qwen3-Omni-Demo"
    model_map = {
        "qwen3-omni": "qwen3-omni",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        user_text = _messages_to_user_text(messages)
        client = _get_gradio_client(self._space)
        result = _call_with_retry(
            client.predict,
            user_text,
            None,
            None,
            None,
            api_name="/chat_predict",
        )
        text = ""
        if isinstance(result, (list, tuple)):
            for v in result:
                s = str(v)
                if "__type__" not in s and len(s) > 5:
                    text = s
                    break
        else:
            text = str(result) if result else ""

        if stream:
            return _fake_stream(text, model)
        return _to_completion(text, model)
