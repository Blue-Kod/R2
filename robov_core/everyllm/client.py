from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import ChatCompletion, ChatCompletionChunk, Choice, ChoiceChunk, Delta, Message, Usage, ToolCall
from .exceptions import EveryLLMError, RateLimitError
from .keys import load_keys
from .auto import TTFTTracker, AutoRouter, get_capabilities, VISION_TEST_IMAGE, VISION_PROMPT
from .tools import builtin_tools, execute_tool
from .providers import (
    PollinationsProvider,
    AnyProvider,
    G4fSpaceProvider,
    AgnesProvider,
    GlmProvider,
    OpenCodeProvider,
    YqcloudProvider,
    PerplexityProvider,
    FeloProvider,
    WeWordleProvider,
    CohereProvider,
    AnyApiProvider,
)

REFINE_MAX_TOKENS = 200
BEST_MODELS_FILE = "BEST_MODELS.json"


def _refine_prompt() -> list[dict]:
    n = random.randint(1000000000, 9999999999)
    return [{"role": "user", "content": f"Say hello in 25 words or fewer. {n}"}]


def _classify_error(e: Exception) -> str:
    """Return a short human-readable error classification."""
    name = type(e).__name__
    msg = str(e).strip()
    low = msg.lower()
    if "rate" in low and ("limit" in low or "429" in low):
        return "rate limit"
    if "401" in low or "403" in low or "unauthorized" in low or "forbidden" in low:
        return "auth error"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "connect" in low or "refused" in low or "unreachable" in low:
        return "connection refused"
    if "dns" in low or "resolve" in low or "name or service" in low:
        return "DNS error"
    if "ssl" in low or "certificate" in low:
        return "SSL error"
    if "not found" in low or "404" in low:
        return "not found"
    if "key" in low and ("invalid" in low or "expired" in low):
        return "invalid API key"
    if name == "AuthenticationError" or "auth" in low:
        return "auth error"
    if msg:
        short = msg[:80]
        return f"{name}: {short}"
    return name


class _Completions:
    def __init__(self, client: EveryLLM):
        self._client = client

    def create(self, model: str, messages: list[dict], **kwargs) -> Any:
        auto_execute = kwargs.pop("auto_execute_tools", False)
        max_rounds = kwargs.pop("max_tool_rounds", 20)
        if auto_execute:
            return self._client._tool_loop(model, messages, max_rounds, **kwargs)
        return self._client._create(model, messages, **kwargs)


class _Chat:
    def __init__(self, client: EveryLLM):
        self.completions = _Completions(client)


class EveryLLM:
    def __init__(
        self,
        keys_path: Optional[str] = None,
        api_keys: Optional[dict[str, str]] = None,
        best_models_path: Optional[str | Path] = None,
    ):
        file_keys = load_keys(keys_path)
        self._keys = {**file_keys, **(api_keys or {})}

        self._model_registry: dict[str, object] = {}
        self._all_providers: list = []
        self._tracker = TTFTTracker()
        self._init_providers()

        self._auto_router = AutoRouter(self._model_registry, self._tracker)

        self._best_models_path = self._resolve_best_path(best_models_path)
        self._load_best_models()

        self._refresh_thread: Optional[threading.Thread] = None
        self._refresh_stop = threading.Event()
        self._refreshing_model: Optional[str] = None

        self.chat = _Chat(self)

    def _init_providers(self):
        for provider_cls in (PollinationsProvider, AnyProvider, G4fSpaceProvider,
                             YqcloudProvider, PerplexityProvider, FeloProvider,
                             WeWordleProvider, CohereProvider):
            p = provider_cls()
            self._all_providers.append(p)
            for model in p.supported_models:
                self._model_registry[model] = p
                caps = get_capabilities(model)
                self._tracker.set_capabilities(
                    model, caps["vision"], caps["thinking"],
                    caps.get("configurable_thinking", False),
                )

        key_providers = [
            (AgnesProvider, "agnes"),
            (GlmProvider, "glm"),
            (OpenCodeProvider, "opencode"),
            (AnyApiProvider, "anyapi"),
        ]
        for provider_cls, key_name in key_providers:
            key = self._keys.get(key_name)
            if not key:
                continue
            p = provider_cls(api_key=key)
            self._all_providers.append(p)
            for model in p.supported_models:
                self._model_registry[model] = p
                caps = get_capabilities(model)
                self._tracker.set_capabilities(
                    model, caps["vision"], caps["thinking"],
                    caps.get("configurable_thinking", False),
                )

    def _create(self, model: str, messages: list[dict], **kwargs) -> Any:
        max_tokens = kwargs.pop("max_tokens", 100)
        stream = kwargs.pop("stream", False)
        auto_model = model

        if self._is_refreshing(model):
            self._refresh_stop.set()
            if self._refresh_thread and self._refresh_thread.is_alive():
                self._refresh_thread.join(timeout=2)

        if model == "auto":
            return self._auto_create(messages, max_tokens, stream, auto_model, **kwargs)

        provider = self._model_registry.get(model)
        if provider is None:
            available = ", ".join(sorted(self._model_registry.keys()))
            raise EveryLLMError(
                f"Unknown model '{model}'. Available: {available}"
            )
        return self._call_with_ttft(provider, model, messages, max_tokens, stream, auto_model, **kwargs)

    def _tool_loop(self, model: str, messages: list[dict], max_rounds: int = 20, **kwargs) -> ChatCompletion:
        tools = kwargs.pop("tools", None) or builtin_tools()
        max_tokens = kwargs.get("max_tokens", 4096)
        kwargs["max_tokens"] = max_tokens
        kwargs["stream"] = False

        history = list(messages)

        for round_num in range(max_rounds):
            result = self._create(model, list(history), tools=tools, **kwargs)
            if not result.choices:
                return result

            choice = result.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                return result

            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                args = tc.function.arguments_dict()
                output = execute_tool(tc.function.name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        return result

    def _auto_create(self, messages, max_tokens, stream, auto_model, **kwargs):
        if not self._tracker.has_data():
            self.refresh()
        model_name, provider = self._auto_router.select()
        return self._call_with_ttft(
            provider, model_name, messages, max_tokens, stream, auto_model, **kwargs
        )

    def _call_with_ttft(
        self, provider, model, messages, max_tokens, stream, auto_model="", **kwargs
    ):
        if stream:
            return self._stream_ttft(provider, model, messages, max_tokens, auto_model, **kwargs)
        return self._nostream_ttft(provider, model, messages, max_tokens, auto_model, **kwargs)

    def _nostream_ttft(self, provider, model, messages, max_tokens, auto_model="", **kwargs):
        start = time.time()
        result = provider.chat_completions_create(
            messages, model, max_tokens, stream=False, **kwargs
        )
        ttft = (time.time() - start) * 1000
        self._tracker.update(model, ttft)
        return self._wrap_result(result, model)

    def _wrap_result(self, result, model: str) -> ChatCompletion:
        if isinstance(result, ChatCompletion):
            result.model_used = model
            return result
        choices = []
        for c in getattr(result, "choices", []):
            msg = getattr(c, "message", c)
            choices.append(Choice(
                index=getattr(c, "index", 0),
                message=Message(
                    role=getattr(msg, "role", "assistant"),
                    content=getattr(msg, "content", "") or "",
                ),
                finish_reason=getattr(c, "finish_reason", None),
            ))
        u = getattr(result, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
            total_tokens=getattr(u, "total_tokens", 0) if u else 0,
        )
        return ChatCompletion(
            id=getattr(result, "id", ""),
            created=getattr(result, "created", 0),
            model=getattr(result, "model", model),
            model_used=model,
            choices=choices,
            usage=usage,
        )

    def _stream_ttft(self, provider, model, messages, max_tokens, auto_model="", **kwargs):
        if not self._tracker.supports_streaming(model):
            return self._fake_stream(provider, model, messages, max_tokens, auto_model, **kwargs)

        start = time.time()
        first = True
        for chunk in provider.chat_completions_create(
            messages, model, max_tokens, stream=True, **kwargs
        ):
            if first:
                ttft = (time.time() - start) * 1000
                self._tracker.update(model, ttft)
                first = False
            if self._is_empty_chunk(chunk):
                continue
            yield self._wrap_chunk(chunk, model)

    def _wrap_chunk(self, chunk, model: str) -> ChatCompletionChunk:
        if isinstance(chunk, ChatCompletionChunk):
            chunk.model_used = model
            return chunk
        choices = []
        for c in getattr(chunk, "choices", []):
            d = getattr(c, "delta", c)
            choices.append(ChoiceChunk(
                index=getattr(c, "index", 0),
                delta=Delta(
                    role=getattr(d, "role", None),
                    content=getattr(d, "content", None) or "",
                    reasoning_content=getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or "",
                ),
                finish_reason=getattr(c, "finish_reason", None),
            ))
        return ChatCompletionChunk(
            id=getattr(chunk, "id", ""),
            created=getattr(chunk, "created", 0),
            model=getattr(chunk, "model", model),
            model_used=model,
            choices=choices,
        )

    @staticmethod
    def _is_empty_chunk(chunk) -> bool:
        if not chunk.choices:
            return True
        return all(
            not c.delta.content and not getattr(c.delta, "reasoning_content", None)
            for c in chunk.choices
        )

    def _fake_stream(self, provider, model, messages, max_tokens, auto_model="", **kwargs):
        start = time.time()
        result = provider.chat_completions_create(
            messages, model, max_tokens, stream=False, **kwargs
        )
        ttft = (time.time() - start) * 1000
        self._tracker.update(model, ttft)

        content = ""
        if hasattr(result, "choices") and result.choices:
            msg = result.choices[0].message
            content = msg.content if hasattr(msg, "content") else str(msg)

        mu = model
        yield ChatCompletionChunk(
            id=result.id if hasattr(result, "id") else "",
            model=result.model if hasattr(result, "model") else model,
            model_used=mu,
            choices=[ChoiceChunk(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=None,
            )],
        )
        yield ChatCompletionChunk(
            id=result.id if hasattr(result, "id") else "",
            model=result.model if hasattr(result, "model") else model,
            model_used=mu,
            choices=[ChoiceChunk(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )],
        )

    def models(self) -> list[str]:
        return sorted(self._model_registry.keys())

    def ttft_scores(self) -> dict[str, float]:
        return self._tracker.get_scores()

    def best_models(self) -> list[dict]:
        data = self._read_best_file()
        return data.get("models", [])

    def refresh(
        self,
        model: str = "all",
        timeout: float = 5.0,
        asynchronously: bool = False,
        delay: float = 0.3,
    ) -> dict[str, dict]:
        targets = self._resolve_refresh_models(model)

        if asynchronously:
            return self._refresh_parallel(targets, timeout, delay)
        return self._refresh_sequential(targets, timeout, delay)

    def _resolve_refresh_models(self, model: str) -> dict[str, object]:
        if model == "all":
            return dict(self._model_registry)
        if model in self._model_registry:
            return {model: self._model_registry[model]}
        available = ", ".join(sorted(self._model_registry.keys()))
        raise EveryLLMError(f"Unknown model '{model}'. Available: {available}")

    def _refresh_sequential(self, targets, timeout, delay) -> dict[str, dict]:
        results = {}

        providers = {}
        for model_name, provider in targets.items():
            pid = id(provider)
            if pid not in providers:
                providers[pid] = {"provider": provider, "models": []}
            providers[pid]["models"].append(model_name)

        def _test_provider_group(pg):
            provider = pg["provider"]
            group_results = {}
            for model_name in pg["models"]:
                ttft, err = self._measure_ttft(provider, model_name, timeout)
                if ttft is not None:
                    self._tracker.update(model_name, ttft)
                    group_results[model_name] = {"ttft_ms": round(ttft), "ok": True}
                else:
                    group_results[model_name] = {
                        "ttft_ms": None, "ok": False, "error": err or "failed",
                    }
            return group_results

        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            futures = {
                pool.submit(_test_provider_group, pg): pg
                for pg in providers.values()
            }
            for future in as_completed(futures):
                try:
                    group_results = future.result()
                    results.update(group_results)
                except Exception:
                    pg = futures[future]
                    for model_name in pg["models"]:
                        results[model_name] = {
                            "ttft_ms": None, "ok": False, "error": "Provider failed",
                        }

        self._save_best_models()
        return results

    def _refresh_parallel(self, targets, timeout, delay) -> dict[str, dict]:
        results = {}
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = {
                pool.submit(self._measure_ttft, provider, name, timeout): name
                for name, provider in targets.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    ttft, err = future.result()
                except Exception:
                    ttft, err = None, "provider crashed"
                if ttft is not None:
                    self._tracker.update(name, ttft)
                    results[name] = {"ttft_ms": round(ttft), "ok": True}
                else:
                    results[name] = {
                        "ttft_ms": None,
                        "ok": False,
                        "error": err or "failed",
                    }

        self._save_best_models()
        return results

    def silent_refresh(
        self,
        model: str = "all",
        timeout: float = 5.0,
        asynchronously: bool = False,
        delay: float = 0.3,
    ) -> None:
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._refresh_stop.clear()
        self._refresh_thread = threading.Thread(
            target=self._background_refresh,
            args=(model, timeout, asynchronously, delay),
            daemon=True,
        )
        self._refresh_thread.start()

    def _background_refresh(self, model, timeout, asynchronously, delay):
        targets = self._resolve_refresh_models(model)

        if asynchronously:
            self._refresh_parallel(targets, timeout, delay)
        else:
            self._refresh_sequential(targets, timeout, delay)

    def deep_refresh(
        self,
        model: str = "all",
        timeout: float = 8.0,
        delay: float = 0.3,
    ) -> dict[str, dict]:
        results = self.refresh(model=model, timeout=timeout, delay=delay)

        targets = self._resolve_refresh_models(model)
        for name, provider in targets.items():
            caps = self._tracker.get_capabilities(name)
            if caps.get("vision"):
                results[name]["vision"] = self._test_vision(provider, name, timeout)
            else:
                results[name]["vision"] = False

        self._save_best_models()

        return results

    def _is_refreshing(self, model: str) -> bool:
        if not (self._refresh_thread and self._refresh_thread.is_alive()):
            return False
        if model == "auto":
            return True
        return self._refreshing_model == model

    def _test_vision(self, provider, model_name: str, timeout: float = 8.0) -> bool:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": VISION_TEST_IMAGE}},
                    ],
                }
            ]
            result = [None]
            exception = [None]

            def _target():
                try:
                    start = time.time()
                    r = provider.chat_completions_create(
                        messages, model_name, 50, stream=False,
                    )
                    result[0] = r
                except Exception as e:
                    exception[0] = e

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive() or exception[0] or not result[0]:
                return False

            r = result[0]
            content = ""
            if hasattr(r, "choices") and r.choices:
                msg = r.choices[0].message
                content = (msg.content or "").upper()
            elif hasattr(r, "content"):
                content = (r.content or "").upper()

            return "red" in content.lower()
        except Exception:
            return False

    def _measure_ttft(self, provider, model_name: str, timeout: float = 10.0):
        """Return (ttft_ms, error_msg). error_msg is None on success."""
        for attempt in range(3):
            result = [None]
            exception = [None]
            is_rate_limit = [False]
            last_error = [""]

            def _target():
                try:
                    prompt = _refine_prompt()
                    try:
                        start = time.time()
                        for chunk in provider.chat_completions_create(
                            prompt, model_name, REFINE_MAX_TOKENS, stream=True
                        ):
                            content = ""
                            reasoning = ""
                            if hasattr(chunk, "choices") and chunk.choices:
                                d = getattr(chunk.choices[0], "delta", chunk.choices[0])
                                content = getattr(d, "content", None) or ""
                                reasoning = (
                                    getattr(d, "reasoning_content", None)
                                    or getattr(d, "reasoning", None)
                                    or ""
                                )
                            if content.strip() or reasoning.strip():
                                result[0] = (time.time() - start) * 1000
                                self._tracker.set_streaming(model_name, True)
                                return
                    except RateLimitError:
                        is_rate_limit[0] = True
                        last_error[0] = "rate limit"
                    except Exception as e:
                        last_error[0] = _classify_error(e)

                    self._tracker.set_streaming(model_name, False)

                    try:
                        start = time.time()
                        r = provider.chat_completions_create(
                            _refine_prompt(), model_name, REFINE_MAX_TOKENS, stream=False
                        )
                        content = ""
                        reasoning = ""
                        if hasattr(r, "choices") and r.choices:
                            msg = getattr(r.choices[0], "message", r.choices[0])
                            content = getattr(msg, "content", None) or ""
                            reasoning = (
                                getattr(msg, "reasoning_content", None)
                                or getattr(msg, "reasoning", None)
                                or ""
                            )
                        if content.strip() or reasoning.strip():
                            result[0] = (time.time() - start) * 1000
                    except RateLimitError:
                        is_rate_limit[0] = True
                        last_error[0] = "rate limit"
                    except Exception as e:
                        last_error[0] = _classify_error(e)
                except Exception as e:
                    exception[0] = e

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                self._tracker.set_streaming(model_name, False)
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None, "timeout (no response)"
            if exception[0]:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None, _classify_error(exception[0])
            if is_rate_limit[0]:
                if attempt < 2:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                return None, "rate limit (429)"
            if result[0] is not None:
                return result[0], None
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            err = last_error[0] or "empty response"
            return None, err
        return None, "failed after 3 attempts"

    def _resolve_best_path(self, path: Optional[str | Path]) -> Path:
        if path is not None:
            return Path(path)
        return Path.cwd() / BEST_MODELS_FILE

    def _read_best_file(self) -> dict:
        p = self._best_models_path
        if not p.exists():
            return {"models": []}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"models": []}

    def _load_best_models(self):
        data = self._read_best_file()

        if not data.get("models"):
            self.refresh()
            return

        scores = self._tracker.get_scores()

        for entry in data.get("models", []):
            name = entry.get("name")
            ttft = entry.get("ttft_ms")
            streaming = entry.get("streaming")
            vision = entry.get("vision")
            thinking = entry.get("thinking")
            configurable_thinking = entry.get("configurable_thinking", False)
            if name and ttft is not None and name not in scores:
                self._tracker.update(name, float(ttft))
            if name and streaming is not None:
                self._tracker.set_streaming(name, streaming)
            if name and vision is not None and thinking is not None:
                self._tracker.set_capabilities(
                    name, vision, thinking, configurable_thinking,
                )

    def _save_best_models(self):
        scores = self._tracker.get_scores()

        def _entry(name, ttft):
            caps = self._tracker.get_capabilities(name)
            return {
                "name": name,
                "ttft_ms": ttft,
                "streaming": self._tracker.supports_streaming(name),
                "vision": caps.get("vision", False),
                "thinking": caps.get("thinking", False),
                "configurable_thinking": caps.get("configurable_thinking", False),
            }

        entries = [_entry(n, t) for n, t in sorted(scores.items(), key=lambda x: x[1])]
        for name in sorted(set(self._model_registry) - set(scores)):
            entries.append(_entry(name, None))

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": entries,
        }

        try:
            with open(self._best_models_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
