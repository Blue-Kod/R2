from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import http.client

from .base import BaseProvider
from ..exceptions import ProviderError, RateLimitError
from ..types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChoiceChunk,
    Delta,
    Message,
    Usage,
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
_STATUS_URL = "/duckchat/v1/status"
_CHAT_URL = "/duckchat/v1/chat"
_HOST = "duck.ai"

_JS_PATH = Path(__file__).parent / "duckai_vqd.js"

_DEFAULT_REASONING_EFFORT: dict[str, str | None] = {
    "gpt-5.4-nano": "low",
    "gpt-5.4-mini": "low",
    "claude-haiku-4-5": "low",
    "mistral-small-2603": None,
    "tinfoil/gpt-oss-120b": "low",
    "tinfoil/gemma4-31b": "low",
}


class _DuckRetry(Exception):
    pass


def _mime_from_data_url(url: str) -> str:
    if url.startswith("data:") and ";" in url:
        mime = url[5:].split(";", 1)[0]
        if mime:
            return mime
    return "image/png"


def _convert_part(part: dict) -> dict:
    ptype = part.get("type")
    if ptype == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        return {"type": "image", "mimeType": _mime_from_data_url(url), "image": url}
    return part


def _convert_messages(messages) -> list[dict]:
    converted = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            msg = {**msg, "content": [_convert_part(p) for p in content]}
        converted.append(msg)
    return converted


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class DuckAIProvider(BaseProvider):
    name = "duckai"
    needs_key = False
    supports_tools = False
    model_map = {
        "duckai/gpt-5.4-nano": "gpt-5.4-nano",
        "duckai/gpt-5.4-mini": "gpt-5.4-mini",
        "duckai/claude-haiku-4-5": "claude-haiku-4-5",
        "duckai/mistral-small-2603": "mistral-small-2603",
        "duckai/gpt-oss-120b": "tinfoil/gpt-oss-120b",
        "duckai/gemma4-31b": "tinfoil/gemma4-31b",
    }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        effort = kwargs.pop(
            "reasoning_effort",
            kwargs.pop("reasoningEffort", _DEFAULT_REASONING_EFFORT.get(model)),
        )
        if stream:
            return self._stream(messages, model, effort)
        return self._non_stream(messages, model, effort)

    def _payload(self, messages, model, effort) -> dict:
        payload = {"canUseTools": True, "messages": _convert_messages(messages), "model": model}
        if effort:
            payload["reasoningEffort"] = effort
        return payload

    def _headers(self, vqd: str) -> dict:
        return {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "pragma": "no-cache",
            "priority": "u=0",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "User-Agent": _USER_AGENT,
            "x-vqd-hash-1": vqd,
            "Referer": "https://duck.ai/",
        }

    def _get_vqd(self) -> str:
        conn = http.client.HTTPSConnection(_HOST, 443, timeout=15)
        try:
            conn.request("GET", _STATUS_URL, headers={
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9,fa;q=0.8",
                "cache-control": "no-store",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "x-vqd-accept": "1",
                "User-Agent": _USER_AGENT,
                "Referer": "https://duck.ai/",
            })
            resp = conn.getresponse()
            challenge = resp.getheader("x-Vqd-hash-1")
            resp.read()
            if resp.status != 200 or not challenge:
                raise ProviderError(
                    "duckai", f"status returned HTTP {resp.status} without x-Vqd-hash-1"
                )
        finally:
            conn.close()
        return _solve_vqd(challenge)

    def _iter_events(self, vqd: str, payload: dict):
        conn = http.client.HTTPSConnection(_HOST, 443, timeout=120)
        try:
            conn.request(
                "POST",
                _CHAT_URL,
                body=json.dumps(payload).encode("utf-8"),
                headers=self._headers(vqd),
            )
            resp = conn.getresponse()

            if resp.status == 429:
                raise RateLimitError("duckai", "Rate limited (429)")
            if resp.status == 418:
                err = resp.read().decode("utf-8", errors="replace")[:200]
                raise _DuckRetry(f"418 teapot: {err}")
            if resp.status != 200:
                err = resp.read().decode("utf-8", errors="replace")[:200]
                raise ProviderError("duckai", f"HTTP {resp.status}: {err}")

            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text.startswith("data: "):
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

    def _non_stream(self, messages, model, effort) -> ChatCompletion:
        last = None
        for attempt in range(3):
            try:
                vqd = self._get_vqd()
                events = self._iter_events(
                    vqd, self._payload(messages, model, effort)
                )
                return self._accumulate(events, model)
            except _DuckRetry as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise ProviderError("duckai", str(last) if last else "duck.ai retry exhausted")

    def _accumulate(self, events, model) -> ChatCompletion:
        content = ""
        reasoning = ""
        for ev in events:
            msg = ev.get("message")
            if isinstance(msg, str) and msg:
                content += msg
                continue
            if ev.get("state") == "text-delta":
                text = ev.get("text")
                if isinstance(text, str) and text:
                    reasoning += text
        created = int(time.time())
        cid = f"duckai-{random.randint(100000000, 999999999)}"
        return ChatCompletion(
            id=cid,
            created=created,
            model=model,
            choices=[Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning or None,
                ),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=_estimate_tokens(content) + _estimate_tokens(reasoning),
                total_tokens=_estimate_tokens(content) + _estimate_tokens(reasoning),
            ),
        )

    def _stream(self, messages, model, effort):
        last = None
        for attempt in range(3):
            try:
                vqd = self._get_vqd()
                events = self._iter_events(
                    vqd, self._payload(messages, model, effort)
                )
                for chunk in self._iter_chunks(events, model):
                    yield chunk
                return
            except _DuckRetry as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise ProviderError("duckai", str(last) if last else "duck.ai retry exhausted")

    def _iter_chunks(self, events, model):
        created = int(time.time())
        cid = f"duckai-{random.randint(100000000, 999999999)}"
        for ev in events:
            msg = ev.get("message")
            if isinstance(msg, str) and msg:
                yield ChatCompletionChunk(
                    id=cid,
                    created=created,
                    model=model,
                    choices=[ChoiceChunk(
                        index=0,
                        delta=Delta(role="assistant", content=msg),
                        finish_reason=None,
                    )],
                )
                continue
            if ev.get("state") == "text-delta":
                text = ev.get("text")
                if isinstance(text, str) and text:
                    yield ChatCompletionChunk(
                        id=cid,
                        created=created,
                        model=model,
                        choices=[ChoiceChunk(
                            index=0,
                            delta=Delta(role="assistant", reasoning_content=text),
                            finish_reason=None,
                        )],
                    )
        yield ChatCompletionChunk(
            id=cid,
            created=created,
            model=model,
            choices=[ChoiceChunk(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )],
        )


def _solve_vqd(challenge: str) -> str:
    _ensure_node_runtime()
    try:
        proc = subprocess.run(
            ["node", str(_JS_PATH)],
            input=challenge,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            cwd=str(_JS_PATH.parent),
            env=dict(os.environ),
        )
    except OSError as e:
        raise ProviderError("duckai", f"failed to run VQD solver: {e}")
    if proc.returncode != 0:
        raise ProviderError(
            "duckai", f"VQD solver failed: {proc.stderr.strip()[:300]}"
        )
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise ProviderError(
            "duckai", f"VQD solver bad output: {proc.stdout[:300]}"
        )
    if out.get("error"):
        raise ProviderError("duckai", f"VQD solver error: {out['error'][:300]}")
    return out["vqd"]


_NODE_READY: bool | None = None
_NODE_ERR = (
    "duck.ai provider needs Node.js (with npm) to solve its VQD challenge. "
    "Install it from https://nodejs.org and rerun. "
    "Or install the JS dependency manually: cd everyllm/providers && npm install"
)


def _check_node() -> bool:
    try:
        proc = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=15
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _jsdom_installed(cwd: str) -> bool:
    try:
        proc = subprocess.run(
            ["node", "-e", "require('jsdom')"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=cwd,
            env=dict(os.environ),
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_node_runtime() -> None:
    global _NODE_READY
    if _NODE_READY is True:
        return

    cwd = str(_JS_PATH.parent)

    if not _check_node():
        raise ProviderError("duckai", _NODE_ERR)

    if _jsdom_installed(cwd):
        _NODE_READY = True
        return

    npm = shutil.which("npm")
    if not npm:
        raise ProviderError("duckai", _NODE_ERR)
    try:
        proc = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=cwd,
            env=dict(os.environ),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ProviderError(
            "duckai",
            f"could not run `npm install` for jsdom: {e}. "
            f"Try manually: cd everyllm/providers && npm install",
        )
    if proc.returncode != 0:
        raise ProviderError(
            "duckai",
            "`npm install` failed: "
            + (proc.stderr or proc.stdout).strip()[:300]
            + ". Try manually: cd everyllm/providers && npm install",
        )

    if not _jsdom_installed(cwd):
        raise ProviderError(
            "duckai",
            "jsdom is still unavailable after `npm install` in everyllm/providers. "
            "Try manually: cd everyllm/providers && npm install",
        )
    _NODE_READY = True
