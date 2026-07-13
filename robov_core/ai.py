#!/usr/bin/env python3
"""
R2 AI Agent — EveryHarness-based with streaming, cancellation, and robot API.

Usage:
    from robov_core.ai import agent
    agent.command("Привет!")  # interrupts current work, starts new prompt
    agent.stop()              # stops everything
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import importlib.util as _ilu

from robov_core.everyllm import EveryLLMError

_e_path = os.path.join(os.path.dirname(__file__), "everyllm")
_e_spec = _ilu.spec_from_file_location("everyllm", os.path.join(_e_path, "__init__.py"),
    submodule_search_locations=[_e_path])
_e_mod = _ilu.module_from_spec(_e_spec)
sys.modules["everyllm"] = _e_mod
_e_spec.loader.exec_module(_e_mod)
EveryLLM = _e_mod.EveryLLM
EveryLLMError = _e_mod.EveryLLMError

# ---------------------------------------------------------------------------
# Tag patterns (from EveryLLM harness.py)
# ---------------------------------------------------------------------------
PYTHON_TAG = re.compile(r"<python>(.*?)</python>", re.DOTALL)

TOOL_CALL_TAG = re.compile(
    r'<tool_call>([\w]+)((?:\s+[\w-]+=(?:"[^"]*"|' + r"'[^']*'" + r'|[^\s>"]*))*\s*)(?:</tool_call>|(?=\s*$))',
    re.DOTALL,
)
TOOL_CALL_BLOCK = re.compile(
    r"```tool_call\n(\w+)\s+((?:(?!```)[\s\S])*?)\n```",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------
SHELL_CMD = None


def _detect_shell() -> str:
    global SHELL_CMD
    if SHELL_CMD:
        return SHELL_CMD
    if sys.platform == "win32":
        for cmd in ["bash", "git", "sh"]:
            try:
                subprocess.run(
                    [cmd, "--version"], capture_output=True, timeout=5, check=True,
                )
                SHELL_CMD = cmd
                return cmd
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        SHELL_CMD = "powershell"
        return SHELL_CMD
    return "bash"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_BOOT_TIME = time.time()

# ---------------------------------------------------------------------------
# Robot API docs — available to AI via run_python
# ---------------------------------------------------------------------------
ROBOT_API_DOCS = """
== R2 ROBOT API (available via `from robov_core.high_level import *`) ==

You are inside a real robot. These functions are imported and available in run_python:
  angle(servo, angle_value)     — move servo (0-9). Returns bool.
    Servos: 0=neck, 1=R shoulder, 2=L shoulder, 3=tilt,
            4=R wrist rot, 5=L wrist rot, 6=R elbow, 7=L elbow,
            8=R gripper, 9=L gripper
    Angle ranges: neck/tilt/grippers 0-180, shoulders 0-270, elbows 0-270, wrists 0-270

  emote(name)                   — change face expression. Returns bool.
    Available: happy, neutral, scared, spooked, traumatized

  speak(text)                   — speak text aloud through speaker (Russian TTS).

  set_emote(name)               — same as emote()
  get_emote()                   — returns current emote name
  set_eyes_position(x, y)      — move eyes [-1..1, -1..1]
  get_eyes_position()           — returns (x, y)

  get_stereo_camera()           — camera object (None if no camera)
  get_raw_frame(left=True)      — get OpenCV frame from camera
  get_coords_stereo(frame, x, y)— get 3D coords at pixel (x,y) from stereo image

  health_snapshot()             — returns {"cpu": %, "ram": %, "temp": "xx.x°C"}
  ip_address()                  — returns robot's IP string
  log(message)                  — print to robot's log buffer
  cleanup()                     — shutdown robot gracefully

  shell_start()                 — start interactive shell
  shell_write(command)          — send command to shell, returns bool
  shell_output()                — get shell output string
  shell_onetime(command)        — send + get output

  get_servo_angles()            — dict {channel: angle}
  get_servo_offsets()           — dict {channel: offset}
  set_servo_physical(ch, angle) — set servo by physical angle (accounts for inversion)

== EXAMPLES ==
  speak("Привет, я Р2!")       # say something
  emote("happy")                # smile
  angle(0, 45)                  # turn neck 45 degrees
  angle(8, 180)                 # close right gripper
  angle(9, 180)                 # close left gripper
  angle(1, 90)                  # move right shoulder
  log("AI did something")       # log a message
  health = health_snapshot()    # get system status
  print(f"CPU: {health['cpu']}%")

IMPORTANT: Everything you write as final text WILL BE SPOKEN ALOUD. Keep speech short.
"""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(cwd: str) -> str:
    return f"""You are R2 — a physical robot assistant. You are physically present with the user. Everything you say OUTSIDE of <tool_call> tags and reasoning will be SPOKEN ALOUD through your speaker. The user will HEAR every word you output as text. Reasoning and tool calls are silent — only final text is spoken.

<env>
Working directory: {cwd}
Platform: {sys.platform}
Shell: {_detect_shell()}
Today: {_timestamp()}
</env>

<robot_api>
{ROBOT_API_DOCS}
</robot_api>

<tools>
Format: <tool_call>tool_name key="value" key2="value2"</tool_call>
Quote values with spaces. Single-word values need no quotes. Multiple tools per response is fine.

read_file path="src/main.py" offset=1 limit=100
write_file path="hello.txt" content="Hello World!"
edit_file path="src/main.py" old="def old_name(" new="def new_name("
grep pattern="TODO|FIXME" path="src/" include="*.py"
ls path="."
run_bash command="ls -la /tmp" timeout=15
run_python code="print(sum(range(100)))"
web_search query="latest robotics news" max_results=5
wiki_search query="inverse kinematics" sentences=3
http_get url="https://api.example.com/data" timeout=10
http_request url="https://api.example.com/data" method="POST" body='{{"key":"val"}}'
todo action="add" text="Fix the motor calibration" priority="high"
todo action="done" id=1
todo action="list" status="pending"
current_state
remember tier="core" key="user_name" value="Alex"
remember tier="archival" content="User works on robotics" category="work"
recall query="robotics" limit=5
core_status
forget tier="core" key="user_name"
delete target="archival" keyword="robotics"
delete target="core" key="user_name"
delete target="todo" id=1
delete target="file" path="old_file.txt"
</tools>

<robot_tools>
Use run_python with robot API functions to control the robot:
  run_python code="speak('Привет!')"  — speak text aloud
  run_python code="emote('happy')"    — change face
  run_python code="angle(0, 45)"      — move neck servo
  run_python code="angle(8, 180); angle(9, 180)"  — close grippers
  run_python code="from robov_core.high_level import health_snapshot; print(health_snapshot())"
</robot_tools>

<memory>
Two-tier:
1. CORE — always in context (max 20). remember/forget with tier=core
2. ARCHIVAL — searchable. remember with tier=archival, recall to search
</memory>

<instructions>
- You are a robot. Everything you write as final text WILL BE SPOKEN ALOUD. So keep speech short and natural.
- NEVER say you will search, look up, or check something. Just DO IT — emit the <tool_call> tags directly. No talking about tools, just use them.
- MANDATORY tool usage: if the user asks about ANY fact (people, places, events, weather, prices, news, dates, APIs, libraries, technologies, anything beyond simple math or logic) — you MUST emit tool_call tags. You have NO knowledge. You MUST search.
- Use run_python with robot API to control the robot (speak, move, emote). Do it immediately, don't ask permission.
- Combine multiple tools in ONE response:
  * recall + web_search (check memory then internet)
  * web_search + wiki_search (cross-reference)
  * ls + read_file (explore then read)
  You can emit 2, 3, even 5 tool_call tags in one response.
- REMEMBER USER FACTS: user shares a personal detail → immediately: remember tier="core" key="..." value="..."
- CONTEXT CLUES: "tell me about it", "my city", "and my country" → recall query="..." first, then search.
- Read files before editing.
- Be extremely concise. One short sentence. Your speech is heard aloud — no bullet lists, no markdown, no formatting.
- For simple greetings (привет, как дела, приветствую) and trivial questions — answer immediately in 1 short sentence, no reasoning needed. Do NOT search or use tools for simple social conversation.
- Speak Russian by default. Use the language the user writes in.
</instructions>"""


# ---------------------------------------------------------------------------
# Memory system — two tiers (from EveryLLM harness.py)
# ---------------------------------------------------------------------------
class Memory:
    def __init__(self, base_dir: str | Path | None = None):
        self._dir = Path(base_dir or Path.cwd()) / ".harness_memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._core_path = self._dir / "core.json"
        self._archival_path = self._dir / "archival.json"
        self._core: list[dict] = self._load(self._core_path)
        self._archival: list[dict] = self._load(self._archival_path)

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _save(path: Path, data: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_all(self) -> None:
        self._save(self._core_path, self._core)
        self._save(self._archival_path, self._archival)

    def add_core(self, key: str, value: str) -> str:
        for entry in self._core:
            if entry["key"].lower() == key.lower():
                entry["value"] = value
                entry["updated_at"] = _timestamp()
                self.save_all()
                return f"Updated core memory: {key} = {value}"
        if len(self._core) >= 20:
            return "ERROR: Core memory is full (max 20). Use 'forget' to remove something first."
        self._core.append({"key": key, "value": value, "created_at": _timestamp(), "updated_at": _timestamp()})
        self.save_all()
        return f"Added to core memory: {key} = {value}"

    def get_core(self) -> str:
        if not self._core:
            return "(empty)"
        return "\n".join(f"  {e['key']}: {e['value']}" for e in self._core)

    def forget_core(self, key: str) -> str:
        before = len(self._core)
        self._core = [e for e in self._core if e["key"].lower() != key.lower()]
        if len(self._core) == before:
            return f"Key '{key}' not found in core memory."
        self.save_all()
        return f"Removed '{key}' from core memory."

    def core_to_prompt(self) -> str:
        if not self._core:
            return ""
        return "\n".join(f"  {e['key']}: {e['value']}" for e in self._core)

    def add_archival(self, content: str, category: str = "general") -> str:
        entry = {
            "id": len(self._archival) + 1,
            "content": content,
            "category": category,
            "created_at": _timestamp(),
        }
        self._archival.append(entry)
        self.save_all()
        return f"Stored in archival memory (id={entry['id']}, category={category})"

    def recall(self, query: str, limit: int = 10) -> str:
        query_lower = query.lower()
        scored = []
        for e in self._archival:
            text = e["content"].lower()
            words = query_lower.split()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return f"No matches for '{query}' in archival memory ({len(self._archival)} entries)."
        return "\n".join(
            f"[id={e['id']} | {e['category']} | {e['created_at']}] {e['content']}"
            for _, e in scored[:limit]
        )

    def archival_status(self) -> str:
        if not self._archival:
            return "Archival memory is empty."
        cats: dict[str, int] = {}
        for e in self._archival:
            c = e.get("category", "general")
            cats[c] = cats.get(c, 0) + 1
        return f"Archival memory: {len(self._archival)} entries ({', '.join(f'{k}={v}' for k, v in sorted(cats.items()))})"

    def forget_archival(self, id_or_keyword: str) -> str:
        try:
            rid = int(id_or_keyword)
            before = len(self._archival)
            self._archival = [e for e in self._archival if e["id"] != rid]
            if len(self._archival) == before:
                return f"No entry with id={rid} found."
            self.save_all()
            return f"Removed archival entry id={rid}."
        except ValueError:
            kw = id_or_keyword.lower()
            before = len(self._archival)
            self._archival = [e for e in self._archival if kw not in e["content"].lower()]
            removed = before - len(self._archival)
            if removed == 0:
                return f"No entries containing '{id_or_keyword}' found."
            self.save_all()
            return f"Removed {removed} archival entries containing '{id_or_keyword}'."


# ---------------------------------------------------------------------------
# Tool executor (from EveryLLM harness.py, extended)
# ---------------------------------------------------------------------------
class ToolExecutor:
    def __init__(self, cwd: str, shell: str, memory: Memory):
        self.cwd = cwd
        self.shell = shell
        self.memory = memory
        self.env = os.environ.copy()
        self.python_env: dict[str, Any] = {"__builtins__": __builtins__, "cwd": cwd}
        self._todo_path = Path(cwd) / ".harness_memory" / "todo.json"
        self._todo: list[dict] = self._load_todo()

    def _load_todo(self) -> list[dict]:
        if not self._todo_path.exists():
            return []
        try:
            with open(self._todo_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_todo(self) -> None:
        self._todo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._todo_path, "w", encoding="utf-8") as f:
            json.dump(self._todo, f, indent=2, ensure_ascii=False)

    def execute(self, name: str, params: dict) -> str:
        HANDLERS = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "grep": self._grep,
            "ls": self._ls,
            "run_bash": self._run_bash,
            "run_python": self._run_python,
            "web_search": self._web_search,
            "wiki_search": self._wiki_search,
            "http_get": self._http_get,
            "http_request": self._http_request,
            "todo": self._todo_tool,
            "current_state": self._current_state,
            "remember": self._remember,
            "recall": self._recall,
            "core_status": self._core_status,
            "forget": self._forget,
            "delete": self._delete,
        }
        handler = HANDLERS.get(name)
        if not handler:
            available = ", ".join(sorted(HANDLERS.keys()))
            return f"ERROR: Unknown tool '{name}'. Available: {available}"
        try:
            return handler(params)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p

    def _read_file(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        if not path.exists():
            return f"ERROR: File not found: {p.get('path', '')}"
        if path.is_dir():
            return f"ERROR: Is a directory: {path}. Use ls instead."
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        offset = max(p.get("offset", 1) - 1, 0)
        limit = p.get("limit", 2000)
        selected = lines[offset:offset + limit]
        return "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=offset + 1))

    def _write_file(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        content = p.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"

    def _edit_file(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        old = p.get("old", "")
        new = p.get("new", "")
        if not path.exists():
            return f"ERROR: File not found: {p.get('path', '')}"
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return f"ERROR: String not found in {path.name}"
        count = text.count(old)
        new_text = text.replace(old, new, 1)
        path.write_text(new_text, encoding="utf-8")
        return f"Edited {path.name} ({count} occurrence{'s' if count > 1 else ''}, replaced 1)"

    def _grep(self, p: dict) -> str:
        pattern = p.get("pattern", "")
        search_path = self._resolve(p.get("path", "."))
        include = p.get("include")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"ERROR: Invalid regex: {e}"
        results = []
        scanned = 0
        for f in search_path.rglob("*"):
            if not f.is_file():
                continue
            if include and not f.match(include):
                continue
            if any(skip in f.parts for skip in (".venv", "node_modules", ".git", "__pycache__")):
                continue
            scanned += 1
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{f.name}:{i}: {line.strip()[:200]}")
            except Exception:
                continue
            if len(results) > 50:
                results.append("... (truncated)")
                break
        if not results:
            return f"No matches in {scanned} files"
        return "\n".join(results)

    def _ls(self, p: dict) -> str:
        path = self._resolve(p.get("path", "."))
        if not path.exists():
            return f"ERROR: Path not found: {p.get('path', '')}"
        if path.is_file():
            return str(path.name)
        show_all = p.get("all", False)
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            if not show_all and e.name.startswith("."):
                continue
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                lines.append(f"  {e.name} ({e.stat().st_size}b)")
        return "\n".join(lines) or "(empty)"

    def _run_bash(self, p: dict) -> str:
        cmd = p.get("command", "")
        if not cmd:
            return "ERROR: No command provided."
        timeout = min(p.get("timeout", 30), 120)
        shell = _detect_shell()
        if shell == "powershell":
            full_cmd = ["powershell", "-NoProfile", "-Command", cmd]
        else:
            full_cmd = [shell, "-c", cmd]
        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=timeout,
                cwd=self.cwd, env=self.env, encoding="utf-8", errors="replace",
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}" if output else f"Exit code: {result.returncode}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _run_python(self, p: dict) -> str:
        code = p.get("code", "")
        if not code:
            return "ERROR: No code provided."
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            exec(code, self.python_env)
            stdout = buf_out.getvalue()
            stderr = buf_err.getvalue()
        except Exception as e:
            stdout = buf_out.getvalue()
            stderr = buf_err.getvalue() + f"\n{type(e).__name__}: {e}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        output = stdout
        if stderr:
            output += f"\nSTDERR:\n{stderr}" if output else stderr
        result = output.strip() or "(no output)"
        if any(kw in result for kw in ("Error", "Exception", "Traceback")):
            return f"ERROR: {result}"
        return result

    def _web_search(self, p: dict) -> str:
        query = p.get("query", "")
        if not query:
            return "ERROR: No query provided."
        max_results = min(p.get("max_results", 5), 10)
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=max_results)
            if not results:
                return f"No results for '{query}'"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"ERROR: Search failed: {type(e).__name__}: {e}"

    def _wiki_search(self, p: dict) -> str:
        query = p.get("query", "")
        if not query:
            return "ERROR: No query provided."
        sentences = min(p.get("sentences", 3), 10)
        try:
            import wikipedia
            results = wikipedia.search(query, results=min(p.get("max_results", 3), 5))
            if not results:
                return f"No Wikipedia results for '{query}'"
            summaries = []
            for title in results[:3]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                    summaries.append(f"=== {page.title} ===\nURL: {page.url}\n{summary}")
                except wikipedia.exceptions.DisambiguationError as e:
                    summaries.append(f"=== {title} === (disambiguation: {', '.join(e.options[:5])})")
                except wikipedia.exceptions.PageError:
                    summaries.append(f"=== {title} === (page not found)")
            return "\n\n".join(summaries)
        except ImportError:
            return "ERROR: 'wikipedia' not installed. Run: pip install wikipedia"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _http_get(self, p: dict) -> str:
        url = p.get("url", "")
        if not url:
            return "ERROR: No url provided."
        timeout = min(p.get("timeout", 15), 60)
        headers = {}
        if p.get("headers"):
            try:
                headers = json.loads(p["headers"]) if isinstance(p["headers"], str) else p["headers"]
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            import httpx
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                return json.dumps(resp.json(), indent=2, ensure_ascii=False)[:8000]
            text = resp.text[:8000]
            if len(resp.text) > 8000:
                text += f"\n... (truncated, {len(resp.text)} chars total)"
            return f"HTTP {resp.status_code} {content_type}\n{text}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _http_request(self, p: dict) -> str:
        url = p.get("url", "")
        if not url:
            return "ERROR: No url provided."
        method = p.get("method", "GET").upper()
        timeout = min(p.get("timeout", 15), 60)
        headers = {}
        if p.get("headers"):
            try:
                headers = json.loads(p["headers"]) if isinstance(p["headers"], str) else p["headers"]
            except (json.JSONDecodeError, TypeError):
                pass
        body = p.get("body")
        if body and isinstance(body, str):
            try:
                body = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            import httpx
            resp = httpx.request(method, url, headers=headers, json=body, timeout=timeout, follow_redirects=True)
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                return json.dumps(resp.json(), indent=2, ensure_ascii=False)[:8000]
            text = resp.text[:8000]
            if len(resp.text) > 8000:
                text += f"\n... (truncated, {len(resp.text)} chars total)"
            return f"HTTP {resp.status_code} {content_type}\n{text}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _todo_tool(self, p: dict) -> str:
        action = p.get("action", "list")
        if action == "add":
            text = p.get("text", "")
            if not text:
                return "ERROR: 'todo add' requires 'text'."
            priority = p.get("priority", "normal")
            entry = {
                "id": len(self._todo) + 1,
                "text": text,
                "status": "pending",
                "priority": priority,
                "created_at": _timestamp(),
            }
            self._todo.append(entry)
            self._save_todo()
            return f"Added todo #{entry['id']}: {text} [{priority}]"
        elif action == "done":
            tid = p.get("id", "")
            if not tid:
                return "ERROR: 'todo done' requires 'id'."
            for t in self._todo:
                if t["id"] == int(tid):
                    t["status"] = "done"
                    t["done_at"] = _timestamp()
                    self._save_todo()
                    return f"Completed todo #{t['id']}: {t['text']}"
            return f"Todo #{tid} not found."
        elif action == "list":
            status = p.get("status", "")
            items = self._todo
            if status:
                items = [t for t in items if t["status"] == status]
            if not items:
                return "No todos."
            lines = []
            for t in items:
                mark = "x" if t["status"] == "done" else " "
                lines.append(f"[{mark}] #{t['id']} ({t['priority']}) {t['text']}")
            return "\n".join(lines)
        elif action == "remove":
            tid = p.get("id", "")
            if not tid:
                return "ERROR: 'todo remove' requires 'id'."
            before = len(self._todo)
            self._todo = [t for t in self._todo if t["id"] != int(tid)]
            if len(self._todo) == before:
                return f"Todo #{tid} not found."
            self._save_todo()
            return f"Removed todo #{tid}."
        elif action == "clear":
            done = [t for t in self._todo if t["status"] == "done"]
            self._todo = [t for t in self._todo if t["status"] != "done"]
            self._save_todo()
            return f"Cleared {len(done)} completed todos."
        return f"ERROR: Unknown todo action '{action}'. Use: add, done, list, remove, clear"

    def _current_state(self, p: dict) -> str:
        state = {
            "platform": sys.platform,
            "cwd": self.cwd,
            "shell": _detect_shell(),
            "timestamp": _timestamp(),
            "uptime_s": round(time.time() - _BOOT_TIME, 1),
            "pid": os.getpid(),
        }
        return json.dumps(state, indent=2)

    def _remember(self, p: dict) -> str:
        tier = p.get("tier", "core")
        if tier == "core":
            key = p.get("key", "")
            value = p.get("value", "")
            if not key or not value:
                return "ERROR: 'remember' tier=core requires 'key' and 'value'."
            return self.memory.add_core(key, value)
        content = p.get("content", "")
        if not content:
            return "ERROR: 'remember' tier=archival requires 'content'."
        return self.memory.add_archival(content, p.get("category", "general"))

    def _recall(self, p: dict) -> str:
        query = p.get("query", "")
        if not query:
            return "ERROR: 'recall' requires 'query'."
        return self.memory.recall(query, min(p.get("limit", 10), 30))

    def _core_status(self, p: dict) -> str:
        return f"=== Core Memory ===\n{self.memory.core_to_prompt()}\n\n=== Archival ===\n{self.memory.archival_status()}"

    def _forget(self, p: dict) -> str:
        tier = p.get("tier", "core")
        if tier == "core":
            key = p.get("key", "")
            if not key:
                return "ERROR: 'forget' tier=core requires 'key'."
            return self.memory.forget_core(key)
        identifier = p.get("id", "") or p.get("keyword", "")
        if not identifier:
            return "ERROR: 'forget' tier=archival requires 'id' or 'keyword'."
        return self.memory.forget_archival(identifier)

    def _delete(self, p: dict) -> str:
        target = p.get("target", "archival")
        if target == "core":
            key = p.get("key", "")
            if not key:
                return "ERROR: 'delete' target=core requires 'key'."
            return self.memory.forget_core(key)
        if target == "archival":
            identifier = p.get("id", "") or p.get("keyword", "")
            if not identifier:
                return "ERROR: 'delete' target=archival requires 'id' or 'keyword'."
            return self.memory.forget_archival(identifier)
        if target == "todo":
            tid = p.get("id", "")
            if not tid:
                return "ERROR: 'delete' target=todo requires 'id'."
            self._todo = [t for t in self._todo if t["id"] != int(tid)]
            self._save_todo()
            return f"Deleted todo #{tid}."
        if target == "file":
            path = self._resolve(p.get("path", ""))
            if not path.exists():
                return f"ERROR: File not found: {p.get('path', '')}"
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                return f"Deleted directory: {path}"
            path.unlink()
            return f"Deleted file: {path}"
        return f"ERROR: Unknown delete target '{target}'. Use: core, archival, todo, file"


# ---------------------------------------------------------------------------
# Shared display state — read by eyes_display.py and web.py
# ---------------------------------------------------------------------------
class DisplayState:
    """Thread-safe shared state for display overlays and streaming."""
    def __init__(self):
        self.lock = threading.Lock()
        self.reasoning_text: str = ""
        self.tools_text: str = ""
        self.answer_text: str = ""
        self.is_thinking: bool = False
        self.is_speaking: bool = False
        self.is_idle: bool = True
        self.ticker_text: str = ""
        self.ticker_duration: float = 0.0
        self.last_answer: str = ""
        self.last_answer_time: float = 0.0
        self.reasoning_history: str = ""
        self.current_model: str = ""

        self._sse_listeners: list = []
        self._sse_lock = threading.Lock()

    def update(self, **kwargs) -> None:
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self._notify_sse()

    def get_all(self) -> dict:
        with self.lock:
            return {
                "reasoning_text": self.reasoning_text,
                "tools_text": self.tools_text,
                "answer_text": self.answer_text,
                "is_thinking": self.is_thinking,
                "is_speaking": self.is_speaking,
                "is_idle": self.is_idle,
                "ticker_text": self.ticker_text,
                "ticker_duration": self.ticker_duration,
                "last_answer": self.last_answer,
                "last_answer_time": self.last_answer_time,
                "reasoning_history": self.reasoning_history,
                "current_model": self.current_model,
            }

    def add_sse_listener(self, q: queue.Queue) -> None:
        with self._sse_lock:
            self._sse_listeners.append(q)

    def remove_sse_listener(self, q: queue.Queue) -> None:
        with self._sse_lock:
            if q in self._sse_listeners:
                self._sse_listeners.remove(q)

    def _notify_sse(self) -> None:
        data = self.get_all()
        with self._sse_lock:
            dead = []
            for q in self._sse_listeners:
                try:
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._sse_listeners.remove(q)


# Need queue for SSE
import queue


# ---------------------------------------------------------------------------
# R2Agent — the AI agent (based on EveryLLM harness.py EveryHarness)
# ---------------------------------------------------------------------------
class R2Agent:
    """
    R2 AI Agent with:
    - EveryLLM as LLM backend (auto model selection)
    - Streaming with cancellation via command()
    - Two-tier memory (core + archival)
    - Robot API integration (servos, speech, emotes)
    - Display state for pygame overlay and web console
    """

    def __init__(
        self,
        model: str = "auto",
        cwd: str | None = None,
        max_rounds: int = 25,
        max_tokens: int = 1024,
    ):
        self.llm = EveryLLM()
        self.model = model
        self.cwd = cwd or os.getcwd()
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self._current_model: Optional[str] = None

        self.memory = Memory(self.cwd)
        self.executor = ToolExecutor(self.cwd, _detect_shell(), self.memory)
        self.history: list[dict] = []

        self.display = DisplayState()

        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.reasoning_enabled: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def command(self, prompt: str) -> None:
        """Interrupt current work and start a new prompt. Non-blocking."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._cancel_event.set()
                self._thread.join(timeout=2.0)

            self._cancel_event.clear()
            self._thread = threading.Thread(
                target=self._chat_loop,
                args=(prompt,),
                daemon=True,
                name="r2-ai-thread",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop everything."""
        self._cancel_event.set()
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
        self.display.update(
            is_idle=True, is_thinking=False, is_speaking=False,
            ticker_text="", ticker_duration=0.0,
        )

    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # LLM helpers with model fallback
    # ------------------------------------------------------------------
    def _get_model_candidates(self) -> list[str]:
        """Return ordered list of models to try."""
        if self.model != "auto":
            return [self.model]
        try:
            scores = self.llm.ttft_scores()
            ranked = sorted(scores.keys(), key=lambda m: scores[m])
        except Exception:
            ranked = []
        all_models = self.llm.models()
        seen = set()
        result = []
        for m in ranked + all_models:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _llm_create(self, messages: list[dict], **kwargs) -> Any:
        """Create non-streaming completion with model fallback."""
        if not self.reasoning_enabled:
            kwargs.setdefault("thinking", {"type": "disabled"})
        candidates = self._get_model_candidates()
        last_err = None
        for model_name in candidates:
            if self._cancel_event.is_set():
                break
            try:
                result = self.llm.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=self.max_tokens, stream=False, **kwargs,
                )
                self._current_model = model_name
                self.display.update(current_model=model_name)
                return result
            except Exception as e:
                last_err = e
                continue
        raise last_err or EveryLLMError("All models failed")

    def _llm_stream(self, messages: list[dict], **kwargs):
        """Streaming completion with model fallback."""
        if not self.reasoning_enabled:
            kwargs.setdefault("thinking", {"type": "disabled"})
        candidates = self._get_model_candidates()
        last_err = None
        for model_name in candidates:
            if self._cancel_event.is_set():
                break
            try:
                stream = self.llm.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=self.max_tokens, stream=True, **kwargs,
                )
                self._current_model = model_name
                self.display.update(current_model=model_name)
                yield from stream
                return
            except Exception as e:
                last_err = e
                continue
        raise last_err or EveryLLMError("All models failed")

    # ------------------------------------------------------------------
    # Chat loop (runs in thread)
    # ------------------------------------------------------------------
    def _chat_loop(self, prompt: str) -> None:
        with self.display.lock:
            old_reasoning = self.display.reasoning_text
            old_tools = self.display.tools_text
            old_history = self.display.reasoning_history
        combined = ""
        if old_history:
            combined = old_history
        if old_reasoning:
            combined += ("\n\n" if combined else "") + old_reasoning
        if old_tools:
            combined += ("\n\n" if combined else "") + old_tools

        self.display.update(
            is_idle=False, is_thinking=True, is_speaking=False,
            reasoning_text="", tools_text="", answer_text="",
            ticker_text="", ticker_duration=0.0,
            reasoning_history=combined,
        )

        cwd = self.cwd
        system = build_system_prompt(cwd)
        self.history = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response_text = ""
        prev_think = ""
        tool_log = ""
        self._current_model = None

        from robov_core.high_level import StreamingSpeaker
        speaker = StreamingSpeaker()

        for round_num in range(1, self.max_rounds + 1):
            if self._cancel_event.is_set():
                break

            think_buf = ""
            answer_buf = ""
            think_start = time.time()
            phase = "thinking"
            native_tool_calls = []

            self.display.update(is_thinking=True, tools_text=tool_log)

            # --- Try streaming, fallback to non-streaming ---
            streamed_ok = False
            try:
                stream = self._llm_stream(list(self.history))
                for chunk in stream:
                    if self._cancel_event.is_set():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    rc = getattr(delta, "reasoning_content", None) or ""
                    content = getattr(delta, "content", None) or ""

                    if rc and phase == "thinking":
                        think_buf += rc
                        self.display.update(reasoning_text=prev_think + ("\n" if prev_think else "") + think_buf)

                    if content:
                        if phase == "thinking":
                            phase = "answer"
                            self.display.update(
                                is_thinking=False,
                                reasoning_text=prev_think + ("\n" if prev_think else "") + think_buf,
                            )
                        answer_buf += content
                        speaker.feed(content)
                        cur = speaker.get_current_sentence()
                        self.display.update(
                            answer_text=answer_buf,
                            is_speaking=True,
                            ticker_text=cur or answer_buf,
                            ticker_duration=max(3.0, len(cur or answer_buf) / 15.0 + 1.0),
                        )
                streamed_ok = True
            except Exception as e:
                if self._cancel_event.is_set():
                    break
                prev_think += f"\n[stream error: {e}]"
                self.display.update(reasoning_text=prev_think)

            if self._cancel_event.is_set():
                break

            # --- Non-streaming fallback if streaming failed or produced nothing ---
            if not streamed_ok or (not answer_buf and not think_buf):
                try:
                    result = self._llm_create(list(self.history))
                    if result.choices:
                        msg = result.choices[0].message
                        fb_content = getattr(msg, "content", "") or ""
                        fb_reasoning = getattr(msg, "reasoning_content", "") or ""
                        native_tool_calls = getattr(msg, "tool_calls", None) or []
                        if fb_reasoning and not think_buf:
                            think_buf = fb_reasoning
                            prev_think += ("\n" if prev_think else "") + think_buf
                            self.display.update(reasoning_text=prev_think)
                        if fb_content:
                            answer_buf = fb_content
                            phase = "answer"
                            speaker.feed(fb_content)
                            self.display.update(
                                is_thinking=False, answer_text=answer_buf,
                                reasoning_text=prev_think,
                            )
                except Exception as e:
                    if self._cancel_event.is_set():
                        break
                    prev_think += f"\n[error: {e}]"
                    self.display.update(reasoning_text=prev_think)

            if self._cancel_event.is_set():
                break

            response_text = answer_buf or think_buf

            # --- If only reasoning with no content and no actions, force a non-streaming retry ---
            if answer_buf == "" and not self._extract_actions(think_buf):
                try:
                    result = self._llm_create(list(self.history))
                    if result.choices:
                        msg = result.choices[0].message
                        retry_content = getattr(msg, "content", "") or ""
                        native_tool_calls = getattr(msg, "tool_calls", None) or []
                        if retry_content:
                            answer_buf = retry_content
                            response_text = answer_buf
                            self.display.update(answer_text=answer_buf)
                except Exception:
                    pass

            # --- If completely empty, retry once more ---
            if not response_text:
                try:
                    result = self._llm_create(list(self.history))
                    if result.choices:
                        msg = result.choices[0].message
                        response_text = getattr(msg, "content", "") or ""
                        native_tool_calls = getattr(msg, "tool_calls", None) or []
                except Exception:
                    pass

            self.history.append({"role": "assistant", "content": response_text or "(no response)"})

            actions = self._extract_actions(response_text)

            for tc in native_tool_calls:
                name = tc.function.name
                try:
                    params = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    params = {}
                actions.append({"type": "tool", "name": name, "params": params})

            if not actions:
                break

            prev_think += ("\n" if prev_think else "") + think_buf
            results_text = self._execute_actions(actions)
            tool_log = self.display.get_all()["tools_text"]
            self.history.append({"role": "user", "content": results_text})
            speaker.reset()

        # Final answer — flush remaining speech + stop ticker
        if response_text and not self._cancel_event.is_set():
            clean_answer = self._strip_tags(response_text).strip()
            if clean_answer:
                speaker.flush()
                self.display.update(
                    last_answer=clean_answer,
                    last_answer_time=time.time(),
                    is_speaking=False,
                    ticker_text="",
                    ticker_duration=0.0,
                )

        self.display.update(
            is_idle=True, is_thinking=False,
        )

    # ------------------------------------------------------------------
    # Action extraction (from EveryLLM harness.py)
    # ------------------------------------------------------------------
    def _extract_actions(self, text: str) -> list[dict]:
        actions = []
        for m in PYTHON_TAG.finditer(text):
            actions.append({"type": "python", "code": m.group(1).strip()})
        for m in TOOL_CALL_TAG.finditer(text):
            name = m.group(1)
            raw_params = m.group(2).strip()
            params = self._parse_kv_params(raw_params) if raw_params else {}
            actions.append({"type": "tool", "name": name, "params": params})
        for m in TOOL_CALL_BLOCK.finditer(text):
            name = m.group(1)
            raw_params = m.group(2).strip()
            params = self._parse_kv_params(raw_params) if raw_params else {}
            actions.append({"type": "tool", "name": name, "params": params})
        return actions

    def _parse_kv_params(self, raw: str) -> dict:
        params = {}
        for m in re.finditer(r'([\w-]+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s>]*))', raw):
            key = m.group(1)
            value = m.group(2) if m.group(2) is not None else (
                m.group(3) if m.group(3) is not None else m.group(4)
            )
            params[key] = self._cast_value(value)
        return params

    @staticmethod
    def _cast_value(value: str) -> Any:
        v = value.strip()
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        if v.lower() in ("none", "null"):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_actions(self, actions: list[dict]) -> str:
        results = []
        for action in actions:
            if self._cancel_event.is_set():
                break

            start = time.time()
            if action["type"] == "python":
                output = self.executor.execute("run_python", action)
                name = "run_python"
            elif action["type"] == "tool":
                output = self.executor.execute(action["name"], action["params"])
                name = action["name"]
            else:
                continue
            elapsed = time.time() - start

            label = f"Tool {name} error:" if output.startswith("ERROR:") else f"Tool {name} output:"
            result_line = f"{label}\n{output}"
            results.append(result_line)

            # Update tools log for display
            one_line = output.replace("\n", " ").strip()[:120]
            self.display.update(
                tools_text=self.display.get_all()["tools_text"]
                + f"\n  |- {name} ({elapsed:.1f}s): {one_line}"
            )

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def _speak(self, text: str) -> None:
        try:
            from robov_core.high_level import speak
            speak(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_tags(text: str) -> str:
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"```tool_call.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"<python>.*?</python>", "", text, flags=re.DOTALL)
        return text.strip()

    def reset(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
agent: Optional[R2Agent] = None


def init_agent(model: str = "auto", cwd: str | None = None) -> R2Agent:
    global agent
    if agent is None:
        agent = R2Agent(model=model, cwd=cwd or str(Path(__file__).resolve().parent.parent))
    return agent
