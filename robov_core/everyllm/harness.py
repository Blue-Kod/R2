from __future__ import annotations

import re
import subprocess
import sys
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional


PYTHON_TAG = re.compile(r"<python>(.*?)</python>", re.DOTALL)
TOOL_TAG = re.compile(r"<tool_call>(\w+)(.*?)</tool_call>", re.DOTALL)
PARAM_LINE = re.compile(r"^\s*(\w+)\s*[:=]\s*(.*?)\s*$")


class Harness:
    def __init__(
        self,
        model_fn: Callable[[list[dict]], str],
        cwd: str | None = None,
        max_rounds: int = 30,
        on_tool: Callable[[str, dict, str], None] | None = None,
        on_python: Callable[[str, str], None] | None = None,
        on_round: Callable[[int, str], None] | None = None,
    ):
        self.model_fn = model_fn
        self.cwd = cwd or os.getcwd()
        self.max_rounds = max_rounds
        self.env: dict[str, Any] = {"__builtins__": __builtins__, "cwd": self.cwd}
        self.history: list[dict] = []
        self.on_tool = on_tool
        self.on_python = on_python
        self.on_round = on_round

    def run(self, user_message: str, system: str = "") -> str:
        self.history = []
        if system:
            self.history.append({"role": "system", "content": system})
        self.history.append({"role": "user", "content": user_message})

        response = ""
        for round_num in range(1, self.max_rounds + 1):
            response = self.model_fn(list(self.history))
            self.history.append({"role": "assistant", "content": response})

            if self.on_round:
                self.on_round(round_num, response)

            actions = self._extract_actions(response)
            if not actions:
                return response

            results_text = self._execute_actions(actions)
            self.history.append({"role": "user", "content": results_text})

        return response

    def _extract_actions(self, text: str) -> list[dict]:
        actions = []

        for m in PYTHON_TAG.finditer(text):
            actions.append({"type": "python", "code": m.group(1).strip()})

        for m in TOOL_TAG.finditer(text):
            tool_name = m.group(1)
            raw_params = m.group(2).strip()
            params = self._parse_tool_params(raw_params)
            actions.append({"type": "tool", "name": tool_name, "params": params})

        return actions

    def _parse_tool_params(self, raw: str) -> dict:
        params = {}
        lines = raw.strip().splitlines()
        multiline_key = None
        multiline_buf = []

        for line in lines:
            if multiline_key:
                if line.strip() == "--end":
                    params[multiline_key] = "\n".join(multiline_buf)
                    multiline_key = None
                    multiline_buf = []
                else:
                    multiline_buf.append(line)
                continue

            m = PARAM_LINE.match(line)
            if m:
                key, value = m.group(1), m.group(2)
                if value == "--":
                    multiline_key = key
                    multiline_buf = []
                else:
                    params[key] = self._cast_value(value)
            elif line.strip():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    params[parts[0]] = self._cast_value(parts[1])

        return params

    @staticmethod
    def _cast_value(value: str) -> Any:
        v = value.strip()
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            return v[1:-1]
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

    def _execute_actions(self, actions: list[dict]) -> str:
        results = []
        for action in actions:
            if action["type"] == "python":
                output = self._exec_python(action["code"])
                if output.startswith("ERROR:"):
                    results.append(f"Python error:\n{output}")
                else:
                    results.append(f"Python output:\n{output}")
            elif action["type"] == "tool":
                output = self._exec_tool(action["name"], action["params"])
                if output.startswith("ERROR:"):
                    results.append(f"Tool {action['name']} error:\n{output}")
                else:
                    results.append(f"Tool {action['name']} output:\n{output}")
        return "\n\n".join(results)

    def _exec_python(self, code: str) -> str:
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = buf_out
        sys.stderr = buf_err
        try:
            exec(code, self.env)
            stdout = buf_out.getvalue()
            stderr = buf_err.getvalue()
        except Exception as e:
            stdout = buf_out.getvalue()
            stderr = buf_err.getvalue() + f"\n{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        output = stdout
        if stderr:
            output += f"\nSTDERR:\n{stderr}" if output else stderr
        result = output.strip() or "(no output)"
        if self.on_python:
            self.on_python(code, result)
        if "Error" in result or "Exception" in result or "Traceback" in result:
            return f"ERROR: {result}"
        return result

    def _exec_tool(self, name: str, params: dict) -> str:
        HANDLERS = {
            "read": self._tool_read,
            "write": self._tool_write,
            "edit": self._tool_edit,
            "grep": self._tool_grep,
            "mkdir": self._tool_mkdir,
            "ls": self._tool_ls,
            "cd": self._tool_cd,
        }
        handler = HANDLERS.get(name)
        if not handler:
            return f"ERROR: Unknown tool '{name}'. Available: {', '.join(HANDLERS)}"
        try:
            result = handler(params)
        except Exception as e:
            result = f"ERROR: {type(e).__name__}: {e}"
        if self.on_tool:
            self.on_tool(name, params, result)
        if result.startswith("ERROR:"):
            return result
        return result

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p

    def _tool_read(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        offset = max(p.get("offset", 1) - 1, 0)
        limit = p.get("limit", 2000)
        if not path.exists():
            return f"ERROR: File not found: {p.get('path', '')}"
        if path.is_dir():
            return f"ERROR: Is a directory: {path}. Use ls instead."
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[offset : offset + limit]
        return "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=offset + 1))

    def _tool_write(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        content = p.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path.name}"

    def _tool_edit(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        old = p.get("old", "")
        new = p.get("new", "")
        if not path.exists():
            return f"ERROR: File not found: {p.get('path', '')}"
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return f"ERROR: String not found in {path.name}"
        count = text.count(old)
        if count > 1 and not p.get("all"):
            return f"ERROR: Found {count} occurrences. Add all=yes to replace all."
        new_text = text.replace(old, new) if p.get("all") else text.replace(old, new, 1)
        path.write_text(new_text, encoding="utf-8")
        return f"Edited {path.name} ({count} replacement{'s' if count > 1 else ''})"

    def _tool_grep(self, p: dict) -> str:
        import re as re_mod
        pattern = p.get("pattern", "")
        search_path = self._resolve(p.get("path", "."))
        include = p.get("include")
        try:
            regex = re_mod.compile(pattern, re_mod.IGNORECASE)
        except re_mod.error as e:
            return f"ERROR: Invalid regex: {e}"
        results = []
        count = 0
        for f in search_path.rglob("*"):
            if not f.is_file():
                continue
            if include and not f.match(include):
                continue
            if any(skip in f.parts for skip in (".venv", "node_modules", ".git", "__pycache__")):
                continue
            count += 1
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
            return f"No matches in {count} files"
        return "\n".join(results)

    def _tool_mkdir(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        path.mkdir(parents=True, exist_ok=True)
        return f"Created: {path}"

    def _tool_ls(self, p: dict) -> str:
        path = self._resolve(p.get("path", "."))
        show_all = p.get("all", False)
        if not path.exists():
            return f"ERROR: Path not found: {p.get('path', '')}"
        if path.is_file():
            return str(path.name)
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            if not show_all and e.name.startswith("."):
                continue
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                size = e.stat().st_size
                lines.append(f"  {e.name} ({size}b)")
        return "\n".join(lines) or "(empty)"

    def _tool_cd(self, p: dict) -> str:
        path = self._resolve(p.get("path", ""))
        if not path.exists():
            return f"ERROR: Path not found: {p.get('path', '')}"
        if not path.is_dir():
            return f"ERROR: Not a directory: {path}"
        self.cwd = str(path)
        self.env["cwd"] = self.cwd
        return f"Changed directory to: {self.cwd}"

    def summary(self) -> str:
        lines = []
        for i, msg in enumerate(self.history):
            role = msg["role"]
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)
