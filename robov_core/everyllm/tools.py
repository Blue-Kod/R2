from __future__ import annotations

import subprocess
import sys
import io
import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


def builtin_tools(cwd: str | None = None) -> list[dict]:
    base = cwd or os.getcwd()
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk. Returns file contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
                        "offset": {"type": "integer", "description": "Line number to start from (1-indexed, default 1)"},
                        "limit": {"type": "integer", "description": "Max lines to read (default 2000)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates parent dirs if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents using regex. Returns matching file paths and line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "Directory to search in (default: cwd)"},
                        "include": {"type": "string", "description": "File glob pattern to include (e.g. '*.py')"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "description": "Fetch content from a URL. Returns the response text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute a Python code snippet and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Execute a shell command and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "workdir": {"type": "string", "description": "Working directory (default: cwd)"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]


def execute_tool(name: str, arguments: dict, cwd: str | None = None) -> str:
    base = cwd or os.getcwd()
    try:
        if name == "read_file":
            return _read_file(arguments, base)
        if name == "write_file":
            return _write_file(arguments, base)
        if name == "grep":
            return _grep(arguments, base)
        if name == "webfetch":
            return _webfetch(arguments)
        if name == "run_python":
            return _run_python(arguments)
        if name == "run_bash":
            return _run_bash(arguments, base)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


def _read_file(args: dict, cwd: str) -> str:
    path = args.get("path", "")
    offset = max(args.get("offset", 1) - 1, 0)
    limit = args.get("limit", 2000)
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[offset : offset + limit]
        result = []
        for i, line in enumerate(selected, start=offset + 1):
            result.append(f"{i}: {line}")
        return "\n".join(result)
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(args: dict, cwd: str) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def _grep(args: dict, cwd: str) -> str:
    pattern = args.get("pattern", "")
    search_path = args.get("path", cwd)
    include = args.get("include")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"
    p = Path(search_path)
    if not p.is_absolute():
        p = Path(cwd) / p
    results = []
    files_searched = 0
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        if include and not f.match(include):
            continue
        if ".venv" in f.parts or "node_modules" in f.parts or ".git" in f.parts:
            continue
        files_searched += 1
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{f.relative_to(p)}:{i}: {line.strip()[:200]}")
        except Exception:
            continue
        if len(results) > 50:
            results.append("... (truncated)")
            break
    if not results:
        return f"No matches found (searched {files_searched} files)"
    return "\n".join(results[:50])


def _webfetch(args: dict) -> str:
    url = args.get("url", "")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EveryLLM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read(50_000).decode("utf-8", errors="replace")
            return data
    except Exception as e:
        return f"Error fetching URL: {e}"


def _run_python(args: dict) -> str:
    code = args.get("code", "")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
    try:
        exec(code, {"__builtins__": __builtins__})
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
    except Exception as e:
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue() + f"\n{type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    output = stdout
    if stderr:
        output += f"\nSTDERR:\n{stderr}" if output else stderr
    return output.strip() or "(no output)"


def _run_bash(args: dict, cwd: str) -> str:
    command = args.get("command", "")
    workdir = args.get("workdir", cwd)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=workdir,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
