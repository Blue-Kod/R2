from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCallFunction:
    name: str = ""
    arguments: str = ""

    def arguments_dict(self) -> dict:
        try:
            return json.loads(self.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ToolCall:
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = field(default_factory=ToolCallFunction)


@dataclass
class Message:
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    model_used: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class ChoiceChunk:
    index: int = 0
    delta: Delta = field(default_factory=Delta)
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    model_used: str = ""
    choices: list[ChoiceChunk] = field(default_factory=list)
