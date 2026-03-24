"""
OpenAI Chat Completions adapter with the same async `chat()` shape as `ollama.AsyncClient`.

Used by eval_runner (and optionally other callers) so ollama_host can drive GPT models
without changing its ReAct loop. API errors are mapped to ollama.ResponseError so
existing host error handling stays valid.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from ollama import ResponseError as OllamaResponseError

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI
except ImportError as e:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore[misc, assignment]
    APIError = Exception  # type: ignore[misc, assignment]
    APITimeoutError = Exception  # type: ignore[misc, assignment]
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None


def _tool_calls_with_string_arguments(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chat Completions API expects each tool call's function.arguments as a JSON string.
    The host keeps dict arguments for Ollama compatibility; stringify here for OpenAI.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        entry = dict(tc)
        fn = entry.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                fn["arguments"] = json.dumps(raw, ensure_ascii=False)
            elif raw is None:
                fn["arguments"] = "{}"
            elif not isinstance(raw, str):
                fn["arguments"] = json.dumps(raw, ensure_ascii=False)
            entry["function"] = fn
        normalized.append(entry)
    return normalized


def _openai_assistant_dict(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    if role != "assistant":
        raise ValueError("expected assistant message")
    out: dict[str, Any] = {"role": "assistant"}
    tcs = msg.get("tool_calls")
    c = msg.get("content")
    if tcs:
        if isinstance(tcs, list):
            out["tool_calls"] = _tool_calls_with_string_arguments(
                [x for x in tcs if isinstance(x, dict)]
            )
        else:
            out["tool_calls"] = tcs
        cs = (c or "").strip()
        out["content"] = cs if cs else None
    else:
        out["content"] = "" if c is None else c
    return out


def _strip_message_for_openai(msg: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields the OpenAI Chat Completions API accepts."""
    role = msg.get("role")
    if role == "system":
        return {"role": "system", "content": msg.get("content") or ""}
    if role == "user":
        out_u: dict[str, Any] = {"role": "user", "content": msg.get("content") or ""}
        if msg.get("name"):
            out_u["name"] = msg["name"]
        return out_u
    if role == "assistant":
        return _openai_assistant_dict(msg)
    if role == "tool":
        tid = msg.get("tool_call_id")
        if not tid:
            raise ValueError("tool message missing tool_call_id (required for OpenAI)")
        return {
            "role": "tool",
            "tool_call_id": tid,
            "content": msg.get("content") or "",
        }
    raise ValueError(f"unsupported message role for OpenAI: {role!r}")


def _normalize_messages_for_openai(messages: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise TypeError(f"OpenAI path expects dict messages, got {type(m)}")
        out.append(_strip_message_for_openai(m))
    return out


def _msg_from_openai_choice(choice: Any) -> SimpleNamespace:
    m = choice.message
    content = m.content or ""
    raw_tcs = getattr(m, "tool_calls", None) or []
    tool_calls: list[SimpleNamespace] = []
    for tc in raw_tcs:
        fn = tc.function
        tool_calls.append(
            SimpleNamespace(
                id=getattr(tc, "id", None),
                function=SimpleNamespace(
                    name=fn.name,
                    arguments=fn.arguments if fn.arguments is not None else "{}",
                ),
            )
        )
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls or None,
        )
    )


class OpenAIChatAdapter:
    """Drop-in replacement for `AsyncClient` in `run_agent_turn` / tool loop."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package required (pip install openai)") from _OPENAI_IMPORT_ERROR
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        *,
        model: str,
        messages: list,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> SimpleNamespace:
        del think
        try:
            api_messages = _normalize_messages_for_openai(messages)
            kwargs: dict[str, Any] = {"model": model, "messages": api_messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if options:
                t = options.get("temperature")
                if t is not None:
                    kwargs["temperature"] = t
                np = options.get("num_predict")
                if np is not None:
                    kwargs["max_completion_tokens"] = int(np)
            resp = await self._client.chat.completions.create(**kwargs)
        except (APIError, APITimeoutError) as e:
            code = int(getattr(e, "status_code", None) or -1)
            raise OllamaResponseError(str(e), code) from e
        except OllamaResponseError:
            raise
        except Exception as e:
            raise OllamaResponseError(str(e), -1) from e
        return _msg_from_openai_choice(resp.choices[0])
