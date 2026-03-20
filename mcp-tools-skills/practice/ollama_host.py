#!/usr/bin/env python3
"""
Local Agent Host — Ollama + Memory MCP with dynamic skill hydration.

Phase 1: Router sees only skill_manifest.yaml (no SOP) → YES/NO.
Phase 2–3: If YES, loads skills/<sop>.md into system prompt and runs tools (ReAct).

Introspection: optional <thinking>...</thinking> blocks (see INTROSPECTION_COT_RULES), trace records
written under host_history/ (history_log.jsonl per turn; history_log.json snapshot on exit).
Toggle QUIET_UI for chat-only stdout.

Usage:
  cd mcp-tools-skills/practice
  pip install -r requirements-ollama-host.txt
  ollama pull llama3.2
  python ollama_host.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient, ResponseError as OllamaResponseError

# Paths relative to this script
PRACTICE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PRACTICE_DIR / "skill_manifest.yaml"
MEMORY_SERVER = PRACTICE_DIR / "memory-mcp" / "server.py"
# Session traces (JSONL + exit snapshot) live here, not in practice/ root.
HOST_HISTORY_DIR = PRACTICE_DIR / "host_history"
# One JSON object per line; gitignored. Set to None to disable writing.
TRACE_LOG_PATH: Path | None = HOST_HISTORY_DIR / "history_log.jsonl"

# True (default): stdout is only "You:" / "Assistant:" (polished). MODEL/TOOLS/ROUTER → stderr.
# False: same sections on stdout (host diagnostics still on stderr — see _host_chatter).
QUIET_UI = True

# Regex for <thinking>...</thinking> (case-insensitive, dot matches newline)
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)

INTROSPECTION_COT_RULES = """
## Introspection (required every reply)

Before your user-visible answer (you may still call tools afterward in the same turn), write a
short private reasoning block wrapped **exactly** in these tags:

<thinking>
- What did the user say? Is it a personal fact, greeting, or something else?
- Should memory tools run? If a prior recall failed or a key was missing, what should you do next (e.g. store the new fact)?
- Which tool (if any) fits, or is plain text enough?
</thinking>

After `</thinking>`, continue with normal behavior: use native tool calls when appropriate, then
answer in plain natural language. Keep the text **outside** the tags concise for the user.
"""


def extract_thinking_blocks(text: str) -> list[str]:
    """Return inner contents of every <thinking>...</thinking> block, in order."""
    if not text:
        return []
    return [m.group(1).strip() for m in _THINKING_RE.finditer(text)]


def strip_thinking_blocks(text: str) -> str:
    """Remove all <thinking>...</thinking> blocks; used for parsing, display, and trace 'stripped' fields."""
    if not text:
        return ""
    return _THINKING_RE.sub("", text).strip()


def build_router_system(skill_id: str) -> str:
    return f"""You are a binary classification engine for routing only — not a chat assistant.

You may start with one optional block `<thinking>...</thinking>` (1–3 sentences) explaining how
the user message relates to trigger_criteria. Then output the structured lines below.

Output format (strict, after any optional thinking block):
1) First line: exactly the word YES or the word NO (uppercase), nothing else on that line.
2) Second line onward: one or two short sentences explaining WHY you chose YES or NO,
   referencing the trigger_criteria (e.g. "User stated a personal location and year" or
   "Only a greeting, no self-fact").

Rules:
- Do not greet the user. Do not answer their question. Do not role-play.
- Do not output markdown fences or bullet lists.

Read the YAML manifest below. For skill id "{skill_id}", line 1 is YES if and only if
the classifier input satisfies that skill's trigger_criteria; otherwise line 1 is NO.

--- SKILL MANIFEST (YAML) ---
"""


# Router phase: allow a second line for rationale; cap length via num_predict.
ROUTER_CHAT_OPTIONS: dict[str, Any] = {
    "temperature": 0,
    "top_p": 0.05,
    "num_predict": 180,
    "stop": ["\n\n\n"],
}

CASUAL_SYSTEM = """You are a helpful, concise assistant.
For this turn you do not have memory tools. Respond naturally to greetings and general chat.
Do not claim to have stored or recalled personal facts unless a future turn activates memory."""

CASUAL_SYSTEM_WITH_COT = CASUAL_SYSTEM.strip() + "\n\n" + INTROSPECTION_COT_RULES.strip()

EXEC_TOOL_RULES = """
## Tool use (host wiring)

- Use **native tool / function calling** only. Never paste fake tool JSON in message text.
- After tools complete, answer in plain natural language.
- For **store**, use real JSON arrays for list values in arguments when appropriate.
- **store** must have a non-empty **value** (not `""`, not `{}`, not `[]`). The host and MCP server
  reject empty values; fix **category vs fact** per the SOP and call again.
"""

EXEC_TOOL_RULES_WITH_COT = EXEC_TOOL_RULES.strip() + "\n\n" + INTROSPECTION_COT_RULES.strip()

# Plain "llama3" does not support tools in Ollama. llama3.2 / llama3.1 / qwen2.5 / mistral do.
DEFAULT_MODEL = "llama3.2"
MAX_TOOL_ROUNDS = 16
# Terminal width for section headers (model vs tools)
_OUT_W = 64


class ModelDoesNotSupportToolsError(Exception):
    """Raised when Ollama reports the tag does not support tool calling."""


TOOL_MODEL_HINT = """
[Host] This model does not support tool calling in Ollama.
       Pull and use a tool-capable model, for example:
         ollama pull llama3.2
         python ollama_host.py --model llama3.2
       Other options: llama3.1, qwen2.5, mistral, gemma3 (see ollama.com search).
"""


def load_manifest_raw() -> str:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    return MANIFEST_PATH.read_text(encoding="utf-8")


def load_manifest() -> dict[str, Any]:
    raw = load_manifest_raw()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("skill_manifest.yaml must parse to a mapping at the top level.")
    return data


def validate_manifest_against_mcp(manifest: dict[str, Any], mcp_tool_names: set[str]) -> None:
    skills = manifest.get("skills")
    if not isinstance(skills, list) or not skills:
        raise ValueError("skill_manifest.yaml must contain a non-empty 'skills' list.")
    for sk in skills:
        if not isinstance(sk, dict):
            continue
        sop_rel = sk.get("sop_file")
        if sop_rel:
            p = PRACTICE_DIR / sop_rel
            if not p.is_file():
                raise FileNotFoundError(f"SOP file missing for skill {sk.get('id')}: {p}")
        for tname in sk.get("required_tools") or []:
            if tname not in mcp_tool_names:
                raise ValueError(
                    f"Skill {sk.get('id')!r} requires unknown tool {tname!r}. "
                    f"MCP offers: {sorted(mcp_tool_names)}"
                )


def parse_router_response(text: str) -> tuple[bool, str]:
    """
    Parse router output: (route_to_skill, rationale).

    Prefers line 1 = YES/NO; remaining lines = human-readable rationale.
    Falls back to scanning early text for YES/NO if structure is messy.
    """
    raw = (text or "").strip()
    raw = re.sub(r"^```\w*\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    if not raw:
        return False, "(empty router response)"

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    def fallback_scan() -> tuple[bool, str]:
        scan = raw[:400]
        m_yes = re.search(r"\bYES\b", scan, re.IGNORECASE)
        m_no = re.search(r"\bNO\b", scan, re.IGNORECASE)
        if m_yes and m_no:
            yes_first = m_yes.start() < m_no.start()
            return (
                yes_first,
                "(fallback) Parsed YES/NO from body; conflicting tokens — used earliest match.",
            )
        if m_yes:
            tail = scan[m_yes.end() :].strip()
            return True, tail or "(fallback) YES with no separate rationale."
        if m_no:
            tail = scan[m_no.end() :].strip()
            return False, tail or "(fallback) NO with no separate rationale."
        compact = re.sub(r"^[^A-Za-z]+", "", scan[:80]).upper()
        if compact.startswith("YES"):
            return True, "(fallback) Unstructured reply; detected YES prefix."
        if compact.startswith("NO"):
            return False, "(fallback) Unstructured reply; detected NO prefix."
        return False, "(fallback) Could not detect YES/NO; defaulting to NO."

    if not lines:
        return fallback_scan()

    first = lines[0].strip()
    fu = first.upper()

    # Line 1: exact YES / NO
    if fu == "YES":
        rationale = "\n".join(lines[1:]).strip() if len(lines) > 1 else "(no rationale lines)"
        return True, rationale
    if fu == "NO":
        rationale = "\n".join(lines[1:]).strip() if len(lines) > 1 else "(no rationale lines)"
        return False, rationale

    # Line 1: YES: reason or YES — reason
    if fu.startswith("YES") and len(first) > 3 and first[3] in " \t:.-":
        rest = first[3:].lstrip(" \t:.-").strip()
        extra = "\n".join(lines[1:]).strip()
        rationale = "\n".join(x for x in (rest, extra) if x).strip() or "(no rationale)"
        return True, rationale
    if fu.startswith("NO") and len(first) > 2 and first[2] in " \t:.-":
        rest = first[2:].lstrip(" \t:.-").strip()
        extra = "\n".join(lines[1:]).strip()
        rationale = "\n".join(x for x in (rest, extra) if x).strip() or "(no rationale)"
        return False, rationale

    return fallback_scan()


def mcp_tool_to_ollama(tool: mcp_types.Tool) -> dict:
    """Convert MCP Tool to Ollama/OpenAI-style tool dict."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
            if tool.inputSchema
            else {"type": "object", "properties": {}},
        },
    }


def tool_result_to_text(result: mcp_types.CallToolResult) -> str:
    """Extract plain text from MCP CallToolResult."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))
    if result.isError and not parts:
        return "Tool returned an error with no message."
    return "\n".join(parts) if parts else "(empty result)"


def normalize_tool_arguments(raw: object) -> dict:
    """Ollama may return arguments as dict or JSON string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def coerce_store_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """If store(value) is a JSON string (e.g. '["a","b"]'), parse to list/dict."""
    if "value" not in args:
        return args
    v = args["value"]
    if not isinstance(v, str):
        return args
    s = v.strip()
    if not (s.startswith("[") or s.startswith("{")):
        return args
    try:
        parsed = json.loads(s)
        out = dict(args)
        out["value"] = parsed
        return out
    except json.JSONDecodeError:
        return args


def validate_store_arguments(args: dict[str, Any]) -> str | None:
    """
    Block schema-drift store calls before MCP (same rules as memory server).
    Returns an error message for the model, or None if OK.
    """
    key = args.get("key")
    if key is None or not str(key).strip():
        return (
            "Error: store requires a non-empty 'key' (taxonomy category, e.g. academic_field). "
            "Do not use a sentence-sized key."
        )
    if "value" not in args:
        return (
            "Error: store requires 'value' with the user's specific fact. "
            "Key = category only; value = what they said (e.g. value='mathematics')."
        )
    value = args["value"]
    if isinstance(value, str) and not value.strip():
        return (
            "Error: value is empty. Put the fact in value, not only in the key name. "
            "Example: store(key='academic_field', value='mathematics')."
        )
    if isinstance(value, dict) and len(value) == 0:
        return (
            "Error: value must not be an empty object {}. "
            "Use a taxonomy key and put the user's words in value."
        )
    if isinstance(value, list) and len(value) == 0:
        return (
            "Error: value must not be an empty list. Use a string or a list of stated items."
        )
    return None


def extract_json_objects_from_text(text: str) -> list[dict[str, Any]]:
    """Find top-level JSON objects in arbitrary text (decoder scan)."""
    decoder = json.JSONDecoder()
    i = 0
    found: list[dict[str, Any]] = []
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                found.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return found


def parse_text_tool_calls(
    content: str,
    known_tool_names: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Smaller models sometimes emit {"name": "store", "parameters": {...}} in plain
    text instead of native tool_calls. Extract and return (tool_name, args) pairs.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for obj in extract_json_objects_from_text(content):
        name = obj.get("name")
        if not isinstance(name, str) or name not in known_tool_names:
            continue
        params = obj.get("parameters")
        if not isinstance(params, dict):
            params = {k: v for k, v in obj.items() if k not in ("name", "type")}
        if name == "store":
            params = coerce_store_arguments(params)
        calls.append((name, params))
    return calls


def strip_parsed_tool_json_blobs(content: str, known_tool_names: set[str]) -> str:
    """Remove JSON blobs that were executed as synthetic tool calls (cleaner history)."""
    decoder = json.JSONDecoder()
    remove_spans: list[tuple[int, int]] = []
    i = 0
    n = len(content)
    while i < n:
        if content[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(content, i)
            if isinstance(obj, dict):
                name = obj.get("name")
                if isinstance(name, str) and name in known_tool_names and (
                    "parameters" in obj or name in obj
                ):
                    remove_spans.append((i, end))
            i = end
        except json.JSONDecodeError:
            i += 1
    if not remove_spans:
        return content.strip()
    out_parts: list[str] = []
    cursor = 0
    for start, end in remove_spans:
        out_parts.append(content[cursor:start])
        cursor = end
    out_parts.append(content[cursor:])
    cleaned = "".join(out_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if cleaned else "(Used tools; see tool results above.)"


async def call_mcp_tool(
    session: ClientSession,
    known_names: set[str],
    name: str,
    arguments: object,
) -> str:
    if name not in known_names:
        return (
            f"Error: Unknown tool '{name}'. "
            f"Valid tools: {', '.join(sorted(known_names))}. "
            "Pick one of these and try again."
        )
    args = normalize_tool_arguments(arguments)
    if name == "store" and isinstance(args, dict):
        args = coerce_store_arguments(args)
        bad = validate_store_arguments(args)
        if bad:
            return bad
    try:
        result = await session.call_tool(name, args)
        return tool_result_to_text(result)
    except Exception as e:  # noqa: BLE001
        return f"Error executing tool '{name}': {e!s}. Try fixing arguments or use another tool."


def _host_chatter(msg: str) -> None:
    """All [Host] diagnostics and phase labels go to stderr so stdout stays readable."""
    print(msg, file=sys.stderr)


def print_banner() -> None:
    print("=" * 60, file=sys.stderr)
    print("  Ollama + Memory MCP — type 'quit' or 'exit' to stop", file=sys.stderr)
    print(
        f"  QUIET_UI={'True' if QUIET_UI else 'False'} → "
        f"{'chat on stdout; sections on stderr' if QUIET_UI else 'sections on stdout'}",
        file=sys.stderr,
    )
    print("=" * 60, file=sys.stderr)


def print_polished_assistant(raw_content: str) -> None:
    """User-visible reply only (thinking tags removed)."""
    clean = strip_thinking_blocks(raw_content or "").strip() or "(no reply)"
    print(f"\nAssistant:\n{clean}\n")


def append_trace_record(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[Host] Trace write failed: {e}", file=sys.stderr)


def _last_assistant_content(messages: list[Any]) -> str:
    """Best-effort last assistant message content (dict or Ollama message object)."""
    for m in reversed(messages):
        if isinstance(m, dict):
            if m.get("role") == "assistant":
                return str(m.get("content") or "")
        role = getattr(m, "role", None)
        if role == "assistant":
            return str(getattr(m, "content", None) or "")
    return ""


def _model_tools_stream() -> Any:
    """MODEL/TOOLS/ROUTER sections: stderr when QUIET_UI (chat-only stdout), else stdout."""
    return sys.stderr if QUIET_UI else sys.stdout


def _print_host_round(round_idx: int) -> None:
    print(
        f"\n{'·' * 3} [Host] Ollama round {round_idx + 1} {'·' * 3}",
        file=_model_tools_stream(),
    )


def print_model_section(
    subtitle: str,
    body: str,
    *,
    strip_thinking: bool = True,
    stream: Any | None = None,
) -> None:
    """
    What the LLM is saying (natural language). Not tool execution.
    By default strips <thinking>...</thinking> so introspection stays in trace files only.
    Output stream follows QUIET_UI unless overridden (stderr when quiet, stdout when verbose).
    """
    out = stream if stream is not None else _model_tools_stream()
    w = _OUT_W
    bar = "═" * w
    text = (body or "").rstrip()
    if strip_thinking:
        text = strip_thinking_blocks(text)
    print(f"\n{bar}", file=out)
    print(f"  MODEL · {subtitle}", file=out)
    print("─" * w, file=out)
    if text:
        for line in text.splitlines():
            print(f"  {line}", file=out)
    else:
        print("  (no text in this turn)", file=out)
    print(bar, file=out)


def print_tools_section_start() -> None:
    """Start of MCP tool execution (not the model speaking to the user)."""
    out = _model_tools_stream()
    w = _OUT_W
    bar = "═" * w
    print(f"\n{bar}", file=out)
    print("  TOOLS · executed via MCP (scratchpad.json on disk)", file=out)
    print("─" * w, file=out)


def print_tools_section_end() -> None:
    print("═" * _OUT_W, file=_model_tools_stream())


def print_router_interpretability(
    decision: bool,
    rationale: str,
    *,
    raw_model_reply: str | None = None,
) -> None:
    """Structured Phase 1: decision + why; optional full model text for audit."""
    out = _model_tools_stream()
    w = _OUT_W
    bar = "═" * w
    print(f"\n{bar}", file=out)
    print("  ROUTER · interpretability", file=out)
    print("─" * w, file=out)
    branch = "YES → hydrate SOP + MCP tools" if decision else "NO → casual chat (no tools)"
    print(f"  Decision: {'YES' if decision else 'NO'}  ({branch})", file=out)
    print("  Why:", file=out)
    text = (rationale or "").strip() or "(no rationale parsed)"
    for line in text.splitlines():
        print(f"    {line}", file=out)
    raw = (raw_model_reply or "").rstrip()
    if raw:
        print("  Model raw (what the router model actually returned):", file=out)
        for line in raw.splitlines():
            print(f"    {line}", file=out)
    print(bar, file=out)


def print_tool_invocation(
    name: str,
    args_summary: str,
    *,
    from_parsed_text: bool = False,
) -> None:
    out = _model_tools_stream()
    note = " — parsed from model text, not native tool_call" if from_parsed_text else ""
    print(f"  ▸ {name}{note}", file=out)
    print(f"     {args_summary}", file=out)


def print_tool_result(snippet: str, max_len: int = 500) -> None:
    out = _model_tools_stream()
    one_line = snippet.replace("\n", " ").strip()
    if len(one_line) > max_len:
        one_line = one_line[: max_len - 1] + "…"
    print(f"  ◀ {one_line}", file=out)


async def run_tool_execution_loop(
    client: Any,
    model: str,
    messages: list,
    ollama_tools: list[dict],
    allowed_tool_names: set[str],
    session: ClientSession,
    mcp_tool_names: set[str],
    *,
    quiet_ui: bool = False,
    turn_trace: dict[str, Any] | None = None,
) -> None:
    """Mutates messages. ReAct until assistant returns without tool calls."""
    if turn_trace is not None:
        turn_trace.setdefault("thinking_executor", [])
        turn_trace.setdefault("tool_calls_traced", [])

    for round_idx in range(MAX_TOOL_ROUNDS):
        _print_host_round(round_idx)
        try:
            response = await client.chat(
                model=model,
                messages=messages,
                tools=ollama_tools,
            )
        except OllamaResponseError as e:
            err = str(getattr(e, "error", e)).lower()
            if "does not support tools" in err:
                raise ModelDoesNotSupportToolsError from e
            raise
        msg = response.message

        tool_calls = getattr(msg, "tool_calls", None) or []
        raw_content = msg.content or ""
        content = raw_content.strip()

        if turn_trace is not None and content:
            turn_trace["thinking_executor"].extend(extract_thinking_blocks(content))

        text_calls: list[tuple[str, dict[str, Any]]] = []
        if not tool_calls and content:
            text_calls = parse_text_tool_calls(
                strip_thinking_blocks(content),
                allowed_tool_names,
            )

        if not tool_calls and not text_calls:
            final = raw_content or ""
            if quiet_ui:
                print_polished_assistant(final)
            else:
                print_model_section(
                    "assistant — this is what you read (final for this turn)",
                    final,
                )
            messages.append({"role": "assistant", "content": final})
            break

        if content.strip() and tool_calls:
            print_model_section(
                "draft text (model will run tools next — not your final answer yet)",
                content,
            )

        if tool_calls:
            tool_calls_out: list[dict[str, Any]] = []
            for tc in tool_calls:
                fn = tc.function
                cid = getattr(tc, "id", None) or str(uuid.uuid4())
                raw_args = fn.arguments
                if isinstance(raw_args, str):
                    args_str = raw_args
                else:
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                tool_calls_out.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": fn.name, "arguments": args_str},
                    }
                )
            asst_content = content if content.strip() else ""
            messages.append(
                {
                    "role": "assistant",
                    "content": asst_content,
                    "tool_calls": tool_calls_out,
                }
            )
            print_tools_section_start()
            for idx, tc in enumerate(tool_calls):
                fn = tc.function
                name = fn.name
                cid = tool_calls_out[idx]["id"]
                if name not in allowed_tool_names:
                    err = (
                        f"Tool '{name}' is not allowed for this skill. "
                        f"Allowed: {', '.join(sorted(allowed_tool_names))}."
                    )
                    print_tool_invocation(
                        name,
                        json.dumps(normalize_tool_arguments(fn.arguments), ensure_ascii=False),
                    )
                    print_tool_result(err)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": cid,
                            "tool_name": name,
                            "content": err,
                        }
                    )
                    continue
                args = normalize_tool_arguments(fn.arguments)
                args_summary = json.dumps(args, ensure_ascii=False)
                print_tool_invocation(name, args_summary)

                result_text = await call_mcp_tool(session, mcp_tool_names, name, fn.arguments)
                print_tool_result(result_text)

                if turn_trace is not None:
                    turn_trace["tool_calls_traced"].append(
                        {
                            "name": name,
                            "arguments": args,
                            "result_preview": (result_text or "")[:800],
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "tool_name": name,
                        "content": result_text,
                    }
                )
            print_tools_section_end()
        else:
            cleaned = strip_parsed_tool_json_blobs(content, allowed_tool_names)
            note = (
                "\n[Host] Note: tool calls were embedded in model text; "
                "prefer native tool calling (see SOP)."
            )
            print(note, file=sys.stderr)
            if cleaned.strip():
                print_model_section(
                    "assistant text (cleaned; JSON tool blobs removed)",
                    cleaned,
                )
            exec_text_calls = [(n, a) for n, a in text_calls if n in allowed_tool_names]
            st_calls: list[dict[str, Any]] = []
            for i, (tname, targs) in enumerate(exec_text_calls):
                cid = f"call_txt_{round_idx}_{i}"
                tdict = targs if isinstance(targs, dict) else {}
                st_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tname,
                            "arguments": json.dumps(tdict, ensure_ascii=False),
                        },
                    }
                )
            if exec_text_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": cleaned if cleaned.strip() else "",
                        "tool_calls": st_calls,
                    }
                )
            else:
                messages.append({"role": "assistant", "content": cleaned})

            print_tools_section_start()
            for i, (name, args) in enumerate(exec_text_calls):
                cid = st_calls[i]["id"]
                raw_args = args
                args_summary = json.dumps(args, ensure_ascii=False)
                print_tool_invocation(
                    name,
                    args_summary,
                    from_parsed_text=True,
                )
                result_text = await call_mcp_tool(session, mcp_tool_names, name, raw_args)
                print_tool_result(result_text)
                if turn_trace is not None:
                    turn_trace["tool_calls_traced"].append(
                        {
                            "name": name,
                            "arguments": dict(args) if isinstance(args, dict) else args,
                            "result_preview": (result_text or "")[:800],
                            "from_parsed_text": True,
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "tool_name": name,
                        "content": result_text,
                    }
                )
            print_tools_section_end()
    else:
        trunc = "\n[Host] Max tool rounds reached; truncating."
        print(trunc, file=sys.stderr)
        if quiet_ui:
            print_polished_assistant("[Host stopped: too many tool rounds.]")
        else:
            print_model_section(
                "assistant (host stopped the loop)",
                "[Host stopped: too many tool rounds.]",
            )
        messages.append(
            {
                "role": "assistant",
                "content": "[Host stopped: too many tool rounds.]",
            }
        )


async def run_agent_turn(
    client: Any,
    model: str,
    user_line: str,
    history: list[dict[str, Any]],
    session: ClientSession,
    ollama_tools: list[dict],
    mcp_tool_names: set[str],
    manifest_raw: str,
    primary: dict[str, Any],
) -> dict[str, Any]:
    """
    One user message: Phase 1 router → casual OR hydrated SOP + tools.
    Returns a trace dict (query, context, thinking, response) for logging.
    """
    primary_id = primary.get("id")
    if not primary_id:
        raise ValueError("Primary skill missing 'id' in manifest.")

    turn_data: dict[str, Any] = {
        "query": user_line,
        "context": {},
        "thinking": {
            "router_blocks": [],
            "executor_blocks": [],
            "casual_blocks": [],
        },
        "response": {
            "phase1_decision": None,
            "router_rationale": "",
            "final_assistant_text_raw": "",
            "final_assistant_text_stripped": "",
            "tool_calls": [],
        },
    }

    user_msg = {"role": "user", "content": user_line}

    _host_chatter("\n[Host] Phase 1 · Router — manifest only (no SOP, no tools)")
    router_system = build_router_system(str(primary_id)) + manifest_raw
    turn_data["context"]["router_system_prompt"] = router_system
    turn_data["context"]["hydrated_manifest_yaml"] = manifest_raw

    router_user_turn = {
        "role": "user",
        "content": (
            "CLASSIFIER_INPUT (single user turn — classify only this):\n"
            "----\n"
            f"{user_line}\n"
            "----\n"
            "Line 1: YES or NO only. Line 2+: one short sentence explaining why."
        ),
    }
    router_messages: list = [
        {"role": "system", "content": router_system},
        *history,
        router_user_turn,
    ]
    try:
        route_resp = await client.chat(
            model=model,
            messages=router_messages,
            options=ROUTER_CHAT_OPTIONS,
            think=False,
        )
    except OllamaResponseError as e:
        err = str(getattr(e, "error", e)).lower()
        if "does not support tools" in err:
            raise ModelDoesNotSupportToolsError from e
        raise

    route_text = route_resp.message.content or ""
    turn_data["thinking"]["router_blocks"] = extract_thinking_blocks(route_text)
    route_for_parse = strip_thinking_blocks(route_text)
    use_skill, router_rationale = parse_router_response(route_for_parse)

    turn_data["response"]["phase1_decision"] = "YES" if use_skill else "NO"
    turn_data["response"]["router_rationale"] = router_rationale

    if not QUIET_UI:
        print_model_section(
            "router · decision (YES/NO — parsed; rationale is below)",
            "YES" if use_skill else "NO",
        )
        print_router_interpretability(
            use_skill,
            router_rationale,
            raw_model_reply=route_text,
        )
    _host_chatter(
        f"[Host] Phase 1 result → "
        f"{'YES — hydrate SOP + tools' if use_skill else 'NO — casual chat (no tools)'}",
    )

    if not use_skill:
        conv = [{"role": "system", "content": CASUAL_SYSTEM_WITH_COT}, *history, user_msg]
        turn_data["context"]["casual_system_prompt"] = CASUAL_SYSTEM_WITH_COT
        turn_data["context"]["executor_system_prompt"] = None
        turn_data["context"]["filtered_tool_names"] = []

        chat_resp = await client.chat(model=model, messages=conv)
        final = chat_resp.message.content or ""
        turn_data["thinking"]["casual_blocks"] = extract_thinking_blocks(final)
        turn_data["response"]["final_assistant_text_raw"] = final
        turn_data["response"]["final_assistant_text_stripped"] = strip_thinking_blocks(final).strip()

        if QUIET_UI:
            print_polished_assistant(final)
        else:
            print_model_section(
                "assistant — this is what you read (final for this turn)",
                final,
            )
        history.append(user_msg)
        history.append({"role": "assistant", "content": final})
        return turn_data

    # --- Phase 2–3: hydrate SOP + filtered tools, then ReAct ---
    _host_chatter("\n[Host] Phase 2–3 · Hydrated SOP + MCP tools (execution)")
    sop_rel = primary.get("sop_file")
    if not sop_rel:
        raise ValueError(f"Skill {primary.get('id')!r} missing sop_file")
    sop_path = PRACTICE_DIR / sop_rel
    sop_text = sop_path.read_text(encoding="utf-8")
    required = set(primary.get("required_tools") or [])
    filtered_tools = [
        t
        for t in ollama_tools
        if t.get("function", {}).get("name") in required
    ]
    if not filtered_tools:
        filtered_tools = ollama_tools
    allowed_names = {t["function"]["name"] for t in filtered_tools}

    exec_system = sop_text.strip() + "\n\n" + EXEC_TOOL_RULES_WITH_COT.strip()
    turn_data["context"]["executor_system_prompt"] = exec_system
    turn_data["context"]["casual_system_prompt"] = None
    turn_data["context"]["filtered_tool_names"] = sorted(allowed_names)
    turn_data["context"]["sop_file"] = str(sop_rel)

    messages_exec = [{"role": "system", "content": exec_system}, *history, user_msg]
    anchor = len(messages_exec)

    exec_trace: dict[str, Any] = {}
    await run_tool_execution_loop(
        client,
        model,
        messages_exec,
        filtered_tools,
        allowed_names,
        session,
        mcp_tool_names,
        quiet_ui=QUIET_UI,
        turn_trace=exec_trace,
    )

    turn_data["thinking"]["executor_blocks"] = exec_trace.get("thinking_executor", [])
    turn_data["response"]["tool_calls"] = exec_trace.get("tool_calls_traced", [])

    last_assistant = _last_assistant_content(messages_exec)
    turn_data["response"]["final_assistant_text_raw"] = last_assistant
    turn_data["response"]["final_assistant_text_stripped"] = strip_thinking_blocks(
        last_assistant
    ).strip()

    history.append(user_msg)
    history.extend(messages_exec[anchor:])
    return turn_data


async def run_chat_loop(
    session: ClientSession,
    ollama_tools: list[dict],
    mcp_tool_names: set[str],
    model: str,
) -> None:
    manifest = load_manifest()
    validate_manifest_against_mcp(manifest, mcp_tool_names)
    manifest_raw = load_manifest_raw()
    skills = manifest["skills"]
    primary = next((s for s in skills if isinstance(s, dict)), None)
    if not primary:
        raise ValueError("No valid skill entries in manifest.")
    primary_id = primary.get("id")
    if not primary_id:
        raise ValueError("Primary skill missing 'id' in manifest.")

    client = AsyncClient()
    history: list[dict[str, Any]] = []
    session_trace: list[dict[str, Any]] = []

    print_banner()
    print(f"[Host] Manifest: {MANIFEST_PATH}", file=sys.stderr)
    print(f"[Host] Router skill id: {primary_id!r}", file=sys.stderr)
    if TRACE_LOG_PATH:
        _host_chatter(f"[Host] Trace log (JSONL): {TRACE_LOG_PATH}")

    while True:
        try:
            user_line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Host] Goodbye.", file=sys.stderr)
            break

        if not user_line:
            continue
        if user_line.lower() in ("quit", "exit", "q"):
            print("[Host] Goodbye.", file=sys.stderr)
            break

        turn_data = await run_agent_turn(
            client,
            model,
            user_line,
            history,
            session,
            ollama_tools,
            mcp_tool_names,
            manifest_raw,
            primary,
        )

        session_trace.append(turn_data)
        append_trace_record(TRACE_LOG_PATH, turn_data)

    # Snapshot full session for easy inspection (optional companion to JSONL)
    snapshot_path = HOST_HISTORY_DIR / "history_log.json"
    if session_trace:
        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(
                json.dumps({"turns": session_trace}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _host_chatter(f"[Host] Wrote session trace snapshot: {snapshot_path}")
        except OSError as e:
            print(f"[Host] Could not write {snapshot_path}: {e}", file=sys.stderr)


async def async_main(model: str) -> int:
    if not MEMORY_SERVER.exists():
        print(f"Error: Memory MCP server not found at {MEMORY_SERVER}", file=sys.stderr)
        return 1

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MEMORY_SERVER)],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = listed.tools
            ollama_tools = [mcp_tool_to_ollama(t) for t in tools]
            known_names = {t.name for t in tools}

            print(
                f"[Host] Connected to Memory MCP. Tools: {', '.join(sorted(known_names))}",
                file=sys.stderr,
            )
            print(
                f"[Host] Model: {model} | Manifest: {MANIFEST_PATH}",
                file=sys.stderr,
            )

            try:
                await run_chat_loop(session, ollama_tools, known_names, model)
            except ModelDoesNotSupportToolsError:
                print(TOOL_MODEL_HINT.strip(), file=sys.stderr)
                return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama host: manifest router + hydrated SOP + Memory MCP",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    code = asyncio.run(async_main(args.model))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
