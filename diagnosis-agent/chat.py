#!/usr/bin/env python3
"""
Clinical router chat — host-driven flow:

- Prints each Question / Result from `nodes/*.md` locally (no LLM).
- After each patient reply, one API call forces `choose_next_node` with an enum of
  targets parsed from `GOTO node_id **Id**` in the current node's Logic (cannot skip ahead).
- Each process replaces `history/history.md` (see `history_log.py`).

  pip install -r requirements.txt
  python chat.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from history_log import HISTORY_FILE, SessionHistory, reset_history_dir
from router import allowed_next_node_ids, load_node, parse_node_display

DEFAULT_MODEL = "gpt-5.1"

CHOOSER_SYSTEM = """You are a deterministic branch selector for a scripted clinical router.

You must call the tool `choose_next_node` exactly once. Pick the single `node_id` from the allowed enum that best matches the patient's latest answer according to the Logic section of the current node markdown.

Rules:
- Use the chief complaint from intake when Logic refers to it.
- If the answer fits no branch clearly, prefer re-asking: choose the current node id only if it appears in the enum and matches a "vague / re-ask" branch; otherwise choose the safest explicit branch.
- Do not invent node ids; only use the enum provided by the tool schema.
- Do not output medical advice as plain text — only the tool call.

Chief complaint at intake: {chief!r}"""


def _api_key_preview(key: str) -> str:
    k = key.strip()
    n = len(k)
    if n <= 8:
        return f"<too short, len={n}>"
    return f"{k[:12]}...{k[-4:]} (len={n})"


def _assistant_message_to_log_json(msg: Any) -> str:
    """
    Serialize `choices[0].message` from Chat Completions for interpretability logs.
    `function.arguments` is kept exactly as the API returns it (a string, usually JSON).
    """
    payload: dict[str, Any] = {
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
        "tool_calls": [],
    }
    refusal = getattr(msg, "refusal", None)
    if refusal is not None:
        payload["refusal"] = refusal

    tcs = getattr(msg, "tool_calls", None) or []
    for tc in tcs:
        fn = tc.function
        payload["tool_calls"].append(
            {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None),
                "function": {
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None),
                },
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _choose_next_tools(allowed: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_next_node",
                "description": "Select the next routing node id per Logic and the patient's answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "Next node id (must be one of the allowed values).",
                            "enum": allowed,
                        }
                    },
                    "required": ["node_id"],
                },
            },
        }
    ]


def choose_next_node_llm(
    client: OpenAI,
    model: str,
    *,
    chief: str,
    current_id: str,
    node_md: str,
    reply: str,
    allowed: list[str],
    log: SessionHistory | None = None,
) -> str:
    system_prompt = CHOOSER_SYSTEM.format(chief=chief)

    def _write_routing(
        *,
        api_called: bool,
        chosen_node_id: str,
        note: str = "",
        assistant_message_json: str | None = None,
    ) -> None:
        if log is None:
            return
        log.routing(
            patient_line=reply,
            system_prompt=system_prompt,
            current_node_id=current_id,
            chief_complaint=chief,
            allowed_node_ids=list(allowed),
            node_markdown_full=node_md,
            api_called=api_called,
            tool_name="choose_next_node",
            chosen_node_id=chosen_node_id,
            note=note,
            assistant_message_json=assistant_message_json,
        )

    if len(allowed) == 1:
        _write_routing(
            api_called=False,
            chosen_node_id=allowed[0],
            note="Only one legal branch; Chat Completions not invoked.",
            assistant_message_json=None,
        )
        return allowed[0]

    tools = _choose_next_tools(allowed)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Current node_id: {current_id}\n\n"
                f"--- node markdown ---\n{node_md}\n---\n\n"
                f"Patient's latest answer (to the question on screen):\n{reply!r}\n\n"
                "Call choose_next_node once with the matching node_id."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "choose_next_node"}},
    )
    msg = resp.choices[0].message
    assistant_json = _assistant_message_to_log_json(msg)
    tcs = getattr(msg, "tool_calls", None) or []
    extra = (msg.content or "").strip()
    extra_note = f" Assistant message text (not used for routing): {json.dumps(extra)}" if extra else ""

    if not tcs:
        print("(model): no tool call; falling back to first allowed branch.", file=sys.stderr)
        _write_routing(
            api_called=True,
            chosen_node_id=allowed[0],
            note="Model returned no tool_calls; host used first allowed id." + extra_note,
            assistant_message_json=assistant_json,
        )
        return allowed[0]

    raw = tcs[0].function.arguments or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        print(f"(model): bad tool JSON; falling back to {allowed[0]!r}.", file=sys.stderr)
        _write_routing(
            api_called=True,
            chosen_node_id=allowed[0],
            note="Invalid JSON in tool arguments; host fell back to first allowed id." + extra_note,
            assistant_message_json=assistant_json,
        )
        return allowed[0]

    choice = (args.get("node_id") or "").strip()
    if choice not in allowed:
        print(f"(model): invalid node_id {choice!r}; falling back to {allowed[0]!r}.", file=sys.stderr)
        _write_routing(
            api_called=True,
            chosen_node_id=allowed[0],
            note=f"Model chose {choice!r} (not in enum); host fell back to first allowed id." + extra_note,
            assistant_message_json=assistant_json,
        )
        return allowed[0]

    _write_routing(
        api_called=True,
        chosen_node_id=choice,
        note=("Tool arguments accepted as-is." + extra_note).strip(),
        assistant_message_json=assistant_json,
    )
    return choice


def _read_line_nonempty(prompt: str, *, allow_quit: bool = True) -> str | None:
    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not line:
            continue
        low = line.lower()
        if allow_quit and low in ("/quit", "/exit", "quit", "exit"):
            return None
        return line


def run_session(client: OpenAI, model: str, chief: str, log: SessionHistory) -> None:
    current = "skills"
    print(f"Patient: {chief}\n")

    while True:
        try:
            md = load_node(current)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        disp = parse_node_display(md)
        if not disp:
            print(
                "Error: node needs YAML front matter with kind: question|result and patient: \"…\".",
                file=sys.stderr,
            )
            sys.exit(1)
        label, text = disp
        print(f"{label}: {text}\n")
        log.display_to_patient(current, label, text)
        terminal = label == "Result"
        if terminal:
            print("Session complete (diagnosis / terminal result). Exiting.")
            sys.exit(0)

        line = input("You: ").strip()
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "quit", "exit"):
            sys.exit(0)
        if low == "/clear":
            return  # main() will re-prompt chief

        allowed = allowed_next_node_ids(md, current)
        if not allowed:
            print("Error: no allowed next nodes in Logic (check GOTO lines).", file=sys.stderr)
            sys.exit(1)

        try:
            nxt = choose_next_node_llm(
                client,
                model,
                chief=chief,
                current_id=current,
                node_md=md,
                reply=line,
                allowed=allowed,
                log=log,
            )
        except Exception as e:
            log.error(str(e))
            print(f"Error: {e}", file=sys.stderr)
            continue

        current = nxt


def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)

    p = argparse.ArgumentParser(description="Host-driven clinical router chat.")
    p.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        help=f"Chat Completions model (default {DEFAULT_MODEL}).",
    )
    args = p.parse_args()

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print(
            "Error: set OPENAI_API_KEY in diagnosis-agent/.env or the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    reset_history_dir()
    session_log = SessionHistory(model=args.model)

    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    if env_path.is_file():
        print(f"Loaded environment from {env_path} (this file overrides shell OPENAI_* vars).")
    print(f"OPENAI_API_KEY in use: {_api_key_preview(api_key)}")
    if base_url:
        print(f"OPENAI_BASE_URL: {base_url}")
    print(f"Model: {args.model}. Commands: /quit, /clear.\n")
    print(f"Interpretability log (reset each run): {HISTORY_FILE}\n")

    print("To start a visit, enter the patient's chief complaint.\n")

    first_visit = True
    while True:
        chief = _read_line_nonempty("Chief complaint: ")
        if chief is None:
            sys.exit(0)

        session_log.intake(chief, session_restart=not first_visit)
        first_visit = False

        try:
            run_session(client, args.model, chief, session_log)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print("Session cleared. Enter the patient's chief complaint again.\n")


if __name__ == "__main__":
    main()
