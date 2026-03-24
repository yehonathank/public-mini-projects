#!/usr/bin/env python3
"""
Clinical router chat — host-driven flow:

- Prints each Question / Result from `nodes/*.md` locally (no LLM).
- After each patient reply, one API call forces `choose_next_node` with an enum of
  targets parsed from `GOTO node_id **Id**` in the current node's Logic (cannot skip ahead).

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


def _print_node_line(md: str) -> bool:
    """
    Print Question: or Result: from node body. Returns True if this is a terminal Result.
    """
    disp = parse_node_display(md)
    if not disp:
        print("Error: node has no # Question or # Result line.", file=sys.stderr)
        sys.exit(1)
    label, text = disp
    print(f"{label}: {text}\n")
    return label == "Result"


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
) -> str:
    if len(allowed) == 1:
        return allowed[0]

    tools = _choose_next_tools(allowed)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CHOOSER_SYSTEM.format(chief=chief)},
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
    tcs = getattr(msg, "tool_calls", None) or []
    if not tcs:
        print("(model): no tool call; falling back to first allowed branch.", file=sys.stderr)
        return allowed[0]

    raw = tcs[0].function.arguments or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        print(f"(model): bad tool JSON; falling back to {allowed[0]!r}.", file=sys.stderr)
        return allowed[0]

    choice = (args.get("node_id") or "").strip()
    if choice not in allowed:
        print(f"(model): invalid node_id {choice!r}; falling back to {allowed[0]!r}.", file=sys.stderr)
        return allowed[0]
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


def run_session(client: OpenAI, model: str, chief: str) -> None:
    current = "skills"
    print(f"Patient: {chief}\n")

    while True:
        try:
            md = load_node(current)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        terminal = _print_node_line(md)
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
            )
        except Exception as e:
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

    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    if env_path.is_file():
        print(f"Loaded environment from {env_path} (this file overrides shell OPENAI_* vars).")
    print(f"OPENAI_API_KEY in use: {_api_key_preview(api_key)}")
    if base_url:
        print(f"OPENAI_BASE_URL: {base_url}")
    print(f"Model: {args.model}. Commands: /quit, /clear.\n")

    print("How the visit starts — enter the patient's chief complaint.\n")

    while True:
        chief = _read_line_nonempty("Chief complaint: ")
        if chief is None:
            sys.exit(0)

        try:
            run_session(client, args.model, chief)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print("Session cleared. Enter the patient's chief complaint again.\n")


if __name__ == "__main__":
    main()
