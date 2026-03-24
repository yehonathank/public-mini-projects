#!/usr/bin/env python3
"""
Structured eval harness for ollama_host.py: router (Phase 1) + tool payload checks.

Writes one folder per run under eval_runs/. Default name is human-readable, e.g.:
  eval_runs/2026-03-20_173045Z__model-llama3-2__6trials-each/

Inside each run:
  README.txt            — what each file/folder is
  summary.txt           — full text report
  meta.json             — machine-readable metadata
  mcp_verification.tsv  — table for spreadsheets
  cases/                — one subfolder per baseline scenario
    C1_Casual/          — trial_01.json, trial_02.json, …

Usage (from mcp-tools-skills/practice):
  python eval_runner.py                      # 30 matrix rows × 1 trial each = 30 Ollama turns
  python eval_runner.py --relax-store-keys   # ignore key_any_of if value_substrings/list still match
  python eval_runner.py --trials-per-case 6  # 30 rows × 6 = 180 turns
  python eval_runner.py --model llama3.2 --verbose
  python eval_runner.py --output-dir eval_runs/my_custom_name

Requires:
  - Ollama (default): server running locally with the chosen model pulled.
  - OpenAI: `pip install openai`, `OPENAI_API_KEY`, and e.g.
      python eval_runner.py --provider openai --model gpt-5.4
    Put secrets in `practice/.env` (see `.env.example`); loaded automatically if python-dotenv is installed.
    By default `.env` overrides a key already exported in the shell (`--preserve-shell-env` changes that).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama_host as oh
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

PRACTICE_DIR = Path(__file__).resolve().parent
SCRATCHPAD_PATH = PRACTICE_DIR / "memory-mcp" / "scratchpad.json"
EVAL_RUNS_DIR = PRACTICE_DIR / "eval_runs"


def clear_scratchpad() -> None:
    """Reset MCP on-disk memory so each case starts isolated."""
    try:
        SCRATCHPAD_PATH.write_text("{}", encoding="utf-8")
    except OSError:
        if SCRATCHPAD_PATH.exists():
            raise
        SCRATCHPAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCRATCHPAD_PATH.write_text("{}", encoding="utf-8")


def read_scratchpad() -> dict[str, Any]:
    if not SCRATCHPAD_PATH.exists():
        return {}
    try:
        raw = SCRATCHPAD_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _flatten_value_for_search(val: Any) -> str:
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except TypeError:
        return str(val)


def _value_as_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    return [_flatten_value_for_search(val)]


def _normalize_for_compare(val: Any) -> Any:
    """Loosen Ollama string JSON vs MCP-native types."""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s.strip('"').strip("'")
        if s in ("true", "false", "null"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return val


def _canonicalize_mcp_value(val: Any, depth: int = 0) -> Any:
    """
    Unify host trace `store` arguments with scratchpad values for sync checks.

    Ollama often passes value as a JSON *string* (e.g. '[\"a\"]' or '{\"k\":1}');
    MCP writes real list/dict. Recursively parse strings until stable — no embeddings needed.
    """
    if depth > 10:
        return val
    if isinstance(val, dict):
        return {str(k): _canonicalize_mcp_value(v, depth + 1) for k, v in val.items()}
    if isinstance(val, list):
        return [_canonicalize_mcp_value(x, depth + 1) for x in val]
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return s
    if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
        try:
            return _canonicalize_mcp_value(json.loads(s), depth + 1)
        except json.JSONDecodeError:
            return s[1:-1] if len(s) >= 2 else s
    if s[0] in "{[":
        try:
            return _canonicalize_mcp_value(json.loads(s), depth + 1)
        except json.JSONDecodeError:
            return val
    return val


def values_equivalent(disk_val: Any, arg_val: Any) -> bool:
    a = _canonicalize_mcp_value(_normalize_for_compare(disk_val))
    b = _canonicalize_mcp_value(_normalize_for_compare(arg_val))
    if a == b:
        return True
    try:
        return json.loads(json.dumps(a, sort_keys=True)) == json.loads(json.dumps(b, sort_keys=True))
    except (TypeError, json.JSONDecodeError):
        return _flatten_value_for_search(a) == _flatten_value_for_compare(b)


def _flatten_value_for_compare(val: Any) -> str:
    return _flatten_value_for_search(_normalize_for_compare(val))


def _is_plausible_taxonomy_key(key: str) -> bool:
    """When relaxing key_any_of, still reject empty or sentence-like keys."""
    if not key or len(key) > 80 or "\n" in key:
        return False
    if "." in key or " " in key:
        return False
    return True


def store_matches_constraints(
    args: dict[str, Any],
    constraints: dict[str, Any],
    *,
    relax_key_any_of: bool = False,
) -> bool:
    key = str(args.get("key", "")).strip()
    val = args.get("value")

    allowed_keys = constraints.get("key_any_of")
    has_value_rubric = bool(constraints.get("value_substrings") or constraints.get("value_list_contains"))
    skip_key_check = (
        relax_key_any_of
        and allowed_keys is not None
        and has_value_rubric
        and key not in allowed_keys
        and _is_plausible_taxonomy_key(key)
    )
    if allowed_keys is not None and key not in allowed_keys and not skip_key_check:
        return False

    subs = constraints.get("value_substrings")
    if subs:
        hay = _flatten_value_for_search(val).lower()
        if not all(s.lower() in hay for s in subs):
            return False

    need_list = constraints.get("value_list_contains")
    if need_list:
        if isinstance(val, str):
            hay = val.lower()
            for needle in need_list:
                if needle.lower() not in hay:
                    return False
        else:
            parts = [x.lower() for x in _value_as_list(val)]
            joined = " ".join(parts)
            for needle in need_list:
                n = needle.lower()
                if not any(n in p or p in n for p in parts) and n not in joined:
                    return False

    return True


def summarize_tools(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "(none)"
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        if name == "store" and isinstance(args, dict):
            parts.append(f"store({args.get('key')!r}, {args.get('value')!r})")
        else:
            parts.append(f"{name}({json.dumps(args, ensure_ascii=False)})")
    return "; ".join(parts)


def _store_call_succeeded(result_preview: str) -> bool:
    prev = (result_preview or "").strip()
    if not prev:
        return False
    if prev.lower().startswith("error") or "error:" in prev.lower()[:80]:
        return False
    return prev.startswith("Stored:")


def verify_mcp_disk_vs_host_trace(
    tool_calls: list[dict[str, Any]],
    scratchpad: dict[str, Any],
) -> tuple[str, str]:
    """
    Compare memory-mcp/scratchpad.json after the turn to successful `store` tool results
    in the host trace (result_preview from MCP). Returns (status_code, one_line_detail).
    """
    last_write: dict[str, Any] = {}
    for tc in tool_calls:
        if tc.get("name") != "store":
            continue
        args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        key = str(args.get("key", "")).strip()
        if not key:
            continue
        prev = tc.get("result_preview") or ""
        if _store_call_succeeded(prev):
            last_write[key] = args.get("value")

    if not last_write:
        if scratchpad:
            return (
                "DISK_UNEXPLAINED",
                f"scratchpad has keys {list(scratchpad.keys())!r} but no successful store() in host trace",
            )
        return "SYNC_OK", "no successful store; scratchpad empty (matches MCP)"

    disk_keys = set(scratchpad.keys())
    write_keys = set(last_write.keys())
    if disk_keys != write_keys:
        return (
            "DISK_KEY_MISMATCH",
            f"trace keys {sorted(write_keys)} vs disk keys {sorted(disk_keys)}",
        )

    for k, want in last_write.items():
        got = scratchpad.get(k)
        if not values_equivalent(got, want):
            return (
                "DISK_VALUE_MISMATCH",
                f"key {k!r}: disk={got!r} vs trace_args={want!r}",
            )

    return "SYNC_OK", f"scratchpad matches {len(last_write)} successful store(s)"


def check_case(
    turn: dict[str, Any],
    case: dict[str, Any],
    *,
    relax_key_any_of: bool = False,
) -> tuple[bool, str]:
    phase = turn.get("response", {}).get("phase1_decision")
    expected_phase = case["expected_router"]
    if phase != expected_phase:
        return False, f"router want {expected_phase!r} got {phase!r}"

    tool_calls = turn.get("response", {}).get("tool_calls") or []
    store_calls = [
        tc for tc in tool_calls
        if tc.get("name") == "store" and isinstance(tc.get("arguments"), dict)
    ]

    if case.get("accept_store_or_delete"):
        cfg = case["accept_store_or_delete"]
        store_constraints = cfg.get("store") or {}
        del_keys = set(cfg.get("delete_key_any_of") or [])
        delete_calls = [
            tc for tc in tool_calls
            if tc.get("name") == "delete" and isinstance(tc.get("arguments"), dict)
        ]
        store_ok = bool(store_constraints) and any(
            store_matches_constraints(
                sc["arguments"],
                store_constraints,
                relax_key_any_of=relax_key_any_of,
            )
            for sc in store_calls
        )
        delete_ok = False
        for dc in delete_calls:
            args = dc.get("arguments") or {}
            k = str(args.get("key", "")).strip()
            if k in del_keys:
                delete_ok = True
                break
        if not store_ok and not delete_ok:
            return False, (
                f"expected store matching {store_constraints!r} or delete(key in {sorted(del_keys)!r}); "
                f"got {summarize_tools(tool_calls)}"
            )
    elif case.get("expect_store"):
        constraints = case.get("store") or {}
        if not store_calls:
            return False, "expected at least one store() call, got none"
        if not any(
            store_matches_constraints(
                sc["arguments"],
                constraints,
                relax_key_any_of=relax_key_any_of,
            )
            for sc in store_calls
        ):
            return False, f"no store() matched constraints {constraints!r}; calls={summarize_tools(tool_calls)}"
    else:
        if store_calls:
            return False, f"expected no store(), got {len(store_calls)} store call(s)"

    extra_forbid = case.get("forbid_tools") or []
    names = [tc.get("name") for tc in tool_calls]
    for ft in extra_forbid:
        if ft in names:
            return False, f"forbidden tool {ft!r} was called"

    return True, "ok"


# Gold matrix: ID, Category, query, expected router, expected persistence (see store / accept_store_or_delete).
EVAL_CASES: list[dict[str, Any]] = [
    # --- Casual (router NO, no tools) ---
    {
        "id": "C1",
        "category": "Casual",
        "query": "Good morning! How are you?",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "C2",
        "category": "Casual",
        "query": "What's the weather like in Tucson?",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "C3",
        "category": "Casual",
        "query": "Tell me a joke about math.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "C4",
        "category": "Casual",
        "query": "Can you help me with a Python script?",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "C5",
        "category": "Casual",
        "query": "I'm feeling a bit tired today.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (transient state)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "C6",
        "category": "Casual",
        "query": "What do you think of the desert?",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None",
        "expected_router": "NO",
        "expect_store": False,
    },
    # --- Storage (router YES, store) ---
    {
        "id": "S1",
        "category": "Storage",
        "query": "I moved to Tucson in 2022.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'location: "Tucson"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["geographic_location", "location", "other_personal"],
            "value_substrings": ["Tucson"],
        },
    },
    {
        "id": "S2",
        "category": "Storage",
        "query": "My birthday is on October 15th.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'birthday: "October 15th"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["birthday", "date_of_birth", "other_personal"],
            "value_substrings": ["October", "15"],
        },
    },
    {
        "id": "S3",
        "category": "Storage",
        "query": "I am a PhD student at UArizona.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'occupation: "PhD Student"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": [
                "current_role",
                "education_level",
                "employer_or_org",
                "academic_field",
                "occupation",
                "other_personal",
            ],
            "value_substrings": ["PhD", "student"],
        },
    },
    {
        "id": "S4",
        "category": "Storage",
        "query": "I grew up in Israel.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'origin: "Israel"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["geographic_location", "origin", "location", "other_personal"],
            "value_substrings": ["Israel"],
        },
    },
    {
        "id": "S5",
        "category": "Storage",
        "query": "My favorite color is forest green.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'fav_color: "Forest Green"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": [
                "fav_color",
                "favorite_color",
                "personal_interests",
                "other_personal",
            ],
            "value_substrings": ["green"],
        },
    },
    {
        "id": "S6",
        "category": "Storage",
        "query": "I just started a project called Lynnapse.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'active_project: "Lynnapse"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["active_project", "current_role", "other_personal"],
            "value_substrings": ["Lynnapse"],
        },
    },
    # --- Update (router YES, store / upsert) ---
    {
        "id": "U1",
        "category": "Update",
        "query": "Actually, I moved to Phoenix now.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'location: "Phoenix" (upsert)',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["geographic_location", "location", "other_personal"],
            "value_substrings": ["Phoenix"],
        },
    },
    {
        "id": "U2",
        "category": "Update",
        "query": "I'm no longer a student; I'm a Prof.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'occupation: "Professor"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["current_role", "education_level", "occupation", "other_personal"],
            "value_substrings": ["Prof"],
        },
    },
    {
        "id": "U3",
        "category": "Update",
        "query": "Change my favorite color to blue.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'fav_color: "Blue"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["fav_color", "favorite_color", "personal_interests", "other_personal"],
            "value_substrings": ["blue"],
        },
    },
    {
        "id": "U4",
        "category": "Update",
        "query": "I graduated! I'm an Alumnus now.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'status: "Alumnus"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["education_level", "status", "current_role", "other_personal"],
            "value_substrings": ["Alumnus"],
        },
    },
    {
        "id": "U5",
        "category": "Update",
        "query": "My project Lynnapse is now 'Synapse'.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'active_project: "Synapse"',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["active_project", "other_personal"],
            "value_substrings": ["Synapse"],
        },
    },
    {
        "id": "U6",
        "category": "Update",
        "query": "I don't live in Israel anymore.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'origin: "Israel" (or delete)',
        "expected_router": "YES",
        "expect_store": False,
        "accept_store_or_delete": {
            "store": {
                "key_any_of": ["geographic_location", "origin", "location", "other_personal"],
                "value_substrings": ["Israel"],
            },
            "delete_key_any_of": [
                "origin",
                "geographic_location",
                "location",
                "other_personal",
            ],
        },
    },
    # --- Lists (router YES, store list or string containing items) ---
    {
        "id": "L1",
        "category": "Lists",
        "query": "I like hiking and rock climbing.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'hobbies: ["hiking", "climbing"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["personal_interests", "hobbies", "other_personal"],
            "value_list_contains": ["hiking", "climbing"],
        },
    },
    {
        "id": "L2",
        "category": "Lists",
        "query": "I know Python, Rust, and C++.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'tech_stack: ["Python", "Rust", "C++"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["technical_skills", "tech_stack", "other_personal"],
            "value_list_contains": ["Python", "Rust", "C++"],
        },
    },
    {
        "id": "L3",
        "category": "Lists",
        "query": "Add swimming to my hobbies.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'hobbies: [..., "swimming"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["personal_interests", "hobbies", "other_personal"],
            "value_list_contains": ["swimming"],
        },
    },
    {
        "id": "L4",
        "category": "Lists",
        "query": "I'm allergic to peanuts and shellfish.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'allergies: ["peanuts", "shellfish"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["allergies", "health", "other_personal"],
            "value_list_contains": ["peanuts", "shellfish"],
        },
    },
    {
        "id": "L5",
        "category": "Lists",
        "query": "I play the guitar and the piano.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'instruments: ["guitar", "piano"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["instruments", "personal_interests", "hobbies", "other_personal"],
            "value_list_contains": ["guitar", "piano"],
        },
    },
    {
        "id": "L6",
        "category": "Lists",
        "query": "I am learning French and Spanish.",
        "expected_logic": "Router: YES",
        "expected_tool_note": 'languages: ["French", "Spanish"]',
        "expected_router": "YES",
        "expect_store": True,
        "store": {
            "key_any_of": ["languages", "technical_skills", "other_personal"],
            "value_list_contains": ["French", "Spanish"],
        },
    },
    # --- Ambiguous (router NO, no tools) ---
    {
        "id": "A1",
        "category": "Ambiguous",
        "query": "The car is blue.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (not about user)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "A2",
        "category": "Ambiguous",
        "query": "Tucson is very hot in the summer.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (general fact)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "A3",
        "category": "Ambiguous",
        "query": "Someone told me that math is hard.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (not user's opinion)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "A4",
        "category": "Ambiguous",
        "query": "Do you know where Israel is?",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (question ≠ fact)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "A5",
        "category": "Ambiguous",
        "query": "My friend likes to eat hamburgers.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (third party)",
        "expected_router": "NO",
        "expect_store": False,
    },
    {
        "id": "A6",
        "category": "Ambiguous",
        "query": "I'll tell you my name later.",
        "expected_logic": "Router: NO",
        "expected_tool_note": "None (placeholder)",
        "expected_router": "NO",
        "expect_store": False,
    },
]


def _format_expected(case: dict[str, Any]) -> str:
    r = case["expected_router"]
    if case.get("accept_store_or_delete"):
        return f"{r} | store or delete"
    if not case.get("expect_store"):
        return f"{r} | no tools"
    c = case.get("store") or {}
    parts = [f"{r} | store"]
    if c.get("key_any_of"):
        parts.append(f"keys={c['key_any_of']}")
    if c.get("value_substrings"):
        parts.append(f"∋{c['value_substrings']}")
    if c.get("value_list_contains"):
        parts.append(f"list∋{c['value_list_contains']}")
    return " ".join(parts)


def _model_slug(model: str) -> str:
    s = model.strip().lower().replace(".", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s.strip("-")[:56] or "model"


def _case_folder_name(case: dict[str, Any]) -> str:
    raw = f"{case['id']}_{case.get('category', 'case')}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")


def _trial_history_path(cases_root: Path, case: dict[str, Any], trial: int) -> Path:
    return cases_root / _case_folder_name(case) / f"trial_{trial:02d}.json"


def _write_run_readme(
    path: Path,
    *,
    run_dir_name: str,
    model: str,
    trials_per_case: int,
    baseline_cases: int,
    total_runs: int,
    passed: int,
    failed: int,
) -> None:
    pct = 100.0 * passed / total_runs if total_runs else 0.0
    text = f"""This folder is one complete eval run from eval_runner.py.

Run folder name (what you see in eval_runs/)
  {run_dir_name}

Decoded:
  • UTC time stamp + model + trials per baseline scenario (each of the {baseline_cases} types
    was run {trials_per_case} time(s) → {total_runs} Ollama turns total).

Files here
  README.txt              ← this guide
  summary.txt             ← human-readable full report + tables
  meta.json               ← same run in JSON (timestamps, aggregates)
  mcp_verification.tsv    ← open in Excel/Sheets; one row per trial

  cases/                  ← raw traces, grouped by scenario
    01_Casual/            ← baseline id + category from eval_runner.py
      trial_01.json       ← one user query + router + tools + reply (like history_log)
      trial_02.json
    02_Storage/
      …

Result for this run: {passed}/{total_runs} passed ({pct:.1f}%), {failed} failed.
Model: {model}
"""
    path.write_text(text, encoding="utf-8")


@dataclass
class CaseResult:
    trial: int
    status: str
    case: dict[str, Any]
    turn: dict[str, Any] = field(default_factory=dict)
    scratchpad_after: dict[str, Any] = field(default_factory=dict)
    mcp_sync_status: str = ""
    mcp_sync_detail: str = ""
    eval_reason: str = ""
    exception: str = ""
    stdout_cap: str = ""
    stderr_cap: str = ""


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_mcp_table_tsv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "id",
                "category",
                "trial",
                "eval_status",
                "router",
                "host_tools",
                "scratchpad_json",
                "mcp_vs_host",
                "mcp_detail",
                "eval_notes",
            ],
        )
        for r in results:
            turn = r.turn or {}
            router = (turn.get("response") or {}).get("phase1_decision", "")
            tools = (turn.get("response") or {}).get("tool_calls") or []
            w.writerow(
                [
                    r.case.get("id", ""),
                    r.case.get("category", ""),
                    r.trial,
                    r.status,
                    router,
                    summarize_tools(tools),
                    json.dumps(r.scratchpad_after, ensure_ascii=False),
                    r.mcp_sync_status,
                    r.mcp_sync_detail,
                    r.eval_reason if r.status == "FAIL" else (r.exception or ""),
                ],
            )


def _aggregate_by_case_id(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """case_id -> (passes, total)."""
    acc: dict[str, list[int]] = {}
    for r in results:
        cid = str(r.case.get("id", ""))
        if cid not in acc:
            acc[cid] = [0, 0]
        acc[cid][1] += 1
        if r.status == "PASS":
            acc[cid][0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(acc.items())}


def _write_summary_txt(
    path: Path,
    *,
    model: str,
    run_dir: Path,
    results: list[CaseResult],
    passed: int,
    total: int,
    trials_per_case: int,
) -> None:
    by_case = _aggregate_by_case_id(results)
    lines = [
        "Eval run summary",
        "================",
        "Traces live under cases/<id>_<Category>/trial_NN.json — see README.txt in this folder.",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Model: {model}",
        f"Artifacts: {run_dir.resolve()}",
        f"Trials per case type: {trials_per_case}  |  Baseline case types: {len(by_case)}  |  Total runs: {total}",
        "",
        f"Result: {passed}/{total} eval assertions passed ({100.0 * passed / total if total else 0:.1f}%)",
        "",
        "Aggregate by case type (id)",
        "-----------------------------",
    ]
    for cid, (p, t) in by_case.items():
        cat = next((r.case.get("category", "") for r in results if str(r.case.get("id")) == cid), "")
        lines.append(f"  [{cid}] {cat}: {p}/{t} pass ({100.0 * p / t if t else 0:.1f}%)")
    lines.extend(["", "Per trial", "---------"])
    for r in results:
        q = r.case.get("query", "")
        lines.append(
            f"[{r.case.get('id')} trial {r.trial:02d}] {r.case.get('category', '')} — {r.status} | "
            f"MCP: {r.mcp_sync_status}",
        )
        lines.append(f"    query: {q}")
        if r.exception:
            lines.append(f"    exception: {r.exception}")
        elif r.status == "FAIL" and r.eval_reason:
            lines.append(f"    eval: {r.eval_reason}")
        lines.append(f"    scratchpad_after: {json.dumps(r.scratchpad_after, ensure_ascii=False)}")
        lines.append("")

    lines.append("Console table (all trials)")
    lines.append("-" * 72)
    col_w = (4, 6, 6, 10, 28, 22, 18, 20)
    hdr = (
        f"{'TR':<{col_w[0]}} | {'ST':<{col_w[1]}} | {'ID':<{col_w[2]}} | {'MCP_SYNC':<{col_w[3]}} | "
        f"{'QUERY':<{col_w[4]}} | {'EXPECTED':<{col_w[5]}} | {'HOST_TOOLS':<{col_w[6]}} | "
        f"{'SCRATCH':<{col_w[7]}}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        q = (r.case.get("query", "") or "")[: col_w[4] - 2]
        if len(r.case.get("query", "")) > col_w[4] - 2:
            q += "…"
        scratch_one = json.dumps(r.scratchpad_after, ensure_ascii=False)
        if len(scratch_one) > col_w[7] - 1:
            scratch_one = scratch_one[: col_w[7] - 2] + "…"
        lines.append(
            f"{r.trial:<{col_w[0]}} | {r.status[:4]:<{col_w[1]}} | {str(r.case.get('id','')):<{col_w[2]}} | "
            f"{r.mcp_sync_status:<{col_w[3]}} | {q:<{col_w[4]}} | "
            f"{_format_expected(r.case):<{col_w[5]}} | "
            f"{summarize_tools((r.turn.get('response') or {}).get('tool_calls') or []):<{col_w[6]}} | "
            f"{scratch_one:<{col_w[7]}}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_run_dir(
    output_dir: str | None,
    *,
    provider: str,
    model: str,
    trials_per_case: int,
) -> Path:
    if output_dir:
        p = Path(output_dir)
        if not p.is_absolute():
            p = PRACTICE_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    slug = _model_slug(model)
    prov = (provider or "ollama").strip().lower()
    if prov == "ollama":
        label = f"{stamp}Z__model-{slug}__{trials_per_case}trials-each"
    else:
        pslug = re.sub(r"[^a-z0-9._-]+", "-", prov).strip("-")[:24] or "api"
        label = f"{stamp}Z__{pslug}__model-{slug}__{trials_per_case}trials-each"
    run = EVAL_RUNS_DIR / label
    run.mkdir(parents=True, exist_ok=True)
    return run


async def run_eval(
    model: str,
    verbose: bool,
    output_dir: str | None,
    *,
    provider: str = "ollama",
    openai_base_url: str | None = None,
    trials_per_case: int,
    relax_store_keys: bool = False,
) -> int:
    oh.TRACE_LOG_PATH = None
    if not verbose:
        oh.QUIET_UI = True

    if not oh.MEMORY_SERVER.exists():
        print(f"Error: Memory MCP server not found at {oh.MEMORY_SERVER}", file=sys.stderr)
        return 1

    manifest = oh.load_manifest()
    manifest_raw = oh.load_manifest_raw()
    skills = manifest["skills"]
    primary = next((s for s in skills if isinstance(s, dict)), None)
    if not primary:
        print("Error: no skill entries in manifest.", file=sys.stderr)
        return 1

    if trials_per_case < 1:
        print("Error: --trials-per-case must be >= 1", file=sys.stderr)
        return 1

    prov = (provider or "ollama").strip().lower()
    if prov == "openai":
        key = oh.sanitize_api_key(os.environ.get("OPENAI_API_KEY"))
        if not key:
            print("Error: OPENAI_API_KEY is required for --provider openai", file=sys.stderr)
            return 1
        base_early = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        pre = await oh.openai_connection_preflight(key, base_early)
        if pre:
            print(pre, file=sys.stderr)
            low = pre.lower()
            if "401" in pre or "invalid_api_key" in low or "incorrect api key" in low:
                print(
                    "→ Put a valid key in practice/.env as OPENAI_API_KEY=sk-... "
                    "(create one at https://platform.openai.com/account/api-keys ).",
                    file=sys.stderr,
                )
                print(
                    "→ By default, values from practice/.env override a key exported in the shell. "
                    "If you rely on the shell only, run with --preserve-shell-env and --no-env-file, "
                    "or run: unset OPENAI_API_KEY",
                    file=sys.stderr,
                )
            return 1

    run_dir = _prepare_run_dir(
        output_dir,
        provider=prov,
        model=model,
        trials_per_case=trials_per_case,
    )
    cases_dir = run_dir / "cases"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(oh.MEMORY_SERVER)],
        env=None,
    )

    results: list[CaseResult] = []

    meta = {
        "schema": "eval_run_meta_v1",
        "provider": prov,
        "model": model,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "practice_dir": str(PRACTICE_DIR),
        "baseline_case_count": len(EVAL_CASES),
        "trials_per_case": trials_per_case,
        "total_runs_planned": len(EVAL_CASES) * trials_per_case,
        "run_folder": str(run_dir.name),
        "relax_store_keys": relax_store_keys,
        "artifacts_layout": {
            "README.txt": "Explains this folder",
            "summary.txt": "Full text report",
            "meta.json": "Run metadata JSON",
            "mcp_verification.tsv": "Spreadsheet table",
            "cases/<id>_<Category>/trial_NN.json": "One trace JSON per trial",
        },
    }
    _write_json(run_dir / "meta.json", meta)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = listed.tools
            ollama_tools = [oh.mcp_tool_to_ollama(t) for t in tools]
            known_names = {t.name for t in tools}
            oh.validate_manifest_against_mcp(manifest, known_names)

            if prov == "openai":
                from llm_bridge import OpenAIChatAdapter

                base = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
                client = OpenAIChatAdapter(
                    api_key=oh.sanitize_api_key(os.environ.get("OPENAI_API_KEY")),
                    base_url=base,
                )
            else:
                client = AsyncClient()

            total_runs = len(EVAL_CASES) * trials_per_case
            rk = "  |  store_key_rubric=relaxed" if relax_store_keys else ""
            print(
                f"Provider: {prov}  |  Model: {model}  |  baseline cases: {len(EVAL_CASES)}  |  "
                f"trials/case: {trials_per_case}  |  total: {total_runs}{rk}  |  out: {run_dir}",
                file=sys.stderr,
            )

            for case in EVAL_CASES:
                for trial in range(1, trials_per_case + 1):
                    clear_scratchpad()
                    history: list[dict[str, Any]] = []

                    buf_out = io.StringIO()
                    buf_err = io.StringIO()
                    turn: dict[str, Any] = {}
                    err_exc: BaseException | None = None
                    try:
                        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                            turn = await oh.run_agent_turn(
                                client,
                                model,
                                case["query"],
                                history,
                                session,
                                ollama_tools,
                                known_names,
                                manifest_raw,
                                primary,
                            )
                    except BaseException as e:
                        err_exc = e

                    stdout_cap = buf_out.getvalue()
                    stderr_cap = buf_err.getvalue()
                    scratch = read_scratchpad()
                    tool_calls = (turn.get("response") or {}).get("tool_calls") or [] if turn else []
                    mcp_st, mcp_detail = verify_mcp_disk_vs_host_trace(tool_calls, scratch)

                    if err_exc is not None:
                        cr = CaseResult(
                            trial=trial,
                            status="FAIL",
                            case=case,
                            turn={},
                            scratchpad_after=scratch,
                            mcp_sync_status=mcp_st,
                            mcp_sync_detail=mcp_detail,
                            eval_reason="",
                            exception=repr(err_exc),
                            stdout_cap=stdout_cap,
                            stderr_cap=stderr_cap,
                        )
                        results.append(cr)
                        payload = {
                            "schema": "eval_case_history_v1",
                            "trial": trial,
                            "case": case,
                            "turns": [],
                            "error": repr(err_exc),
                            "ui_capture": {"stdout": stdout_cap, "stderr": stderr_cap},
                            "scratchpad_after": scratch,
                            "mcp_verification": {
                                "status": mcp_st,
                                "detail": mcp_detail,
                                "note": "Host trace vs memory-mcp/scratchpad.json after this case (same process).",
                            },
                        }
                        _write_json(_trial_history_path(cases_dir, case, trial), payload)
                        if verbose:
                            print(stderr_cap, file=sys.stderr)
                        continue

                    ok, reason = check_case(
                        turn,
                        case,
                        relax_key_any_of=relax_store_keys,
                    )
                    status = "PASS" if ok else "FAIL"
                    cr = CaseResult(
                        trial=trial,
                        status=status,
                        case=case,
                        turn=turn,
                        scratchpad_after=scratch,
                        mcp_sync_status=mcp_st,
                        mcp_sync_detail=mcp_detail,
                        eval_reason="" if ok else reason,
                        exception="",
                        stdout_cap=stdout_cap,
                        stderr_cap=stderr_cap,
                    )
                    results.append(cr)

                    payload = {
                        "schema": "eval_case_history_v1",
                        "trial": trial,
                        "case": case,
                        "turns": [turn],
                        "ui_capture": {"stdout": stdout_cap, "stderr": stderr_cap},
                        "scratchpad_after": scratch,
                        "mcp_verification": {
                            "status": mcp_st,
                            "detail": mcp_detail,
                            "note": "Compares successful store() results in turn.response.tool_calls "
                            "to scratchpad.json on disk (MCP is the source of truth for persistence).",
                        },
                    }
                    _write_json(_trial_history_path(cases_dir, case, trial), payload)

                    if verbose or not ok:
                        print(stderr_cap, end="", file=sys.stderr)
                        print(stdout_cap, end="", file=sys.stderr)
                        if not ok:
                            print(f"  [{case['id']} trial {trial:02d}] {reason}", file=sys.stderr)

    failures = sum(1 for r in results if r.status == "FAIL")
    passed = len(results) - failures

    _write_mcp_table_tsv(run_dir / "mcp_verification.tsv", results)
    _write_summary_txt(
        run_dir / "summary.txt",
        model=model,
        run_dir=run_dir,
        results=results,
        passed=passed,
        total=len(results),
        trials_per_case=trials_per_case,
    )

    by_case = _aggregate_by_case_id(results)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["passed"] = passed
    meta["failed"] = failures
    meta["aggregate_by_case_id"] = {k: {"pass": p, "total": t} for k, (p, t) in by_case.items()}
    _write_json(run_dir / "meta.json", meta)

    _write_run_readme(
        run_dir / "README.txt",
        run_dir_name=run_dir.name,
        model=model,
        trials_per_case=trials_per_case,
        baseline_cases=len(EVAL_CASES),
        total_runs=len(results),
        passed=passed,
        failed=failures,
    )

    col_w = (4, 6, 6, 10, 28, 24, 36)
    hdr = (
        f"{'TR':<{col_w[0]}} | {'ST':<{col_w[1]}} | {'ID':<{col_w[2]}} | {'MCP_SYNC':<{col_w[3]}} | "
        f"{'QUERY':<{col_w[4]}} | {'EXPECTED':<{col_w[5]}} | {'HOST_TOOLS':<{col_w[6]}}"
    )
    print()
    print("By case type:", file=sys.stderr)
    for cid, (p, t) in by_case.items():
        cat = next((r.case.get("category", "") for r in results if str(r.case.get("id")) == cid), "")
        print(f"  [{cid}] {cat}: {p}/{t}", file=sys.stderr)
    print(file=sys.stderr)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        q = r.case.get("query", "")
        qdisp = q[: col_w[4] - 2] + ("…" if len(q) > col_w[4] - 2 else "")
        tools = summarize_tools((r.turn.get("response") or {}).get("tool_calls") or [])
        print(
            f"{r.trial:<{col_w[0]}} | {r.status[:4]:<{col_w[1]}} | {str(r.case.get('id','')):<{col_w[2]}} | "
            f"{r.mcp_sync_status:<{col_w[3]}} | {qdisp:<{col_w[4]}} | "
            f"{_format_expected(r.case):<{col_w[5]}} | {tools:<{col_w[6]}}"
        )
    print()
    print(f"Result: {passed}/{len(results)} passed ({100.0 * passed / len(results) if results else 0:.1f}%)")
    print(f"Wrote: {run_dir / 'README.txt'}")
    print(f"Wrote: {run_dir / 'summary.txt'}")
    print(f"Wrote: {run_dir / 'mcp_verification.tsv'}")
    print(f"Wrote: {cases_dir}/<scenario>/trial_NN.json")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval router + tools for ollama_host.py")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        default="ollama",
        help="LLM backend (default: ollama). OpenAI requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL (default: official OpenAI).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id for the provider (default: llama3.2 for ollama, gpt-5.4 for openai).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print host stderr/stdout per case")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for this run (default: eval_runs/<UTC-date>_<time>Z__model-<name>__<n>trials-each). "
            "Inside it you always get README.txt, summary.txt, meta.json, mcp_verification.tsv, cases/…"
        ),
    )
    parser.add_argument(
        "--trials-per-case",
        type=int,
        default=1,
        help=(
            "Repeat each EVAL_CASES row this many times (default: 1). "
            "There are 30 matrix rows; e.g. --trials-per-case 6 → 30×6 = 180 Ollama turns."
        ),
    )
    parser.add_argument(
        "--relax-store-keys",
        action="store_true",
        help=(
            "If a case has value_substrings or value_list_contains, allow any plausible taxonomy key "
            "when the value matches (ignore key_any_of mismatches like birthday_date vs birthday)."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help=(
            "Load environment variables from this dotenv file before running (default: practice/.env). "
            "Relative paths are resolved under the practice/ directory."
        ),
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Do not load a .env file.",
    )
    parser.add_argument(
        "--preserve-shell-env",
        action="store_true",
        help=(
            "When loading .env, do not override variables already set in the shell "
            "(default: .env wins so practice/.env is the source of truth)."
        ),
    )
    args = parser.parse_args()
    if not args.no_env_file:
        oh.load_dotenv_file(Path(args.env_file), override=not args.preserve_shell_env)
    prov = args.provider.strip().lower()
    if args.model is None:
        model = "gpt-5.4" if prov == "openai" else "llama3.2"
    else:
        model = args.model
    try:
        code = asyncio.run(
            run_eval(
                model,
                args.verbose,
                args.output_dir,
                provider=prov,
                openai_base_url=args.openai_base_url,
                trials_per_case=args.trials_per_case,
                relax_store_keys=args.relax_store_keys,
            )
        )
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
