#!/usr/bin/env python3
"""
Load clinical state-machine nodes from `nodes/<node_id>.md`.

Intended for tooling (e.g. an LLM `execute_node` handler) and quick CLI checks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NODES_DIR = Path(__file__).resolve().parent / "nodes"
# Safe ids only: no path components or weird characters.
NODE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Targets written as: GOTO node_id **Some_Node**
GOTO_NODE_PATTERN = re.compile(
    r"GOTO\s+node_id\s+\*\*([A-Za-z][A-Za-z0-9_]*)\*\*",
    re.IGNORECASE,
)


def _strip_wrapping_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1].strip()
    return s


def parse_node_display(md: str) -> tuple[str, str] | None:
    """
    First patient-visible line from a node: (#) Question: … or (#) Result: …
    Returns (\"Question\"|\"Result\", text) or None.
    """
    patterns: list[tuple[str, str]] = [
        (r"^#\s*Question:\s*(.+)$", "Question"),
        (r"^Question:\s*(.+)$", "Question"),
        (r"^#\s*Result:\s*(.+)$", "Result"),
        (r"^Result:\s*(.+)$", "Result"),
    ]
    for rx, label in patterns:
        m = re.search(rx, md, re.MULTILINE)
        if m:
            return label, _strip_wrapping_quotes(m.group(1))
    return None


def extract_question_from_markdown(md: str) -> str:
    """Backward-compatible: Question text only, or Result text if no Question (terminal nodes)."""
    p = parse_node_display(md)
    return p[1] if p else ""


def extract_branch_target_node_ids(md: str) -> list[str]:
    """Ordered unique node ids appearing as `GOTO node_id **Id**` in the markdown."""
    seen: set[str] = set()
    out: list[str] = []
    for m in GOTO_NODE_PATTERN.finditer(md):
        nid = m.group(1)
        if NODE_ID_PATTERN.fullmatch(nid) and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def allowed_next_node_ids(node_md: str, current_id: str) -> list[str]:
    """
    Branch targets from Logic that have a node file, plus `current_id` for re-asking
    when Logic allows staying on the same node.
    """
    out: list[str] = []
    seen: set[str] = set()
    for nid in extract_branch_target_node_ids(node_md):
        if not (NODES_DIR / f"{nid}.md").is_file():
            continue
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    if current_id not in seen:
        out.append(current_id)
    return out


def load_node(node_id: str) -> str:
    nid = node_id.strip()
    if not NODE_ID_PATTERN.fullmatch(nid):
        raise ValueError(f"Invalid node_id (use letters, digits, underscore): {node_id!r}")
    path = NODES_DIR / f"{nid}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing node file for {nid!r}: {path}")
    return path.read_text(encoding="utf-8")


def list_node_ids() -> list[str]:
    if not NODES_DIR.is_dir():
        return []
    out: list[str] = []
    for p in sorted(NODES_DIR.glob("*.md")):
        out.append(p.stem)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Load or list clinical router nodes.")
    p.add_argument("node_id", nargs="?", help="Node id (filename without .md under nodes/).")
    p.add_argument("--list", action="store_true", help="List available node ids.")
    args = p.parse_args()

    if args.list:
        for nid in list_node_ids():
            print(nid)
        return

    if not args.node_id:
        p.print_help()
        sys.exit(1)

    try:
        print(load_node(args.node_id), end="")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
