#!/usr/bin/env python3
"""
Load clinical state-machine nodes from `nodes/<node_id>.md`.

Each node file starts with optional `---` front matter:

  kind: question | result
  patient: "Text shown in the terminal"

Optional front-matter keys:

  ehr_auto_goto: LRQ_Step_2
  ehr_auto_when: appendectomy_in_pmh

If `ehr_auto_when` is true for the session EHR, the host skips printing this node's `patient:` line
and jumps to `ehr_auto_goto` before any LLM routing (chart-first). Predicate names are defined in
`ehr_reader.EHR_AUTO_WHEN_PREDICATES` / `ehr_reader.ehr_predicate_holds`.

The remainder is markdown (typically a `# Logic:` section with `GOTO node_id **Id**` lines).
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


def split_node_document(text: str) -> tuple[dict[str, str], str]:
    """
    Split `---` YAML-style front matter from the rest of the file.
    Returns (metadata dict with lowercased keys, body markdown).
    If there is no leading `---` block, returns ({}, full text).
    """
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        raw = lines[i].strip()
        if raw and not raw.startswith("#") and ":" in raw:
            key, _, val = raw.partition(":")
            k = key.strip().lower()
            meta[k] = _strip_wrapping_quotes(val.strip())
        i += 1

    if i >= len(lines):
        return {}, text

    body = "\n".join(lines[i + 1 :])
    return meta, body


def node_logic_body(md: str) -> str:
    """Body after front matter (or whole file if no front matter). Used for GOTO parsing."""
    meta, body = split_node_document(md)
    return body if meta else md


def meta_ehr_auto_redirect(meta: dict[str, str]) -> tuple[str | None, str | None]:
    """Parse `ehr_auto_goto` / `ehr_auto_when` for host pre-display chart pruning."""
    goto = (meta.get("ehr_auto_goto") or "").strip()
    when = (meta.get("ehr_auto_when") or "").strip()
    if not goto or not when:
        return None, None
    return goto, when


def parse_node_display(md: str) -> tuple[str, str] | None:
    """
    Patient-visible line from node front matter:
      kind: question | result
      patient: "text shown in the terminal"
    Returns (\"Question\"|\"Result\", text) or None if invalid.
    """
    meta, _ = split_node_document(md)
    kind = (meta.get("kind") or "").lower().strip()
    patient = (meta.get("patient") or "").strip()
    if kind not in ("question", "result") or not patient:
        return None
    label = "Question" if kind == "question" else "Result"
    return label, patient


def extract_branch_target_node_ids(md: str) -> list[str]:
    """Ordered unique node ids appearing as `GOTO node_id **Id**` in the Logic body."""
    body = node_logic_body(md)
    seen: set[str] = set()
    out: list[str] = []
    for m in GOTO_NODE_PATTERN.finditer(body):
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
