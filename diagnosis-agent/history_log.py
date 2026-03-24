"""
Append-only Markdown trace of patient lines, LLM context, tool outputs, and host display.

The `history/` directory is wiped at the start of each `python chat.py` process.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_FILE = HISTORY_DIR / "history.md"


def reset_history_dir() -> None:
    """Remove prior `history/` and recreate an empty folder."""
    if HISTORY_DIR.exists():
        shutil.rmtree(HISTORY_DIR)
    HISTORY_DIR.mkdir(parents=True)


def _fence(lang: str, body: str) -> str:
    body = body.replace("\r\n", "\n")
    if "```" in body:
        fence = "````"
        while fence in body:
            fence += "`"
        return f"{fence}{lang}\n{body}\n{fence}\n\n"
    return f"```{lang}\n{body}\n```\n\n"


class SessionHistory:
    """Writes `history/history.md` for one process (may include multiple intakes after /clear)."""

    def __init__(self, *, model: str) -> None:
        self._model = model
        self._display_seq = 0
        self._routing_seq = 0
        HISTORY_FILE.write_text(
            "# Patient / agent interaction history\n\n"
            f"_Auto-generated. Folder reset on each `python chat.py` run._\n\n"
            f"- **Started:** {datetime.now(timezone.utc).isoformat()} (UTC)\n"
            f"- **Model:** `{model}`\n\n"
            "---\n\n",
            encoding="utf-8",
        )

    def _append(self, text: str) -> None:
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def intake(self, chief: str, *, session_restart: bool = False) -> None:
        title = "## Intake (new visit)" if session_restart else "## Intake"
        self._append(
            f"{title}\n\n"
            f"- **Chief complaint (patient):** {json.dumps(chief)}\n\n"
            "---\n\n"
        )

    def ehr_loaded(self, relative_path: str, host_summary: str, full_record_json: str) -> None:
        self._append(
            "## Synthetic EHR (session)\n\n"
            f"- **File:** `{relative_path}`\n"
            "- **Model access:** chart slices **only** via `check_chart` when node Logic requires it "
            "(not pre-loaded into routing prompts).\n"
            "- **Host digest** (deterministic; audit / pre-display redirect only — not sent to the model):\n\n"
            f"{_fence('markdown', host_summary)}"
            "#### Full record (JSON)\n\n"
            f"{_fence('json', full_record_json)}"
            "---\n\n",
        )

    def display_to_patient(self, node_id: str, label: str, text: str) -> None:
        """What the host printed (from node front matter), not from the LLM."""
        self._display_seq += 1
        self._append(
            f"## Display {self._display_seq} — node `{node_id}`\n\n"
            "### Shown to patient (host)\n\n"
            f"- **Kind:** `{label}` (from node `kind:` front matter)\n"
            f"- **Text:** {json.dumps(text)}\n\n"
            "_Printed by Python from `nodes/{node_id}.md`; the LLM does not author this line._\n\n"
            "---\n\n"
        )

    def host_pre_display_ehr_redirect(self, *, from_node: str, to_node: str, when: str) -> None:
        """Logged when the host skips a node's patient line because `ehr_auto_when` matched the chart."""
        self._append(
            "## Host EHR pre-display redirect\n\n"
            f"- **Skipped displaying:** `{from_node}`\n"
            f"- **Matched:** `ehr_auto_when: {when}`\n"
            f"- **Next node (no question shown for skipped node):** `{to_node}`\n\n"
            "_Deterministic host evaluation before `Display`; avoids redundant questions when the chart already implies the branch._\n\n"
            "---\n\n",
        )

    def routing(
        self,
        *,
        patient_line: str,
        system_prompt: str,
        current_node_id: str,
        chief_complaint: str,
        allowed_node_ids: list[str],
        node_markdown_full: str,
        api_called: bool,
        tool_name: str,
        chosen_node_id: str,
        note: str = "",
        assistant_message_json: str | None = None,
        ehr_prefetch_markdown: str | None = None,
        chart_mining_trace: str | None = None,
    ) -> None:
        self._routing_seq += 1
        self._append(
            f"## Routing {self._routing_seq} — after patient reply\n\n"
            f"- **Patient said (`You:`):** {json.dumps(patient_line)}\n\n"
            "### Context sent to the model\n\n"
            "#### System message\n\n"
            f"{_fence('text', system_prompt)}"
            "#### User message (structured)\n\n"
            f"- **`current_node_id`:** `{current_node_id}`\n"
            f"- **Chief complaint (repeated for model):** {json.dumps(chief_complaint)}\n"
            f"- **Allowed `node_id` values (enum):** {json.dumps(allowed_node_ids)}\n\n"
        )
        if ehr_prefetch_markdown:
            self._append(
                "#### EHR — host prefetch for this node (`ehr_prefetch`)\n\n"
                f"{_fence('markdown', ehr_prefetch_markdown)}"
            )
        self._append(
            "#### Full current node file (as sent to the model)\n\n"
            f"{_fence('markdown', node_markdown_full)}"
        )
        if chart_mining_trace:
            self._append(
                "### Chart-mining phase (`check_chart` only)\n\n"
                "_Optional rounds before routing; each round is one Chat Completions call with `tool_choice=auto`._\n\n"
                f"{chart_mining_trace}\n"
            )
        self._append(
            "### Routing phase (`choose_next_node`)\n\n"
            f"- **Chat Completions API called (routing):** {'yes' if api_called else 'no (single branch — skipped)'}\n"
            f"- **Expected tool name (schema):** `{tool_name}`\n"
        )
        if assistant_message_json is not None:
            self._append(
                "- **Exact assistant `message` from API** (`choices[0].message`) — "
                "this is what the model returned to activate tool(s); "
                "`function.arguments` is the raw string from the API (often a JSON string):\n\n"
                f"{_fence('json', assistant_message_json)}"
            )
        else:
            self._append(
                "- **Exact assistant `message` from API:** _not applicable (no Chat Completions call)._\n\n"
            )
        self._append(f"- **Host resolved next `node_id`:** `{chosen_node_id}`\n")
        if note:
            self._append(f"- **Note:** {note}\n")
        self._append(
            "\n_The next **Display** section shows what the host prints after moving to that node._\n\n"
            "---\n\n"
        )

    def error(self, message: str) -> None:
        self._append(f"\n> **Run error (routing not applied):** {message}\n\n")
