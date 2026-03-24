# Diagnosis agent

Lightweight tooling around a terminal chat client and related assets (skills, patient fixtures). This document will grow as the project does.

## Prerequisites

- Python 3.10+ recommended
- An [OpenAI API key](https://platform.openai.com/account/api-keys) with access to your chosen chat model

## Setup

From this directory (`diagnosis-agent/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` (gitignored) from the template:

```bash
cp .env.example .env
```

On macOS, to avoid overwriting an existing `.env`:

```bash
cp -n .env.example .env
```

Edit `.env` and set at least `OPENAI_API_KEY`. Optional variables are documented in `.env.example`.

`chat.py` loads `diagnosis-agent/.env` **before** reading options and uses **`override=True`**, so `OPENAI_*` entries in that file replace the same variables already set in your environment (for example an old `OPENAI_API_KEY` from `~/.zshrc` or the IDE). To rely on the shell instead, omit `OPENAI_API_KEY` from `.env` (or comment that line out) so the process keeps the exported value.

## Running the terminal chat

With the venv activated:

```bash
python chat.py
```

`chat.py` is **host-driven** for patient-visible text: it prints each **`Question:`** / **`Result:`** from `nodes/<node_id>.md` itself (no LLM). After each `You:` answer, routing is **two-phase**: (1) optional **chart-mining** — the model may call **`check_chart(category=…)`** zero or more times (`tool_choice=auto`, capped per turn) to pull slices from the synthetic EHR; (2) **routing** — a forced **`choose_next_node`** tool call whose `node_id` is restricted to an **enum** parsed from `GOTO node_id **SomeId**` lines in the current node’s Logic (no skipping ahead arbitrarily).

- **Synthetic EHR:** `ehr/ehr_patient_1.json` (flat, categorized JSON). The host loads it for **`check_chart`** and for optional **pre-display** redirects. The routing model does **not** get an automatic chart dump: it must call **`check_chart(category=…)`** when node **Logic** says to (**Pre-check** / **IF EHR**). Session start still logs the full JSON plus a deterministic **`host_ehr_summary`** for human audit only (not sent to the model).
- **Pre-display chart prune (host):** optional `ehr_auto_goto` + `ehr_auto_when` in front matter. If `ehr_reader.ehr_predicate_holds(when, ehr)` is true, the host **does not print** that node's `patient:` line and moves to `ehr_auto_goto` first (logged in `history.md` as **Host EHR pre-display redirect**). This is host-only chart access, not a model tool call.

- **Model:** defaults to `gpt-5.1`; override with `OPENAI_MODEL` in `.env` or `python chat.py --model <id>`.
- **Base URL:** optional `OPENAI_BASE_URL` in `.env` for OpenAI-compatible endpoints.

Flow: **chief complaint** → `Patient: …` → protocol questions at `You:` one step at a time → **`Result:`** exits the program. `/clear` (at `You:`) restarts with a new chief complaint from `skills`. `/quit` exits.

Offline check (no API key): `python test_router_logic.py` validates GOTO parsing.

**Interpretability log:** each run wipes `history/` and writes `history/history.md` — synthetic EHR load (audit), chief complaint, each screen the host shows, each patient reply, the full LLM context (system + structured user fields + node file), **chart-mining** rounds (`check_chart`), routing tool JSON, and resolved next node. The folder is gitignored.

## Clinical state machine (`nodes/` + `router.py`)

- **Nodes:** one Markdown file per state: `nodes/<node_id>.md` (e.g. `skills`, `LRQ_Step_1`).
- **Front matter (required):** YAML-style block at the top — `kind: question` or `kind: result`, and `patient: "…"` (the line printed as `Question:` / `Result:`). Optional **`ehr_auto_goto`** / **`ehr_auto_when`** trigger a **host-only** skip of the patient line when the chart matches (see `ehr_reader.EHR_AUTO_WHEN_PREDICATES`). In `# Logic:`, use **Pre-check** lines to tell the model when to call **`check_chart`** before applying **IF EHR** branches. **`GOTO node_id **NextNode**`** lines define the allowed routing enum.
- **Loader:** `router.py` reads a node by id (safe basename only).

```bash
python router.py --list
python router.py skills
python router.py LRQ_Step_1
```

**Cursor / agent prompt:** see `prompts/CURSOR_CLINICAL_STATE_MACHINE.md` for the strict router instructions to paste as a system prompt.

The older `skills/skills.md` note is superseded by **`nodes/skills.md`** as the entry node.

---

*Add sections below (architecture, evaluation, etc.) as the project expands.*
