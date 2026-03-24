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

`chat.py` is **host-driven**: it prints each **`Question:`** / **`Result:`** from `nodes/<node_id>.md` itself (no LLM). After each `You:` answer, the model makes **one** forced **`choose_next_node`** tool call whose `node_id` is restricted to an **enum** parsed from `GOTO node_id **SomeId**` lines in the current node’s Logic — so it cannot skip ahead through the tree in a single response.

- **Model:** defaults to `gpt-5.1`; override with `OPENAI_MODEL` in `.env` or `python chat.py --model <id>`.
- **Base URL:** optional `OPENAI_BASE_URL` in `.env` for OpenAI-compatible endpoints.

Flow: **chief complaint** → `Patient: …` → protocol questions at `You:` one step at a time → **`Result:`** exits the program. `/clear` (at `You:`) restarts with a new chief complaint from `skills`. `/quit` exits.

Offline check (no API key): `python test_router_logic.py` validates GOTO parsing.

**Interpretability log:** each run wipes `history/` and writes `history/history.md` — chief complaint, each screen the host shows, each patient reply, the full LLM context (system + structured user fields + node file), tool JSON, and resolved next node. The folder is gitignored.

## Clinical state machine (`nodes/` + `router.py`)

- **Nodes:** one Markdown file per state: `nodes/<node_id>.md` (e.g. `skills`, `LRQ_Step_1`).
- **Front matter (required):** YAML-style block at the top — `kind: question` or `kind: result`, and `patient: "…"` (the line printed as `Question:` / `Result:`). Below that, `# Logic:` and `GOTO node_id **NextNode**` lines define allowed branches.
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
