# Ollama Agent Host

`ollama_host.py` runs a local **Ollama** model with the **Memory MCP** server (stdio). **Phase 1** routes on `skill_manifest.yaml` only; **Phase 2–3** hydrates `skills/personal_context_sop.md` (plus tool rules) and runs a **ReAct-style loop**: Ollama may request tools → the host validates `store` args when needed → MCP runs → results return until the model replies with text only.

## How the agent, skills, tools, and MCP fit together


| Piece          | What it is                             | Role                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent**      | The Ollama LLM (e.g. `llama3.2`)       | Reads the conversation, follows instructions, and **decides** when to use a tool and with what arguments. It does **not** run Python or touch the disk directly.<br><br>**Context:** The agent starts each user turn with a context that includes:<br>- The `skills.md` content (as a system prompt)<br>- The full conversation history so far<br>- The available tool definitions.<br>After every tool call, the host appends the tool result as a message, and the updated conversation (including user input, tool outcomes, assistant replies, etc.) forms the new context for the next round. This lets the agent “see” everything that’s happened so far and base its next actions on the growing conversation state. |
| **Skills / SOP** | `skill_manifest.yaml` + `skills/personal_context_sop.md` | **Manifest** triggers routing; **SOP** (after YES) defines taxonomy keys, category-vs-value rules, and procedure. See `skills.md` for a legacy pointer only—the host does **not** load it automatically.                                                                                                                                                                                                                              |
| **Tools**      | `store`, `recall`, `list_keys`, …      | **Named actions** with a JSON schema. **`store`** requires a non-empty **value** (not `""`, `{}`, `[]`): key = category, value = the user’s fact. The host and MCP reject empty values so the model can retry with correct args.                                                                                                                                                                                                                  |
| **MCP server** | `memory-mcp/server.py` (child process) | The **implementation** of those tools. It runs over **stdio**; when the host sends `tools/call`, it updates or reads `**scratchpad.json`** on disk. MCP is the **nervous system** between clients and the real logic.                                                                                                                                                                                    |
| **Host**       | `ollama_host.py`                       | **Orchestrator**: starts the MCP server, loads tools from MCP, sends `skills.md` + chat to Ollama, and when the model asks for a tool, **calls MCP** and feeds the **tool result** back into the chat until the model answers with plain text.                                                                                                                     |


**End-to-end flow (one user turn):**

1. **You** type a message → host appends it to the message list (short-term memory).
2. **Host** sends messages + tool schemas to **Ollama** (system prompt already includes **skills**).
3. **Agent** may return **native tool calls** (or, with small models, JSON-like text the host can parse as a fallback).
4. **Host** executes each call via the **MCP client** → **MCP server** runs the matching Python function → reads/writes **scratchpad.json** (long-term memory).
5. **Host** appends **tool** messages with the string results and calls Ollama again.
6. Repeat from step 3 until the model responds **without** new tool calls; that text is printed as **Assistant**.

So: **skills** steer the agent; **tools** are the contract the agent speaks; **MCP** is where tools are actually run; the **host** closes the loop between Ollama and MCP.

## Default model (tool-capable)

The default is `**llama3.2`** (`DEFAULT_MODEL` in `ollama_host.py`). Plain `**llama3**` does **not** support tool calling in Ollama.

## Prerequisites

- [Ollama](https://ollama.com) installed and running (app or `ollama serve`)
- The **default model** pulled locally (see below)
- Python 3.10+

## Ensure the model is on your machine

Before first run, install the default model:

```bash
ollama pull llama3.2
```

**Check that it is present:**

```bash
ollama list
```

You should see a line like `llama3.2` (or `llama3.2:latest`). If not, run `ollama pull llama3.2` again.

**Confirm Ollama can load it:**

```bash
ollama show llama3.2
```

If this errors with “model not found”, the tag is not installed — pull it first.

## Setup

From `mcp-tools-skills/practice/`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-ollama-host.txt
```

## Run

```bash
cd mcp-tools-skills/practice
source venv/bin/activate   # if using venv
python ollama_host.py      # uses default llama3.2
```

Override the model:

```bash
python ollama_host.py --model llama3.1
```

Use the same interpreter that has `mcp` installed — the host spawns `memory-mcp/server.py` with `sys.executable`.

## What you should see

With **`QUIET_UI = True`** (default):

- **Stdout:** only `\nYou:` prompts and `\nAssistant:\n...` (no `<thinking>` in the assistant text).
- **Stderr:** banner, `[Host] …` lines, **MODEL** / **ROUTER** / **TOOLS** / ReAct rounds (full router raw, including any `<thinking>` there). Tip: `python ollama_host.py 2>/dev/null` if you want *only* the chat on stdout.

With **`QUIET_UI = False`:** the same sections print on **stdout**; host/bootstrap lines still use **stderr** so startup stays separated from the transcript a little.

Example: *I moved to Tucson in 2022* → router **YES** → then `store` under **TOOLS** (on stderr if quiet).

## Troubleshooting


| Issue                                               | What to try                                                                                                                                                     |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `does not support tools` (400)                      | Do not use plain `llama3`. Use `llama3.2`, `llama3.1`, `qwen2.5`, etc.                                                                                          |
| Model missing / 404                                 | `ollama pull llama3.2` then `ollama list`                                                                                                                       |
| Model never calls tools                             | Prefer `llama3.2` / `llama3.1` / `qwen2.5`; confirm `ollama show <model>` works                                                                                 |
| `store` error: empty value / `{}`                   | Model put the fact in the **key** only. Follow SOP: taxonomy **key** (e.g. `academic_field`) + **value** with the fact (e.g. `"mathematics"`).                  |
| Model prints `{"name": "store", ...}` in plain text | The host tries to parse and run those as a fallback; the system prompt also tells the model to use native tools only. Try `llama3.1` if `llama3.2` stays noisy. |
| Connection refused                                  | Start Ollama (`ollama serve` or the app)                                                                                                                        |
| MCP import error                                    | Activate venv and `pip install -r requirements-ollama-host.txt`                                                                                                 |
| Wrong Python for MCP server                         | Run `ollama_host.py` with the same `python` that can `import mcp`                                                                                               |


## Introspection: `<thinking>` traces & quiet UI

In `ollama_host.py` (top-level constants):

| Constant | Purpose |
|----------|---------|
| **`QUIET_UI`** | **`True` (default):** stdout is only **`You:`** / **`Assistant:`** (thinking stripped). **MODEL** / **TOOLS** / **ROUTER** / ReAct rounds go to **stderr** so you can `2>debug.log` or ignore them. **`False`:** those sections print on **stdout** (still **no** `<thinking>` inside MODEL boxes — stripped for display; full trace in JSONL). |
| **`TRACE_LOG_PATH`** | Default `practice/host_history/history_log.jsonl` — **one JSON object per user turn** (query, hydrated context, thinking blocks, tool trace, final reply raw/stripped). Set to **`None`** to disable. |
| **`INTROSPECTION_COT_RULES`** | Appended to the casual system prompt and to the executor (SOP + tools) so the model is nudged to emit `<thinking>...</thinking>` before acting. The host **keeps** full text in API history; **printed** assistant text **never** shows `<thinking>` (trace files keep raw + extracted blocks). |

After you exit the REPL (EOF, `quit`, or Ctrl+C), the host also writes **`host_history/history_log.json`** — a single JSON file with `{ "turns": [ ... ] }` for the whole session.

The **`host_history/`** directory is gitignored (JSONL + snapshot stay out of the repo).

## Options

```bash
python ollama_host.py --model mistral
```

