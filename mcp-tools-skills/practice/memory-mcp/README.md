# Memory MCP — Persistent Agentic Memory

An MCP server that provides a long-term scratchpad. The agent can store and recall key-value pairs across conversation turns.

## Setup

```bash
cd mcp-tools-skills/practice/memory-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run with MCP Inspector

```bash
cd mcp-tools-skills/practice/memory-mcp
npx -y @modelcontextprotocol/inspector ./venv/bin/python server.py
```

### Port already in use

```bash
kill -9 $(lsof -t -i :6277) 2>/dev/null
kill -9 $(lsof -t -i :6274) 2>/dev/null
npx -y @modelcontextprotocol/inspector ./venv/bin/python server.py
```

## Tools

| Tool | Purpose |
|------|---------|
| `store(key, value)` | Save a key-value pair to the scratchpad. Creates `scratchpad.json` if it doesn't exist. |
| `recall(key)` | Look up a value by key. Returns the value or a "not found" message. |
| `recall_item(key, index)` | Get the item at index from a stored list (0-based). |
| `list_keys()` | List all keys in the scratchpad. |
| `delete(key)` | Remove a key-value pair from the scratchpad. |

### Storing lists

The `value` in `store` can be a string, list, or dict. In the Inspector, for a list use JSON:

- **List**: `["A", "B", "C"]` — e.g. `store("cleaned_columns", ["A", "B", "C"])`
- **Dict**: `{"x": "col_a", "y": "col_b"}` — e.g. `store("axis_map", {"x": "col_a", "y": "col_b"})`

`recall` returns lists and dicts as JSON strings.

## Usage Flow

1. **Turn 1**: Agent calls `store("cleaned_columns", "A, B, C")` to remember which columns were cleaned.
2. **Turn 10**: Agent calls `recall("cleaned_columns")` to get `"A, B, C"` without re-scanning the sheet.

The scratchpad file (`scratchpad.json`) persists on disk in this folder.

## Requirements

- Python 3.10+
