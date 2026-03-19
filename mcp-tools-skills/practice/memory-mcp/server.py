"""
Memory MCP Server — Persistent Agentic Memory

Exposes tools to store and recall key-value pairs in a local scratchpad file.
The agent can persist facts across conversation turns.
"""

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Memory MCP", json_response=True)

# Scratchpad file lives next to this script
SCRATCHPAD_PATH = Path(__file__).parent / "scratchpad.json"


def _load_scratchpad() -> dict:
    """Load scratchpad from disk. Returns empty dict if file doesn't exist."""
    if not SCRATCHPAD_PATH.exists():
        return {}
    try:
        with open(SCRATCHPAD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_scratchpad(data: dict) -> None:
    """Write scratchpad to disk. Creates file if it doesn't exist."""
    with open(SCRATCHPAD_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _format_value(val: Any) -> str:
    """Format a value for return. Lists/dicts become JSON strings."""
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return str(val)


@mcp.tool()
def store(key: str, value: str | list[str] | dict[str, Any]) -> str:
    """Save a key-value pair to the long-term scratchpad. Value can be a string, a list (e.g. ['A','B','C']), or a dict. Use this to remember facts, variable names, column choices, or any state the agent needs to recall later."""
    data = _load_scratchpad()
    data[key] = value
    _save_scratchpad(data)
    return f"Stored: {key} = {_format_value(value)}"


@mcp.tool()
def recall(key: str) -> str:
    """Look up a value by key from the long-term scratchpad. Returns the stored value (lists/dicts as JSON), or a message if the key is not found."""
    data = _load_scratchpad()
    if key not in data:
        return f"Key '{key}' not found in scratchpad."
    return _format_value(data[key])


@mcp.tool()
def recall_item(key: str, index: int) -> str:
    """Get the item at a given index from a stored list. Uses 0-based indexing (0 = first item, 1 = second, etc.). Returns an error if the key is not found, the value is not a list, or the index is out of range."""
    data = _load_scratchpad()
    if key not in data:
        return f"Key '{key}' not found in scratchpad."
    val = data[key]
    if not isinstance(val, list):
        return f"Key '{key}' does not store a list (found {type(val).__name__})."
    if index < 0 or index >= len(val):
        return f"Index {index} out of range for list of length {len(val)}."
    return _format_value(val[index])


@mcp.tool()
def list_keys() -> str:
    """List all keys currently stored in the scratchpad. Useful to see what the agent has remembered."""
    data = _load_scratchpad()
    if not data:
        return "Scratchpad is empty."
    keys = list(data.keys())
    return "Stored keys: " + ", ".join(keys)


@mcp.tool()
def delete(key: str) -> str:
    """Remove a key-value pair from the scratchpad. Use this to forget or correct stored facts."""
    data = _load_scratchpad()
    if key not in data:
        return f"Key '{key}' not found in scratchpad."
    del data[key]
    _save_scratchpad(data)
    return f"Deleted: {key}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
