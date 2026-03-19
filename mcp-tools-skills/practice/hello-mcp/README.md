# Hello World MCP Server

A minimal MCP (Model Context Protocol) server for practice. No Google Drive, credentials, or external services required.

## Setup

```bash
cd mcp-tools-skills/practice/hello-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Test with MCP Inspector (recommended)

Run the Inspector with the server as a child process (stdio transport — avoids proxy/token issues):

```bash
cd mcp-tools-skills/practice/hello-mcp
npx -y @modelcontextprotocol/inspector ./venv/bin/python server.py
```

The Inspector opens in your browser and connects automatically. No manual "Connect" needed.

### Port already in use

If you see `Proxy Server PORT IS IN USE at port 6277` or `MCP Inspector PORT IS IN USE at port 6274`, a previous Inspector run didn't fully shut down. Free the ports, then run again:

```bash
kill -9 $(lsof -t -i :6277) 2>/dev/null
kill -9 $(lsof -t -i :6274) 2>/dev/null
npx -y @modelcontextprotocol/inspector ./venv/bin/python server.py
```

## Run the Server Standalone (streamable HTTP)

To run the server for remote/browser clients (e.g. at `http://localhost:8000/mcp`), change `transport="stdio"` to `transport="streamable-http"` in `server.py`, then:

```bash
python server.py
```

## What's Included

- **Tool**: `hello(name)` — Returns a greeting. Try `hello("MCP")` → `"Hello, MCP!"`
- **Resource**: `greeting://hello` — A read-only resource returning a greeting string

## Requirements

- Python 3.10+
