# Lesson 1: How MCP Works — Hello World

This lesson explains the Model Context Protocol (MCP) and what our Hello World server is doing under the hood.

---

## What is MCP?

**Model Context Protocol (MCP)** is a standard way for AI applications (like Cursor, Claude, or custom agents) to connect to external data and tools. Instead of each app building its own integrations for Google Sheets, databases, APIs, etc., MCP defines a common protocol. Any MCP-compatible client can talk to any MCP-compatible server.

Think of it as **USB for AI**: a standard connector so different devices (clients) can plug into different peripherals (servers).

---

## The Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   MCP Client    │  ←── protocol ──→  │   MCP Server    │
│ (Cursor, Claude,│   (JSON-RPC 2.0)   │ (your Python    │
│  Inspector)     │                    │  server)        │
└─────────────────┘                    └─────────────────┘
```

- **Client**: Requests tools, resources, and prompts. Sends commands and receives results.
- **Server**: Exposes capabilities (tools, resources, prompts). Executes logic and returns data.
- **Protocol**: JSON-RPC 2.0 over a transport (stdio, HTTP, etc.).

---

## The Three Primitives

MCP servers expose three kinds of capabilities:


| Primitive     | Purpose                                                    | Example                            |
| ------------- | ---------------------------------------------------------- | ---------------------------------- |
| **Tools**     | Functions the AI can call. Take inputs, return outputs.    | `hello(name)` → `"Hello, MCP!"`    |
| **Resources** | Read-only data identified by a URI.                        | `greeting://hello` → greeting text |
| **Prompts**   | Pre-written templates the AI can use to generate messages. | "Summarize this document"          |


Our Hello World server uses **Tools** and **Resources**. We did not define any Prompts.

---

## What Our Server Does (Line by Line)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Hello World MCP", json_response=True)
```

- **FastMCP**: A high-level framework from the official MCP Python SDK. It lets you define tools and resources with decorators instead of low-level JSON-RPC.
- `**mcp`**: The server instance. The name appears when clients connect.
- `**json_response=True`**: Ensures responses are valid JSON (helpful for clients).

---

```python
@mcp.tool()
def hello(name: str = "World") -> str:
    """Return a friendly greeting. Use this to verify the MCP server is working."""
    return f"Hello, {name}!"
```

- `**@mcp.tool()**`: Registers this function as an MCP **tool**. The client can discover it and call it by name.
- `**name`**: Input parameter. The client sends this as JSON. Default is `"World"`.
- **Docstring**: Becomes the tool description. The AI uses it to decide when to call this tool.
- **Return value**: Sent back to the client as the tool result.

---

```python
@mcp.resource("greeting://hello")
def get_hello_resource() -> str:
    """A simple read-only resource that returns a greeting."""
    return "Hello from MCP! This is a resource."
```

- `**@mcp.resource("greeting://hello")**`: Registers a **resource** at the URI `greeting://hello`.
- **Resources** are read-only. The client fetches them by URI; they don't take arguments like tools.
- The URI scheme (`greeting://`) is arbitrary. You choose it for your domain.

---

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- `**mcp.run()**`: Starts the server and listens for client messages.
- `**transport="stdio"**`: Uses standard input/output. The client spawns our process and talks over stdin/stdout. No network ports.

---

## Transports: How Client and Server Connect


| Transport           | How it works                                                                              | When to use                                           |
| ------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **stdio**           | Client spawns server process. Communication over stdin/stdout.                            | Local development, Inspector, single-client setups.   |
| **streamable-http** | Server runs as HTTP service. Client connects to a URL (e.g. `http://localhost:8000/mcp`). | Remote access, multiple clients, web/browser clients. |


We use **stdio** so the MCP Inspector can spawn our Python process and connect without network setup or proxy tokens.

---

## The MCP Inspector

The **MCP Inspector** is a debugging tool. It acts as an MCP client so you can:

1. **Discover** what tools and resources your server exposes
2. **Call tools** with custom inputs and see results
3. **Read resources** by URI
4. **Test prompts** (if your server defines them)

When you run:

```bash
npx -y @modelcontextprotocol/inspector ./venv/bin/python server.py
```

the Inspector:

1. Starts a proxy (port 6277) and web UI (port 6274)
2. **Spawns** `./venv/bin/python server.py` as a child process
3. Connects to it via **stdio** (stdin/stdout)
4. Shows the server's tools and resources in the browser

No manual "Connect" is needed because the Inspector starts the server itself.

1. **Discover what tools and resources your server exposes**
  > This means you can see a list of all the functions (“tools”) and special data (“resources”) your server can provide. Think of it like opening the settings menu in an app and seeing all the features that are available for you to use or try out.
2. **Call tools with custom inputs and see results**
  > You can test any function your server supports by choosing it and giving it different values. For example, you could try saying hello to many names, or experiment with how tools behave. It's like pressing buttons on a calculator or filling out a form to see what happens.
3. **Read resources by URI**
  > “Reading a resource by URI” means you can look up a piece of information on your server, using a special address (like a website link, but for your server’s data). If your server has information available at certain “addresses,” you can view or retrieve it easily with the Inspector.
4. **Test prompts (if your server defines them)**
  > Your server can also provide templates (called prompts) that help guide how information is gathered or displayed. If these are available, you can try them out and see how they work—similar to filling in a template letter to see the finished message.

---

## The Request Flow (When You Call `hello`)

1. You click **Run** in the Inspector with `name: "MCP"`.
2. Inspector sends a JSON-RPC request over stdin: `{"method": "tools/call", "params": {"name": "hello", "arguments": {"name": "MCP"}}}`
3. Our server receives it, runs `hello("MCP")`, gets `"Hello, MCP!"`.
4. Server sends a JSON-RPC response over stdout: `{"result": {"content": [{"type": "text", "text": "Hello, MCP!"}]}}`
5. Inspector displays the result.

You never see the JSON-RPC directly; the Inspector and FastMCP handle it.  
  
Q: Where does the code sit? Does the server have access to the terminal? did it clone the code from the repository?

> A: The code for your MCP server sits on your machine—in this case, in the folder you cloned from the repository. When you run the commands given in the setup instructions, you’re working directly with files on your computer. The server itself runs as a local process (a Python program you start with the Inspector or with `python server.py`). The server runs in the terminal (or as a subprocess of the Inspector), and it has access only to what your process can reach on your own computer. The MCP Inspector doesn’t automatically clone code; you need to git clone or otherwise get the code onto your machine first.

---

Q: What runs the code? The CPU on my computer?
> A: Yes—the code is run by the CPU on your own computer. When you launch the MCP server (using the Inspector command or directly), your computer’s Python interpreter starts the server process, and all computations and logic are executed right on your local hardware.

---

Q: How does the server interact with the computer? via the port?

> A: By default, in this lesson, the server and the Inspector communicate using **stdio** (standard input/output: pipes provided by the operating system). No network ports are used for communication between the Inspector and the server in this mode; data is sent as text between processes. If you switch to `streamable-http` transport, then the server listens on a network **port** (like 8000), and clients (such as the Inspector) connect to that port using HTTP requests. The port is just a numbered communication endpoint on your computer.

---

Q: What is a proxy? What is a server?

> A: 
> - **Server:** A server is a program that listens for requests (from users, apps, or clients) and provides responses—like computing the output of a tool or returning a resource. In this lesson, your MCP server listens for MCP requests and responds with results. 
> - **Proxy:** A proxy is an intermediary program that sits between a client and a server. It “proxies” (forwards) requests and responses between them. In the Inspector setup, a proxy is used to handle some logistics like connecting the Inspector web UI (running in your browser) to the server process. This is especially useful for connecting different types of clients or controlling access.

---

## Key Takeaways

1. **MCP = standard protocol** for AI apps to use external tools and data.
2. **Tools** = callable functions. **Resources** = read-only data by URI. **Prompts** = templates.
3. **FastMCP** simplifies building servers with decorators.
4. **Transports** define how client and server connect (stdio vs HTTP).
5. **MCP Inspector** spawns your server and acts as a client for testing.

---

## Next Steps

- Try adding a second tool (e.g. `add(a, b)` that returns `a + b`).
- Add a resource with a dynamic URI, e.g. `greeting://{name}`.
- Switch to `transport="streamable-http"` and connect via URL (requires killing ports 6274/6277 if Inspector was running).

