"""
Hello World MCP Server

A minimal MCP server for practice. Exposes a simple tool and resource.
No external services (Google Drive, etc.) required.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Hello World MCP", json_response=True)


@mcp.tool()
def hello(name: str = "World") -> str:
    """Return a friendly greeting. Use this to verify the MCP server is working."""
    return f"Hello, {name}!"


@mcp.resource("greeting://hello")
def get_hello_resource() -> str:
    """A simple read-only resource that returns a greeting."""
    return "Hello from MCP! This is a resource."


if __name__ == "__main__":
    # Use stdio for Inspector (spawns server directly, no proxy). Use "streamable-http" for browser/remote clients.
    mcp.run(transport="stdio")
