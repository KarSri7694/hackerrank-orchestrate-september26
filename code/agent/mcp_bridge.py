from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path

from fastmcp import Client


class MCPToolBridge:
    def __init__(self, server, dataset_image_root: str | Path):
        self.server = server
        self.dataset_image_root = Path(dataset_image_root)

    async def _list(self):
        async with Client(self.server) as client:
            return await client.list_tools()

    async def _call(self, name, arguments):
        async with Client(self.server) as client:
            result = await client.call_tool(name, arguments)
            if getattr(result, "structured_content", None) is not None:
                return result.structured_content
            items = []
            for item in getattr(result, "content", []):
                if getattr(item, "type", None) == "text":
                    try:
                        items.append(json.loads(item.text))
                    except (ValueError, TypeError):
                        items.append(item.text)
            return items[0] if len(items) == 1 else items

    def list_tools(self):
        return asyncio.run(self._list())

    def call(self, name: str, arguments: dict):
        return asyncio.run(self._call(name, arguments))

    def openai_tools(self):
        return [{"type": "function", "name": tool.name, "description": tool.description or "",
                 "parameters": _strict_object_schema(tool.inputSchema), "strict": True}
                for tool in self.list_tools()]


def _strict_object_schema(schema):
    """Make FastMCP/Pydantic schemas valid for strict OpenAI function calls.

    FastMCP commonly omits ``additionalProperties`` for ordinary object
    schemas. OpenAI strict tools require it to be false at every object node,
    including objects nested under arrays or ``anyOf``. Work on a copy so the
    MCP server's own schema is never mutated.
    """
    result = deepcopy(schema)
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            # Strict function schemas require every declared property to be
            # present. Optional application inputs are already represented by
            # nullable schemas, so requiring the key is lossless and avoids
            # provider-side schema rejection.
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
        for value in node.values():
            if isinstance(value, dict):
                visit(value)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
    visit(result)
    return result
