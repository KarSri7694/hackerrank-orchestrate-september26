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
        return asyncio.run(self._call(name, _normalise_tool_arguments(name, arguments)))

    def openai_tools(self):
        result = []
        for tool in self.list_tools():
            schema = getattr(tool, "input_schema", None)
            if schema is None:  # Compatibility with pre-MCP-SDK-v2 objects.
                schema = tool.inputSchema
            result.append({"type": "function", "name": tool.name, "description": tool.description or "",
                           "parameters": _strict_object_schema(schema), "strict": True})
        return result


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


def _normalise_tool_arguments(name: str, arguments: dict) -> dict:
    """Repair JSON values that a permissive local model returned as strings.

    The advertised ``simulate_plan`` schema requires ``spending_changes`` to
    be either an array or null.  Qwen nevertheless emitted the literal JSON
    strings ``"[]"`` and ``"null"`` in production traces.  FastMCP correctly
    rejects those strings before the tool function runs.  Decode only those
    JSON containers, leaving all other invalid values intact for normal schema
    validation rather than silently accepting arbitrary model output.
    """
    if name != "simulate_plan" or not isinstance(arguments, dict):
        return arguments
    result = dict(arguments)
    value = result.get("spending_changes")
    if not isinstance(value, str):
        return result
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return result
    if parsed is None or isinstance(parsed, list):
        result["spending_changes"] = parsed
    return result
