from __future__ import annotations

import asyncio
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
        return [{"type": "function", "name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema, "strict": True} for tool in self.list_tools()]

