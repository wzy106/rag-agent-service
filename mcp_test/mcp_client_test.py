import asyncio
from fastmcp import Client
from pathlib import Path

async def main():
    async with Client(Path("mcp_server.py")) as client:
        tools = await client.list_tools()
        print("可用工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        result = await client.call_tool("add", {"a": 3, "b": 5})
        print(f"\nadd(3, 5) = {result}")

        result = await client.call_tool("greet", {"name": "小明"})
        print(f"greet('小明') = {result}")

asyncio.run(main())