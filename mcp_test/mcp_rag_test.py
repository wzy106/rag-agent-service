import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_rag_server.py") as client:
        tools = await client.list_tools()
        print("可用工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        result = await client.call_tool("search_knowledge", {"query": "MCP 是什么？"})
        print(f"\n检索结果：\n{result.data}")

asyncio.run(main())