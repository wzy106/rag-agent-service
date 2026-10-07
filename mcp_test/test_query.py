import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_rag_server.py") as client:
        result = await client.call_tool("search_knowledge", {"query": "@tool 装饰器做了什么？"})
        print(result.data)

asyncio.run(main())