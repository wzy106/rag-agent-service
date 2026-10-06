import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_rag_server.py") as client:
        # 无关问题
        result = await client.call_tool("search_knowledge", {"query": "今天天气怎么样？"})
        print(f"【无关问题】\n{result.data}\n")

        # 正常问题
        result = await client.call_tool("search_knowledge", {"query": "MMR 是什么？"})
        print(f"【正常问题】\n{result.data[:200]}...")

asyncio.run(main())