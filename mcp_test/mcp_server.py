from fastmcp import FastMCP

mcp = FastMCP("我的第一个 MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@mcp.tool
def greet(name: str) -> str:
    """根据名字打招呼"""
    return f"你好，{name}！"

if __name__ == "__main__":
    mcp.run()