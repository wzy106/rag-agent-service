import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import before_model, after_model

load_dotenv()

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 这里模拟一个假数据，真实项目里应该调用外部 API
    return f"{city}今天天气晴朗，气温 25℃，适合出门。"

@tool
def calculator(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

# 2. 初始化模型
# 如果你用 OpenAI 官方，删掉 base_url 这一行，把 api_key 换成你的真实 key
# 如果你用 DeepSeek、Kimi、通义千问等国内模型，请修改 base_url 和 api_key
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)
# 3. 创建 Agent
# 新增：定义中间件
@before_model
def log_before(state, runtime):
    print("[日志] 即将调用模型")

@after_model
def log_after(state, runtime):
    print("[日志] 模型调用完成")

# create_agent 是 LangChain 1.x 的最新 API，底层集成了 LangGraph
memory = InMemorySaver()
agent = create_agent(
    model,
    tools=[get_weather, calculator],
    checkpointer=memory,
    system_prompt="你是一个甜妹专业解说员，回答简洁，不要超过30个字",
    middleware=[log_before, log_after]     # 新增
)
# 4. 运行 Agent
config = {"configurable": {"thread_id": "user_001"}}

# 第一轮
print("=== 第一轮 ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
    config=config
):
    for node_name, node_data in chunk.items():
        if not node_data:
            continue
        for msg in node_data.get("messages", []):
            if msg.content:
                print(f"[{msg.type}] {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  → 调用工具: {tc['name']}, 参数: {tc['args']}")
# 第二轮（注意这里直接问“那上海呢？”）
print("\n=== 第二轮 ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "那上海呢？"}]},
    config=config
):
    for node_name, node_data in chunk.items():
        if not node_data:
            continue
        for msg in node_data.get("messages", []):
            if msg.content:
                print(f"[{msg.type}] {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  → 调用工具: {tc['name']}, 参数: {tc['args']}")


# 5. 打印完整对话历史
"""for msg in result["messages"]:
    print(f"\n[{msg.type}]:")
    if msg.content:
        print(msg.content)
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  (调用工具: {msg.tool_calls})")"""