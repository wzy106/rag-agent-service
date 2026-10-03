import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}今天天气晴朗，气温 25℃，适合出门。"

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

agent = create_agent(model, tools=[get_weather])

# 这里改用 stream_events
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
    version="v3",
)

for snapshot in stream.values:
    latest_message = snapshot["messages"][-1]
    
    if isinstance(latest_message, HumanMessage):
        print(f"User: {latest_message.content}")
        continue
    
    # 处理 content_blocks
    if hasattr(latest_message, "content_blocks"):
        for block in latest_message.content_blocks:
            if block["type"] == "text":
                print(f"Agent: {block['text']}")
            elif block["type"] == "tool_call":
                print(f"Calling tool: {block['name']}, args: {block['args']}")