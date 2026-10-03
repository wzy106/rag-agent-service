import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 定义你希望模型输出的结构 (保持不变)
class WeatherInfo(BaseModel):
    """天气信息"""
    city: str = Field(description="城市名称")
    weather: str = Field(description="天气状况，如晴、多云")
    temperature: int = Field(description="温度，摄氏度")

# 2. 初始化模型 (保持不变)
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)
# 3. 创建输出解析器
parser = PydanticOutputParser(pydantic_object=WeatherInfo)

# 4. 创建提示词模板，并指示模型输出格式
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的天气信息提取助手，请严格按以下格式输出，只输出JSON，不要包含其他任何文字。\n{format_instructions}"),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

# 5. 构建并调用链
chain = prompt | model | parser

# 6. 调用
result = chain.invoke({"query": "北京今天天气怎么样？"})
print(result)
print(type(result))
print(result.city, result.weather, result.temperature)