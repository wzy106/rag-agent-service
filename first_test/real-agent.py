# -*- coding: utf-8 -*-
"""
一个最小可运行的 ReAct Agent（接真实 LLM 版）
"""

import os
import re
import math
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 0. 初始化
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = "deepseek-flash"  # 换 OpenAI 官方就改成 gpt-4o-mini


# ============================================================
# 1. 工具层
# ============================================================

def tool_calculator(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


def tool_weather(city: str) -> str:
    """天气工具（模拟数据）"""
    fake_data = {
        "北京": "晴天，25度",
        "上海": "小雨，18度",
        "深圳": "多云，30度",
    }
    return fake_data.get(city, f"没有找到 {city} 的天气数据")


TOOLS = {
    "calculator": tool_calculator,
    "weather": tool_weather,
}


# ============================================================
# 2. 真实 LLM
# ============================================================

SYSTEM_PROMPT = """你是一个 ReAct Agent。

可用工具：
- calculator：计算数学表达式，输入是表达式，例如 123 * 456
- weather：查询天气，输入是城市名，例如 北京

你必须严格按以下两种格式之一输出，不要输出其他内容。

格式一（需要调用工具时）：
Thought: 你的思考
Action: 工具名
Action Input: 工具参数

格式二（已经知道答案时）：
Thought: 你的思考
Final Answer: 最终答案

重要规则：
- 如果历史里已经有 Observation 能回答问题，直接输出 Final Answer，不要再调用工具。
- 一次只输出一个 Action。
"""


def real_llm(history: str) -> str:
    """调用真实大模型"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": history},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ============================================================
# 3. ReAct 核心循环
# ============================================================

def run_agent(question: str, max_steps: int = 5) -> str:
    print(f"\n========== 开始处理：{question} ==========\n")

    history = f"Question: {question}\n"

    for step in range(1, max_steps + 1):
        print(f"----- 第 {step} 轮 -----")

        # A. 调用 LLM
        llm_output = real_llm(history)
        print(f"[LLM 输出]\n{llm_output}\n")

        # B. 是否给出最终答案
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[1].strip()
            if final_answer:
                return final_answer
            return "模型没有给出具体答案"

        # C. 解析 Action
        action_match = re.search(r"Action:\s*(\w+)", llm_output)
        input_match = re.search(r"Action Input:\s*(.+)", llm_output)

        if not action_match or not input_match:
            return f"解析失败：LLM 输出格式不对\n{llm_output}"

        action_name = action_match.group(1).strip()
        action_input = input_match.group(1).strip()

        # D. 执行工具
        if action_name not in TOOLS:
            observation = f"没有名为 {action_name} 的工具"
        else:
            try:
                observation = TOOLS[action_name](action_input)
            except Exception as e:
                observation = f"工具执行失败：{e}"

        print(f"[执行工具] {action_name}({action_input})")
        print(f"[观察结果] {observation}\n")

        # E. 拼回 history
        history += f"{llm_output}\nObservation: {observation}\n"

        # F. 裁剪 history，防止无限增长
        MAX_HISTORY = 3000
        if len(history) > MAX_HISTORY:
            history = "...(早期记录已省略)...\n" + history[-MAX_HISTORY:]

    return "达到最大步数，任务未完成"



# ============================================================
# 4. 程序入口
# ============================================================

if __name__ == "__main__":
    answer1 = run_agent("北京今天天气怎么样？")
    print(f">>> 最终答案：{answer1}\n")

    answer2 = run_agent("123 乘以 456 等于多少？")
    print(f">>> 最终答案：{answer2}\n")