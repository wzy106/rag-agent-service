# -*- coding: utf-8 -*-
"""
一个最小可运行的 ReAct Agent
不需要任何第三方库，直接 python agent.py 即可运行
"""

import re
import math


# ============================================================
# 1. 工具层：Agent 的"手脚"
# ============================================================

def tool_calculator(expression: str) -> str:
    """计算器工具：计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


def tool_weather(city: str) -> str:
    """天气工具：模拟查询天气"""
    fake_data = {
        "北京": "晴天，25度",
        "上海": "小雨，18度",
        "深圳": "多云，30度",
    }
    return fake_data.get(city, f"没有找到 {city} 的天气数据")


# 工具注册表：名字 -> 函数
TOOLS = {
    "calculator": tool_calculator,
    "weather": tool_weather,
}


# ============================================================
# 2. 假 LLM：先用它跑通逻辑，之后可替换为真实 API
# ============================================================

def fake_llm(prompt: str) -> str:
    """
    模拟大模型的输出。
    真实场景下，把这里换成 OpenAI / DeepSeek 的 API 调用。
    """

    # 情况1：用户问天气，且还没有拿到观察结果
    if "天气" in prompt and "Observation" not in prompt:
        return (
            "Thought: 用户想知道天气，我应该调用 weather 工具。\n"
            "Action: weather\n"
            "Action Input: 北京"
        )

    # 情况2：已经拿到天气结果
    if "Observation: 晴天，25度" in prompt:
        return (
            "Thought: 我已经知道北京天气了。\n"
            "Final Answer: 北京今天晴天，25度，适合穿短袖。"
        )

    # 情况3：已经拿到计算结果（要放在计算判断前面）
    if "Observation:" in prompt and "56088" in prompt:
        return (
            "Thought: 计算完成。\n"
            "Final Answer: 123 乘以 456 等于 56088。"
        )

    # 情况4：用户问计算，且还没有拿到观察结果
    if ("乘" in prompt or "*" in prompt) and "Observation" not in prompt:
        return (
            "Thought: 这是一个计算问题，我调用 calculator 工具。\n"
            "Action: calculator\n"
            "Action Input: 123 * 456"
        )

    # 兜底
    return (
        "Thought: 我不知道该怎么办。\n"
        "Final Answer: 抱歉，我暂时无法处理这个问题。"
    )


# ============================================================
# 3. ReAct 核心循环
# ============================================================

def run_agent(question: str, max_steps: int = 5) -> str:
    """
    Agent 主循环：
    想 -> 做 -> 看 -> 再想 -> 直到给出 Final Answer
    """
    print(f"\n========== 开始处理：{question} ==========\n")

    # history 就是"记忆"，每轮把 LLM 输出和工具结果拼进去
    history = f"Question: {question}\n"

    for step in range(1, max_steps + 1):
        print(f"----- 第 {step} 轮 -----")

        # A. 让 LLM 思考
        llm_output = fake_llm(history)
        print(f"[LLM 输出]\n{llm_output}\n")

        # B. 如果 LLM 给出了最终答案，结束
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[1].strip()
            return final_answer

        # C. 解析 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.+)", llm_output)
        input_match = re.search(r"Action Input:\s*(.+)", llm_output)

        if not action_match or not input_match:
            return "解析失败：LLM 输出格式不对"

        action_name = action_match.group(1).strip()
        action_input = input_match.group(1).strip()

        # D. 执行工具
        if action_name not in TOOLS:
            observation = f"没有名为 {action_name} 的工具"
        else:
            observation = TOOLS[action_name](action_input)

        print(f"[执行工具] {action_name}({action_input})")
        print(f"[观察结果] {observation}\n")

        # E. 关键一步：把这一轮的内容拼回 history
        history += f"{llm_output}\nObservation: {observation}\n"

    return "达到最大步数，任务未完成"


# ============================================================
# 4. 程序入口
# ============================================================

if __name__ == "__main__":
    # 测试1：天气
    answer1 = run_agent("北京今天天气怎么样？")
    print(f">>> 最终答案：{answer1}\n")

    # 测试2：计算
    answer2 = run_agent("123 乘以 456 等于多少？")
    print(f">>> 最终答案：{answer2}\n")