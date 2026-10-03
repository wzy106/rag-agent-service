from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    text: str
    count: int

def node_a(state: State) -> State:
    print("进入 node_a")
    return {"text": state["text"] + " hello", "count": state["count"] + 1}

def node_b(state: State) -> State:
    print("进入 node_b")
    return {"text": state["text"] + " world", "count": state["count"] + 1}

def route(state: State) -> str:
    return "b"

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test_001"}}

# 第一轮
print("=== 第一轮 ===")
result1 = graph.invoke({"text": "start", "count": 0}, config=config)
print(result1)

# 第二轮（注意：不传 text 和 count，看看能不能记住上次的状态）
print("\n=== 第二轮 ===")
result2 = graph.invoke({}, config=config)
print(result2)