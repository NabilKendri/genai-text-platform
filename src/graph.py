import json
import sys
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from search import search

llm = ChatOllama(model="llama3.2", temperature=0)


# Shape of the structured output we want from the LLM
class Extraction(BaseModel):
    company: Optional[str] = Field(description="Main company or organization, if any")
    event: str = Field(description="Main event in a few words")
    sentiment: Literal["positive", "negative", "neutral"]
    category: Literal["world", "sports", "business", "scitech"]


# Shared notebook passed between nodes
class State(TypedDict, total=False):
    query: str
    mode: str
    docs: list
    answer: str
    extraction: dict
    attempts: int


# --- Nodes ---
def route(state):
    mode = "extract" if state["query"].lower().startswith("extract") else "question"
    return {"mode": mode, "attempts": 0}


def retrieve(state):
    query = state["query"].split(":", 1)[-1] if state["mode"] == "extract" else state["query"]
    k = 1 if state["mode"] == "extract" else 3
    return {"docs": [doc for doc, _ in search(query, k=k)]}


def generate_answer(state):
    context = "\n\n".join(d.page_content for d in state["docs"])
    prompt = (
        "Answer the question using only this context.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['query']}"
    )
    return {"answer": llm.invoke(prompt).content}


def extract_info(state):
    text = state["docs"][0].page_content
    try:
        result = llm.with_structured_output(Extraction).invoke(
            f"Extract information from this news article:\n\n{text}"
        )
        data = result.model_dump() if result else None
    except Exception:
        data = None
    return {"extraction": data, "attempts": state["attempts"] + 1}


# --- Decisions (conditional edges) ---
def pick_path(state):
    return state["mode"]


def check_extraction(state):
    if state.get("extraction") or state["attempts"] >= 2:
        return "done"
    return "retry"


# --- Build the graph ---
builder = StateGraph(State)
builder.add_node("route", route)
builder.add_node("retrieve", retrieve)
builder.add_node("generate_answer", generate_answer)
builder.add_node("extract_info", extract_info)

builder.add_edge(START, "route")
builder.add_edge("route", "retrieve")
builder.add_conditional_edges("retrieve", pick_path,
                              {"question": "generate_answer", "extract": "extract_info"})
builder.add_edge("generate_answer", END)
builder.add_conditional_edges("extract_info", check_extraction,
                              {"retry": "extract_info", "done": END})

graph = builder.compile()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Why are oil prices rising?"
    result = graph.invoke({"query": query})
    if result["mode"] == "extract":
        print(json.dumps(result.get("extraction"), indent=2))
        print("True label:", result["docs"][0].metadata["label"])
    else:
        print(result["answer"])

