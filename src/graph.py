import json
import sys
from typing import Literal, Optional, TypedDict

#This defines the exact shape of the LLM output.
from pydantic import BaseModel, Field
#Talk to the local Ollama model
from langchain_ollama import ChatOllama
#Build the flowchart.
from langgraph.graph import StateGraph, START, END

from search import search

#Connects to llama3.2 running in Ollama.
#temperature=0 -> No randomness, same input = same output.
llm = ChatOllama(model="llama3.2", temperature=0)


# Shape of the structured output we want from the LLM (the output template)
class Extraction(BaseModel):
    #Text or None
    company: Optional[str] = Field(
    description="Company the article is about. Never the news agency (Reuters, AP, AFP)."
    )
    #Text
    event: str = Field(description="Main event in a few words")
    #Only positive, negative, neutral
    sentiment: Literal["positive", "negative", "neutral"]
    #Only the 4 AG News labels. Forces the LLM to awnser in this exact format.
    category: Literal["world", "sports", "business", "scitech"]


# Shared notebook passed between nodes.
    #Each node reads from it and writes to it.
class State(TypedDict, total=False):
    #Filled by the user
    query: str
    #Filled by the route
    mode: str
    #Filled by retrieve
    docs: list
    #filled by generate_awnser
    answer: str
    #Filled by extract_info
    extraction: dict
    attempts: int


# --- Nodes ---
#Looks at your question and decides what kind of request it is.
def route(state):
    #if it starts with extract, you want structured info.
    #else, you're asking a question
    mode = "extract" if state["query"].lower().startswith("extract") else "question"
    #resets the retry counter to 0, so we start fresh
    return {"mode": mode, "attempts": 0}


#Goes to the vector DB and fethces the most relevant articles.
def retrieve(state):
    #In extract mode, it first remoes the word "extract"
    query = state["query"].split(":", 1)[-1] if state["mode"] == "extract" else state["query"]
    k = 1 if state["mode"] == "extract" else 3
    return {"docs": [doc for doc, _ in search(query, k=k)]}

#Takes the 3 articles and glues them into one block of text.
def generate_answer(state):
    context = "\n\n".join(d.page_content for d in state["docs"])
    prompt = (
        "Answer the question using only this context.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['query']}"
    )
    return {"answer": llm.invoke(prompt).content}


#Take the one article and asks the LLM to fill in the Extraction form.
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
def pick_path(state): #Reads the mode decided by route.
    return state["mode"]


def check_extraction(state): #Checks if the extraction worked
    if state.get("extraction") or state["attempts"] >= 2:
        return "done"
    return "retry"


# --- Build the graph ---
builder = StateGraph(State) #creates an empty flowchart that will pas our notebook around
#add_node put each function on the board as a box.
builder.add_node("route", route)
builder.add_node("retrieve", retrieve)
builder.add_node("generate_answer", generate_answer)
builder.add_node("extract_info", extract_info)

builder.add_edge(START, "route") #Every request starts with route.
builder.add_edge("route", "retrieve") #always search
builder.add_conditional_edges("retrieve", pick_path, #after searching, asks pick_path which way to go
                              {"question": "generate_answer", "extract": "extract_info"})
builder.add_edge("generate_answer", END) #after awnsering we're done.
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

