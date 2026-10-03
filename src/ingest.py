#This program Load raw files, clean text, split into chunks and save JSONL.
    #By calling main, documents are loaded through load_documents, cleaned through clean_text, split into chunks and saved to a JSONL file.

#We import html to fix HTML codes.
import html
#We import json to write each chunk as a JSON line.
import json
#we import re for Regular expressions, to clean the text and remove unwanted characters.
import re
#we import Path to handle file/folder paths.
from pathlib import Path

#we import Document from langchain_core.documents to represent each document with its content and metadata.
from langchain_core.documents import Document
#We import RecursiveCharacterTextSplitter from langchain_text_splitters to split the documents into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
#We import text from sqlalchemy to handle SQL queries and database interactions.
from sqlalchemy import text

#RAW is the path to the folder where the raw data is stored.
RAW = Path("data/raw")
# OUT is the path to the file where the processed chunks will be saved.
OUT = Path("data/processed/chunks.jsonl")

#this function is simply to load the documents from the folder
def load_documents(folder):
    docs = []
    for path in sorted(folder.glob("*.txt"))[:2000]:
        text = path.read_text(encoding="utf-8")
        _, doc_id, label = path.stem.split("_")
        docs.append(Document(
            page_content=text,
            metadata={"doc_id": doc_id, "label": label, "source": path.name},
        ))
    return docs

#this function is simply to clean the text, removing unwanted characters and formatting issues.
def clean_text(text):
    text = html.unescape(text)
    text = text.replace("#39;", "'").replace("quot;", '"').replace("amp;", "&")
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#This function is simply to split the documents into smaller chunks, making it easier for processing and analysis.
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

#this function calls the other functions to load, clean,split and saves them to a JSONL file.
def main():
    docs = load_documents(RAW)
    print(f"Loaded: {len(docs)} documents")

    for d in docs:
        d.page_content = clean_text(d.page_content)
    docs = [d for d in docs if d.page_content]
    print(f"After cleaning: {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, "text": c.page_content, **c.metadata}) + "\n")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

    #1 article = 1 chunk, 50000 articles = 50000 chunks, 50000 chunks = 50000 lines in the JSONL file.