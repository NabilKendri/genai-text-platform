#The purpose of this program is to build the search database.
#it turns every chunk of text into a vector and stores it in ChromaDB.
#The goal is for search.py to later find chunks by meaning.
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHUNKS = Path("data/processed/chunks.jsonl")
DB_DIR = "chroma_db"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 1000

#The job of this function is to read chunks.jsonl.
#chunks.jsonl contains the text chunks and their metadata.
#The function will then return a list of Document objects.
def load_chunks(path):
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.pop("text")
            docs.append(Document(page_content=text, metadata=row))
    return docs

#The job of this function is to return the HuggingFaceEmbeddings object.
def get_embeddings():
    #it loads the model specified in the MODEL variable.
    return HuggingFaceEmbeddings(model_name=MODEL)

#The role of this function is to embed chunks in batches of 1000 and save in chroma
def build_index(docs, embeddings):
    db = Chroma(collection_name="news", embedding_function=embeddings, persist_directory=DB_DIR)
    for start in range(0, len(docs), BATCH):
        batch = docs[start:start + BATCH]
        ids = [str(d.metadata["chunk_id"]) for d in batch]
        db.add_documents(batch, ids=ids)
        print(f"Indexed {start + len(batch)}/{len(docs)}", flush=True)
    return db

#The role of this function is to run the 3 previous functions in order
def main():
    #1- read chunks in jsonl.
    docs = load_chunks(CHUNKS)
    #2- display how many chunks were loaded.
    print(f"Loaded {len(docs)} chunks", flush=True)
    #3-attribute vectors (referring to the loaded embedding model to the jsonl docs) to capture meaning and store in db.
    build_index(docs, get_embeddings())
    print(f"Saved to {DB_DIR}/")


if __name__ == "__main__":
    main()

