import sys

from langchain_chroma import Chroma
from index import DB_DIR, get_embeddings


def search(query, k=5):
    db = Chroma(collection_name="news", embedding_function=get_embeddings(), persist_directory=DB_DIR)
    return db.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "tech companies cutting jobs"
    for doc, score in search(query):
        print(f"[{score:.3f}] {doc.metadata['label']:8} | {doc.page_content[:100]}")