#The purpose of this program is to take a question and returns the 5 most similar chunks from the vector DB.
import sys
from langchain_chroma import Chroma
from index import DB_DIR, get_embeddings


    #db is a variable connected to Chroma vector database.
def search(query, k=5):
    #When calling Chroma(), you need 1- to associate it with a variable, in this case "db"
           #2-give the embedding function to know which model turns text into vectors
               #3-to tell to which directory it lives.
    db = Chroma(collection_name="news", embedding_function=get_embeddings(), persist_directory=DB_DIR)
    #find the k closest chunks.
    return db.similarity_search_with_score(query, k=k)

#this block runs only when you launch search.py directly.
    #it reads the question from the terminal, searches and print the 5 results nicely.
if __name__ == "__main__":
    #sys.argv[1:] is to skip the filename.
    #tech companies cutting jobs is the default question if you type nothing.
    query = " ".join(sys.argv[1:]) or "tech companies cutting jobs"
    for doc, score in search(query):
        #The lower the score is the more similar it is, tje ,ost optimal is below 0.8.
            #The chroma's default score is the squared Euclidean distance.
        print(f"[{score:.3f}] {doc.metadata['label']:8} | {doc.page_content[:100]}")