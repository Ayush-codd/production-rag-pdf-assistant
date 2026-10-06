import os

from langchain_community.vectorstores import FAISS

from embedding import get_embeddings


DB_PATH = "db"


def create_vector_db(chunks):

    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(
        DB_PATH,
        exist_ok=True
    )

    vector_db.save_local(DB_PATH)

    return vector_db


def load_vector_db():

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            "FAISS database not found. "
            "Run: python ingest.py"
        )

    embeddings = get_embeddings()

    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db