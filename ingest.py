from load_pdfs import load_all_pdfs
from chunk import create_chunks
from vector_store import create_vector_db


def main():

    print("Loading PDFs...")

    documents = load_all_pdfs()

    print(f"Loaded pages: {len(documents)}")

    print("Creating chunks...")

    chunks = create_chunks(documents)

    print(f"Created chunks: {len(chunks)}")

    print("Creating FAISS database...")

    create_vector_db(chunks)

    print("FAISS database created successfully.")


if __name__ == "__main__":
    main()