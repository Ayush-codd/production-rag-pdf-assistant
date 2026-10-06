from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        source = chunk.metadata.get(
            "source_file",
            "unknown"
        )

        page = chunk.metadata.get(
            "page_number",
            0
        )

        chunk.metadata["chunk_id"] = (
            f"{source}_page_{page}_chunk_{index}"
        )

    return chunks