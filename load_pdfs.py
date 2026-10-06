import os

from langchain_community.document_loaders import PyPDFLoader


DATA_DIR = "data"


def load_all_pdfs(folder_path=DATA_DIR):
    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Data folder not found: {folder_path}"
        )

    pdf_files = sorted(
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside '{folder_path}'"
        )

    for file_name in pdf_files:

        pdf_path = os.path.join(
            folder_path,
            file_name
        )

        loader = PyPDFLoader(pdf_path)

        pages = loader.load()

        for page in pages:

            page.metadata["source_file"] = file_name

            page.metadata["page_number"] = (
                page.metadata.get("page", 0) + 1
            )

        documents.extend(pages)

    return documents