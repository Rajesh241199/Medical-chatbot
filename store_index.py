from src.helper import (
    INDEX_NAME,
    create_pinecone_index_if_not_exists,
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    index_documents_to_pinecone,
)


DATA_PATH = "Data/"


def main():
    create_pinecone_index_if_not_exists(index_name=INDEX_NAME)

    extracted_data = load_pdf_file(DATA_PATH)
    print(f"Loaded documents/pages: {len(extracted_data)}")

    filtered_data = filter_to_minimal_docs(extracted_data)

    text_chunks = text_split(filtered_data)
    print(f"Created text chunks: {len(text_chunks)}")

    index_documents_to_pinecone(
        documents=text_chunks,
        index_name=INDEX_NAME
    )

    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()