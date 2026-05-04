import os
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def load_environment_variables():
    """
    Load required environment variables from .env file.
    """
    load_dotenv(override=True)

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing. Please add it to your .env file.")

    return pinecone_api_key


def get_pinecone_client():
    """
    Create Pinecone client.
    """
    pinecone_api_key = load_environment_variables()
    return Pinecone(api_key=pinecone_api_key)


def create_pinecone_index_if_not_exists(index_name=INDEX_NAME):
    """
    Create Pinecone index only if it does not already exist.
    Used by store_index.py.
    """
    pc = get_pinecone_client()

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        print(f"Pinecone index created: {index_name}")
    else:
        print(f"Pinecone index already exists: {index_name}")

    return pc


def load_pdf_file(data):
    """
    Load PDF files from the given data directory.
    Used by store_index.py during indexing.
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only source metadata and original page content.
    This keeps Pinecone metadata lightweight.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


def text_split(extracted_data):
    """
    Split loaded PDF documents into smaller text chunks.
    Used by store_index.py during indexing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    """
    Load HuggingFace embedding model.
    This model returns 384-dimensional embeddings.
    Pinecone index dimension must be 384.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def index_documents_to_pinecone(
    documents,
    index_name=INDEX_NAME
):
    """
    Upload document chunks to Pinecone.
    Used by store_index.py.
    Do not call this from app.py.
    """
    embeddings = download_hugging_face_embeddings()

    test_vector = embeddings.embed_query("test")

    if len(test_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {len(test_vector)}"
        )

    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )

    return docsearch


def get_pinecone_vectorstore(index_name=INDEX_NAME):
    """
    Connect to an existing Pinecone index.
    Used by app.py for retrieval.
    This function does not upload or re-index documents.
    """
    pc = get_pinecone_client()

    if not pc.has_index(index_name):
        raise ValueError(
            f"Pinecone index '{index_name}' does not exist. Run store_index.py first."
        )

    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    return docsearch


def get_retriever(index_name=INDEX_NAME, k=3):
    """
    Create retriever from existing Pinecone vector store.
    Used by app.py for RAG question answering.
    """
    docsearch = get_pinecone_vectorstore(index_name=index_name)

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever
