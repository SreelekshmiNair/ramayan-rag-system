from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: list[Document], config: RAGConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def get_embedding_model(config: RAGConfig) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=config.embedding_model_name)


def build_vectorstore(chunks: list[Document], config: RAGConfig) -> Chroma:
    config.db_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model(config)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.db_dir),
    )


def load_vectorstore(config: RAGConfig) -> Chroma:
    embeddings = get_embedding_model(config)
    return Chroma(
        persist_directory=str(config.db_dir),
        embedding_function=embeddings,
    )
