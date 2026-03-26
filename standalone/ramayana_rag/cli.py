import argparse
from pathlib import Path

from .config import ModelConfig, RAGConfig
from .ingest import build_vectorstore, load_pdf_documents, load_vectorstore, split_documents
from .llm import load_llm
from .rag import baseline_answer, rag_answer, summarize_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project RAMAYANA RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pdf", type=Path, required=True, help="Path to a Ramayana PDF")
    common.add_argument(
        "--db-dir",
        type=Path,
        default=Path("ramayana_db"),
        help="Directory for the persisted Chroma database",
    )
    common.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name",
    )
    common.add_argument("--chunk-size", type=int, default=512)
    common.add_argument("--chunk-overlap", type=int, default=50)
    common.add_argument("--model-path", help="Local GGUF model path")
    common.add_argument(
        "--model-repo",
        default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        help="Hugging Face repo for the GGUF model",
    )
    common.add_argument(
        "--model-file",
        default="mistral-7b-instruct-v0.2.Q6_K.gguf",
        help="GGUF filename inside the model repo",
    )
    common.add_argument("--n-gpu-layers", type=int, default=38)
    common.add_argument("--n-ctx", type=int, default=2300)
    common.add_argument("--n-batch", type=int, default=512)
    common.add_argument("--max-tokens", type=int, default=256)
    common.add_argument("--temperature", type=float, default=0.0)
    common.add_argument("--top-p", type=float, default=0.95)
    common.add_argument("--top-k-sampling", type=int, default=50)

    index_parser = subparsers.add_parser(
        "index",
        parents=[common],
        help="Load a PDF and build the local Chroma index",
    )
    index_parser.add_argument("--show-stats", action="store_true")

    ask_parser = subparsers.add_parser(
        "ask",
        parents=[common],
        help="Ask a question with RAG grounding",
    )
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--retrieval-k", type=int, default=3)
    ask_parser.add_argument("--show-sources", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare",
        parents=[common],
        help="Compare baseline and RAG answers for one question",
    )
    compare_parser.add_argument("--question", required=True)
    compare_parser.add_argument("--retrieval-k", type=int, default=3)
    compare_parser.add_argument("--show-sources", action="store_true")

    return parser


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, RAGConfig]:
    model_config = ModelConfig(
        model_repo=args.model_repo,
        model_filename=args.model_file,
        local_model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
    )
    rag_config = RAGConfig(
        pdf_path=args.pdf,
        db_dir=args.db_dir,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k_sampling=args.top_k_sampling,
    )
    return model_config, rag_config


def run_index(rag_config: RAGConfig, show_stats: bool) -> None:
    documents = load_pdf_documents(rag_config.pdf_path)
    chunks = split_documents(documents, rag_config)
    build_vectorstore(chunks, rag_config)
    print(f"Indexed {len(documents)} pages into {len(chunks)} chunks.")
    print(f"Saved vector store to: {rag_config.db_dir}")
    if show_stats and chunks:
        print(f"First chunk preview:\n{chunks[0].page_content[:500]}")


def run_ask(
    model_config: ModelConfig,
    rag_config: RAGConfig,
    question: str,
    retrieval_k: int,
    show_sources: bool,
) -> None:
    llm = load_llm(model_config)
    vectorstore = load_vectorstore(rag_config)
    answer, documents = rag_answer(
        llm,
        vectorstore,
        question,
        retrieval_k=retrieval_k,
        max_tokens=rag_config.max_tokens,
        temperature=rag_config.temperature,
        top_p=rag_config.top_p,
        top_k=rag_config.top_k_sampling,
    )
    print("RAG Answer:\n")
    print(answer)
    if show_sources:
        print("\nRetrieved Sources:\n")
        print(summarize_sources(documents))


def run_compare(
    model_config: ModelConfig,
    rag_config: RAGConfig,
    question: str,
    retrieval_k: int,
    show_sources: bool,
) -> None:
    llm = load_llm(model_config)
    vectorstore = load_vectorstore(rag_config)
    baseline = baseline_answer(
        llm,
        question,
        max_tokens=rag_config.max_tokens,
        temperature=max(rag_config.temperature, 0.2),
        top_p=rag_config.top_p,
        top_k=rag_config.top_k_sampling,
    )
    rag_response, documents = rag_answer(
        llm,
        vectorstore,
        question,
        retrieval_k=retrieval_k,
        max_tokens=rag_config.max_tokens,
        temperature=rag_config.temperature,
        top_p=rag_config.top_p,
        top_k=rag_config.top_k_sampling,
    )
    print("Baseline Answer:\n")
    print(baseline)
    print("\nRAG Answer:\n")
    print(rag_response)
    if show_sources:
        print("\nRetrieved Sources:\n")
        print(summarize_sources(documents))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_config, rag_config = build_configs(args)

    if args.command == "index":
        run_index(rag_config, args.show_stats)
        return

    if not rag_config.db_dir.exists():
        parser.error(
            f"Vector store directory '{rag_config.db_dir}' does not exist. Run the index command first."
        )

    if args.command == "ask":
        run_ask(
            model_config,
            rag_config,
            args.question,
            args.retrieval_k,
            args.show_sources,
        )
        return

    if args.command == "compare":
        run_compare(
            model_config,
            rag_config,
            args.question,
            args.retrieval_k,
            args.show_sources,
        )


if __name__ == "__main__":
    main()
