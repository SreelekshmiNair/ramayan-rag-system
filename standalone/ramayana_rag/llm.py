from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .config import ModelConfig


BASELINE_SYSTEM_MESSAGE = """You are an expert in Hindu mythology with comprehensive knowledge of the Ramayana.
Provide accurate, relevant, and clearly structured answers.
If you are unsure, say so instead of inventing details."""

RAG_SYSTEM_MESSAGE = """You are a scholarly assistant with deep knowledge of the Ramayana.
Use only the provided context to answer the question.
Do not add claims that are not supported by the context.
If the context does not contain enough information, say that clearly.
End your answer with a short Sources section listing the page numbers you used when available."""


def resolve_model_path(config: ModelConfig) -> str:
    if config.local_model_path:
        return config.local_model_path

    return hf_hub_download(
        repo_id=config.model_repo,
        filename=config.model_filename,
    )


def load_llm(config: ModelConfig) -> Llama:
    model_path = resolve_model_path(config)
    return Llama(
        model_path=model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        n_batch=config.n_batch,
    )


def format_instruct_prompt(system_message: str, user_message: str) -> str:
    return (
        f"[INST]<<SYS>>\n{system_message}\n<</SYS>>\n\n"
        f"{user_message.strip()}[/INST]"
    )
