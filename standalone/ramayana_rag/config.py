from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    model_repo: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_filename: str = "mistral-7b-instruct-v0.2.Q6_K.gguf"
    local_model_path: str | None = None
    n_ctx: int = 2300
    n_gpu_layers: int = 38
    n_batch: int = 512


@dataclass(slots=True)
class RAGConfig:
    pdf_path: Path
    db_dir: Path = Path("ramayana_db")
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95
    top_k_sampling: int = 50
