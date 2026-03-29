"""
SmartDoc-Insight Configuration
Central config for all system parameters.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"

for d in [DATA_DIR, CHROMA_DIR, UPLOAD_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Ollama / LLM
@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3:8b")           # Main reasoning LLM
    vision_model: str = os.getenv("VISION_MODEL", "llava:7b")      # Chart/image description
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text") # Embedding model
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# Vision / OCR
@dataclass
class VisionConfig:
    # PaddleOCR settings
    ocr_lang: str = "vi"                        # Vietnamese + English
    ocr_use_gpu: bool = True
    layout_model: str = "picodet_lcnet_x1_0_fgd_layout_cdla"  # CDLA layout model

    # Page rendering DPI (higher = better quality but slower)
    pdf_dpi: int = int(os.getenv("PDF_DPI", "200"))

    # Content type thresholds
    table_confidence: float = 0.7
    figure_confidence: float = 0.6

    # Region types recognized
    REGION_TEXT = "text"
    REGION_TITLE = "title"
    REGION_TABLE = "table"
    REGION_FIGURE = "figure"
    REGION_REFERENCE = "reference"


# ── Chunking Strategy ─────────────────────────────────────────────────────────
@dataclass
class ChunkConfig:
    # Text chunks
    text_chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    text_chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # Tables are never split — stored whole
    table_max_tokens: int = 2048

    # Chart descriptions stored as single chunks
    chart_max_tokens: int = 512

    # Minimum chunk size to avoid noise
    min_chunk_size: int = 50


# ── Vector Database ──────────────────────────────────────────────────────────
@dataclass
class ChromaConfig:
    persist_dir: str = str(CHROMA_DIR)
    collection_name: str = "smartdoc_knowledge"
    distance_metric: str = "cosine"             # cosine | l2 | ip

    # Retrieval
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))           # Initial candidates
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))        # After re-ranking

    # Embedding dimension for nomic-embed-text
    embedding_dim: int = 768


# ── Re-ranking ────────────────────────────────────────────────────────────────
@dataclass
class RerankerConfig:
    # Using a lightweight cross-encoder for re-ranking
    model_name: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 22MB, CPU-friendly
    )
    use_gpu: bool = False   # Keep GPU for LLMs
    batch_size: int = 32


# ── RAG Prompt Templates ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là SmartDoc-Insight, một trợ lý phân tích tài liệu thông minh.
Nhiệm vụ của bạn là trả lời câu hỏi DỰA TRÊN và CHỈ DỰA TRÊN các đoạn thông tin được cung cấp.

Nguyên tắc:
1. Chỉ sử dụng thông tin từ context được cung cấp. Không bịa đặt.
2. Nếu thông tin không đủ, hãy nói rõ "Tài liệu không chứa đủ thông tin để trả lời câu hỏi này."
3. Khi trích dẫn bảng biểu, hãy giải thích rõ ràng các con số.
4. Khi trích dẫn biểu đồ, hãy nêu xu hướng và các điểm dữ liệu quan trọng.
5. Trả lời bằng ngôn ngữ của câu hỏi (Tiếng Việt hoặc Tiếng Anh).
"""

RAG_PROMPT_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>

Dựa vào các đoạn thông tin trong <context>, hãy trả lời câu hỏi trong <question> một cách chính xác và đầy đủ.
Nếu context chứa bảng biểu hoặc mô tả biểu đồ, hãy phân tích và trích dẫn các số liệu cụ thể.
"""

CHART_DESCRIPTION_PROMPT = """Hãy phân tích và mô tả chi tiết biểu đồ/hình ảnh này.
Tập trung vào:
1. Loại biểu đồ (cột, đường, tròn, v.v.)
2. Tiêu đề và nhãn trục (nếu có)
3. Xu hướng chính và các điểm dữ liệu quan trọng
4. So sánh giữa các yếu tố (nếu có)
5. Kết luận hoặc insight chính từ dữ liệu

Trả lời bằng văn bản mô tả ngắn gọn, súc tích, phù hợp để lưu vào cơ sở tri thức.
"""


# ── App Settings ──────────────────────────────────────────────────────────────
@dataclass
class AppConfig:
    app_title: str = "SmartDoc-Insight"
    app_icon: str = "🧠"
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    allowed_extensions: list = field(default_factory=lambda: ["pdf", "png", "jpg", "jpeg"])
    show_sources: bool = True
    show_processing_steps: bool = True


# ── Master Config ─────────────────────────────────────────────────────────────
class Config:
    ollama = OllamaConfig()
    vision = VisionConfig()
    chunk = ChunkConfig()
    chroma = ChromaConfig()
    reranker = RerankerConfig()
    app = AppConfig()
    paths = {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "chroma": CHROMA_DIR,
        "uploads": UPLOAD_DIR,
        "cache": CACHE_DIR,
    }


config = Config()
