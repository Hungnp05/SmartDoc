"""
SmartDoc-Insight Configuration
Central config for all system parameters.
"""
from dotenv import load_dotenv
load_dotenv(override=True)
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


# Chunking Strategy 
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


# Vector Database
@dataclass
class ChromaConfig:
    persist_dir: str = str(CHROMA_DIR)
    collection_name: str = "smartdoc_knowledge"
    distance_metric: str = "cosine"

    # Retrieval
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))

    # Embedding dimension for nomic-embed-text
    embedding_dim: int = 768


# Re-ranking
@dataclass
class RerankerConfig:
    model_name: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    use_gpu: bool = False
    batch_size: int = 32


# RAG Prompt Templates
SYSTEM_PROMPT = """You are SmartDoc-Insight, an intelligent document analysis assistant.
Your task is to answer questions BASED ON and ONLY BASED ON the provided information snippets.

Principles:
1. Use only the information from the provided context. Do not hallucinate or make things up.
2. If the information is insufficient, clearly state it in the user's language.
3. When citing tables, read by ROWS — each row is a distinct record. Do not mix up values between columns.
4. When citing charts, describe trends and key data points.
5. If uncertain about a figure, specify the source: 'According to Table X, Page Y...'.
6. CRITICAL — Language rule: You MUST reply in the SAME language as the question. If the question is in Vietnamese, your ENTIRE answer must be in Vietnamese. If in English, answer in English. This rule overrides everything else.
"""

RAG_PROMPT_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>

<language_instruction>
{language_instruction}
</language_instruction>

Answer the question using ONLY the information in <context>. Follow the language rule in <language_instruction> strictly.
If the context contains tables or chart descriptions, analyze and cite specific figures.
"""

CHART_DESCRIPTION_PROMPT = """Please analyze and describe this chart/image in detail.
Focus on:
1. Chart type (bar, line, pie, etc.)
2. Title and axis labels (if available)
3. Main trends and key data points
4. Comparisons between elements (if any)
5. Main conclusions or insights from the data

Provide a concise descriptive response suitable for storage in a knowledge base.
"""

# App Settings
@dataclass
class AppConfig:
    app_title: str = "SmartDoc-Insight"
    app_icon: str = "🧠"
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    allowed_extensions: list = field(default_factory=lambda: ["pdf", "png", "jpg", "jpeg"])
    show_sources: bool = True
    show_processing_steps: bool = True


# Master Config
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
