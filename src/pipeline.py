"""
SmartDoc-Insight Pipeline Orchestrator
────────────────────────────────────────
Top-level API that wires all 4 layers together.
Import this in the Streamlit app for clean access.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartDocPipeline:
    """
    Unified pipeline orchestrator for SmartDoc-Insight.
    Wraps all 4 layers into a single clean interface.

    Usage:
        pipeline = SmartDocPipeline()
        pipeline.ingest("report.pdf", progress_cb=my_callback)
        response = pipeline.ask("What was Q3 revenue?")
    """

    def __init__(self, config=None):
        from src.config import config as default_config
        from src.models.ollama_client import OllamaClient
        from src.layers.vision_processing import VisionProcessingLayer
        from src.layers.knowledge_base import KnowledgeBaseLayer
        from src.layers.retrieval_reasoning import RetrievalReasoningLayer

        self.config = config or default_config

        logger.info("Initializing SmartDoc-Insight pipeline...")
        self.ollama = OllamaClient(self.config)
        self.vision = VisionProcessingLayer(self.config, self.ollama)
        self.kb = KnowledgeBaseLayer(self.config, self.ollama)
        self.rag = RetrievalReasoningLayer(self.config, self.kb, self.ollama)
        logger.info("Pipeline ready.")

    def ingest(self, file_path: str | Path, progress_callback=None) -> dict:
        """
        Full ingestion pipeline: vision → chunk → embed → store.

        Args:
            file_path: Path to PDF or image file
            progress_callback: Optional callable(stage: str)

        Returns:
            dict with ingestion stats
        """
        file_path = Path(file_path)

        def _vision_progress(page_idx, total, msg):
            if progress_callback:
                progress_callback(f"[Vision] {msg} ({page_idx+1}/{total})")

        def _kb_progress(msg):
            if progress_callback:
                progress_callback(f"[KB] {msg}")

        # Layer A: Vision processing
        if progress_callback:
            progress_callback(f"Starting vision processing for {file_path.name}...")

        processed_doc = self.vision.process_document(file_path, _vision_progress)

        # Layer B: Knowledge base ingestion
        chunk_count = self.kb.ingest_document(processed_doc, _kb_progress)

        # Summarize what was found
        region_summary = {}
        for region in processed_doc.all_regions():
            region_summary[region.region_type] = region_summary.get(region.region_type, 0) + 1

        return {
            "source_file": file_path.name,
            "total_pages": processed_doc.total_pages,
            "total_chunks": chunk_count,
            "regions": region_summary,
        }

    def ask(self, question: str, source_filter: str = None):
        """
        Query the knowledge base and generate an answer.

        Args:
            question: Natural language question
            source_filter: Optional filename to restrict search scope

        Returns:
            RAGResponse with .answer and .sources
        """
        return self.rag.query(question, filter_source=source_filter)

    def ask_stream(self, question: str, source_filter: str = None):
        """Streaming version of ask(). Yields tokens."""
        return self.rag.query_stream(question, filter_source=source_filter)

    def stats(self) -> dict:
        """Return knowledge base stats."""
        return self.kb.get_stats()

    def list_documents(self) -> list:
        """List all indexed documents."""
        return self.kb.list_documents()

    def check_models(self) -> dict:
        """Check which Ollama models are available."""
        return {
            "llm": self.ollama.is_model_available(self.config.ollama.llm_model),
            "vision": self.ollama.is_model_available(self.config.ollama.vision_model),
            "embed": self.ollama.is_model_available(self.config.ollama.embed_model),
            "available_models": self.ollama.list_models(),
        }
