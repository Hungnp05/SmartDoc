"""
SmartDoc-Insight Test Suite
────────────────────────────
Tests for all major components.
Run: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Minimal config mock for unit tests."""
    from src.config import config
    return config


@pytest.fixture
def mock_ollama():
    """Mock Ollama client that returns dummy responses."""
    client = MagicMock()
    client.query.return_value = "Doanh thu quý 3 tăng 20% so với quý 2."
    client.embed.return_value = [[0.1] * 768]
    client.vision_query.return_value = "Biểu đồ cột cho thấy doanh thu tăng đều qua các quý."
    client.list_models.return_value = ["llama3:8b", "llava:7b", "nomic-embed-text"]
    client.is_model_available.return_value = True
    return client


@pytest.fixture
def sample_table_html():
    return """
    <table>
        <tr><th>Quý</th><th>Doanh thu (tỷ VND)</th><th>Tăng trưởng</th></tr>
        <tr><td>Q1 2024</td><td>1,200</td><td>—</td></tr>
        <tr><td>Q2 2024</td><td>1,450</td><td>+20.8%</td></tr>
        <tr><td>Q3 2024</td><td>1,740</td><td>+20.0%</td></tr>
        <tr><td>Q4 2024</td><td>2,100</td><td>+20.7%</td></tr>
    </table>
    """


@pytest.fixture
def sample_image():
    """Create a simple test image (white background with black text)."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White
    # Add some black region to simulate text
    img[50:150, 50:350] = 50
    return img


# ── Table Extractor Tests ─────────────────────────────────────────────────────

class TestTableExtractor:

    def test_html_to_markdown_basic(self, sample_table_html):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown(sample_table_html)
        assert "Quý" in result
        assert "Doanh thu" in result
        assert "---" in result  # Separator row
        assert "Q3 2024" in result
        assert "|" in result

    def test_html_to_markdown_preserves_numbers(self, sample_table_html):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown(sample_table_html)
        assert "1,740" in result
        assert "+20.0%" in result

    def test_html_to_markdown_empty(self):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown("<table></table>")
        # Should not crash, returns something
        assert isinstance(result, str)

    def test_html_to_markdown_single_row(self):
        from src.utils.table_extractor import html_table_to_markdown
        html = "<table><tr><th>A</th><th>B</th></tr></table>"
        result = html_table_to_markdown(html)
        assert "A" in result
        assert "B" in result


# ── Smart Chunker Tests ───────────────────────────────────────────────────────

class TestSmartChunker:

    def test_text_chunking_basic(self, mock_config):
        from src.layers.knowledge_base import SmartChunker
        chunker = SmartChunker(mock_config)
        long_text = "Đây là một câu văn. " * 100
        chunks = chunker._split_text(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= mock_config.chunk.text_chunk_size * 1.2  # Some tolerance

    def test_short_text_not_split(self, mock_config):
        from src.layers.knowledge_base import SmartChunker
        chunker = SmartChunker(mock_config)
        short_text = "Doanh thu tăng 20%."
        chunks = chunker._split_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_table_stored_as_single_chunk(self, mock_config, mock_ollama):
        """Tables should never be split — stored as one chunk."""
        from src.layers.knowledge_base import SmartChunker
        from src.layers.vision_processing import ProcessedDocument, ProcessedPage, DocumentRegion, BoundingBox

        chunker = SmartChunker(mock_config)

        # Create a mock document with one table region
        region = DocumentRegion(
            region_type="table",
            bbox=BoundingBox(0, 0, 100, 100),
            confidence=0.9,
            page_num=0,
            content="| Q1 | Q2 | Q3 |\n|---|---|---|\n| 100 | 120 | 144 |"
        )
        page = ProcessedPage(page_num=0, page_summary="Quarterly data", regions=[region])
        doc = ProcessedDocument(source_file="test.pdf", total_pages=1, pages=[page])

        chunks = chunker.chunk_document(doc)
        # Table should result in exactly 1 chunk (not split)
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
        assert len(table_chunks) == 1
        assert "Q1" in table_chunks[0].text
        assert "Q3" in table_chunks[0].text

    def test_chunk_metadata_populated(self, mock_config, mock_ollama):
        """Every chunk must have required metadata fields."""
        from src.layers.knowledge_base import SmartChunker
        from src.layers.vision_processing import ProcessedDocument, ProcessedPage, DocumentRegion, BoundingBox

        chunker = SmartChunker(mock_config)
        region = DocumentRegion(
            region_type="text", bbox=BoundingBox(0, 0, 100, 50),
            confidence=0.95, page_num=2,
            content="Báo cáo tài chính năm 2024."
        )
        page = ProcessedPage(page_num=2, page_summary="Financial report", regions=[region])
        doc = ProcessedDocument(source_file="annual.pdf", total_pages=10, pages=[page])

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "page" in chunk.metadata
            assert "content_type" in chunk.metadata
            assert chunk.metadata["source_file"] == "annual.pdf"
            assert chunk.metadata["page"] == 3  # 1-indexed


# ── Retrieval & Reasoning Tests ───────────────────────────────────────────────

class TestRetrievalReasoning:

    def test_context_builder_basic(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="1", text="Doanh thu Q3 là 1,740 tỷ đồng.",
                    metadata={"page": 2, "content_type": "table", "source_file": "report.pdf", "page_summary": "Revenue data"}
                ),
                score=0.92
            )
        ]
        context = ContextBuilder.build(chunks, "Doanh thu Q3?")
        assert "1,740" in context
        assert "table" in context.lower()
        assert "Trang 2" in context

    def test_context_builder_empty(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        context = ContextBuilder.build([], "test query")
        assert "không tìm thấy" in context.lower() or len(context) < 100

    def test_source_conversion(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="2", text="Biểu đồ tăng trưởng.",
                    metadata={"page": 3, "content_type": "figure", "source_file": "report.pdf", "page_summary": "Charts"}
                ),
                score=0.85
            )
        ]
        sources = ContextBuilder.chunks_to_sources(chunks)
        assert len(sources) == 1
        assert sources[0].content_type == "figure"
        assert sources[0].page == 3
        assert sources[0].score == 0.85

    def test_source_type_emoji(self):
        from src.layers.retrieval_reasoning import Source
        s_table = Source("content", "table", 1, "file.pdf", 0.9)
        s_figure = Source("content", "figure", 1, "file.pdf", 0.9)
        s_text = Source("content", "text", 1, "file.pdf", 0.9)
        assert s_table.type_emoji == "📊"
        assert s_figure.type_emoji == "📈"
        assert s_text.type_emoji == "📝"


# ── Cross-Encoder Re-ranker Tests ─────────────────────────────────────────────

class TestReranker:

    def test_rerank_fewer_than_topk(self, mock_config):
        """If fewer chunks than top_k, return all without re-ranking."""
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        reranker = CrossEncoderReranker(mock_config)
        chunks = [
            RetrievedChunk(chunk=Chunk("1", "text", {}), score=0.8),
            RetrievedChunk(chunk=Chunk("2", "text", {}), score=0.7),
        ]
        result = reranker.rerank("query", chunks, top_k=5)
        assert len(result) == 2  # All returned since fewer than top_k

    def test_score_based_rerank_fallback(self, mock_config):
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        reranker = CrossEncoderReranker(mock_config)
        chunks = [
            RetrievedChunk(chunk=Chunk("1", "a", {}), score=0.5),
            RetrievedChunk(chunk=Chunk("2", "b", {}), score=0.9),
            RetrievedChunk(chunk=Chunk("3", "c", {}), score=0.7),
            RetrievedChunk(chunk=Chunk("4", "d", {}), score=0.3),
            RetrievedChunk(chunk=Chunk("5", "e", {}), score=0.8),
        ]
        result = reranker._score_based_rerank(chunks, top_k=3)
        assert len(result) == 3
        assert result[0].score == 0.9  # Highest first
        assert result[1].score == 0.8


# ── Integration-style test (no Ollama needed) ─────────────────────────────────

class TestIntegration:

    def test_pipeline_import(self):
        """Ensure pipeline can be imported without errors."""
        from src.pipeline import SmartDocPipeline
        assert SmartDocPipeline is not None

    def test_config_import(self):
        from src.config import config
        assert config.ollama.llm_model == "llama3:8b"
        assert config.ollama.vision_model == "llava:7b"
        assert config.chroma.top_k == 10
        assert config.chroma.rerank_top_k == 4

    def test_chunk_size_config(self):
        from src.config import config
        assert config.chunk.text_chunk_size > 0
        assert config.chunk.text_chunk_overlap < config.chunk.text_chunk_size

    def test_data_dirs_created(self):
        from src.config import CHROMA_DIR, UPLOAD_DIR, CACHE_DIR
        assert CHROMA_DIR.exists()
        assert UPLOAD_DIR.exists()
        assert CACHE_DIR.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
