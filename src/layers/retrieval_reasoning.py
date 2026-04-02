"""
Layer C: Retrieval & Reasoning Layer
──────────────────────────────────────
Implements the full RAG query pipeline:
  1. Retrieve  → ChromaDB vector similarity search
  2. Re-rank   → Cross-encoder for precision improvement
  3. Assemble  → Build structured context for LLM
  4. Generate  → Llama3-8B produces the final answer
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, Generator, Any, List
from collections.abc import Generator

logger = logging.getLogger(__name__)


# Data Models

@dataclass
class Source:
    """A source citation in the final answer."""
    content: str
    content_type: str
    page: int
    source_file: str
    score: float

    @property
    def type_emoji(self):
        return {"table": "📊", "figure": "📈", "title": "📌", "text": "📝"}.get(
            self.content_type, "📄"
        )

    @property
    def display_label(self):
        return f"{self.type_emoji} [{self.content_type.upper()}] Trang {self.page}"


@dataclass
class RAGResponse:
    """Complete RAG response with answer and citations."""
    answer: str
    sources: list[Source]
    query: str
    retrieved_count: int
    reranked_count: int


# Cross-Encoder Re-ranker

class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a lightweight cross-encoder.
    Cross-encoders compare (query, document) pairs jointly for higher precision.
    Model: ms-marco-MiniLM-L-6-v2 (~22MB, CPU, fast)
    """

    def __init__(self, config):
        self.model_name = config.reranker.model_name
        self.use_gpu = config.reranker.use_gpu
        self._model = None

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        """
        Re-rank chunks by relevance to query.
        Returns top_k chunks sorted by cross-encoder score (descending).
        """
        if not chunks:
            return []

        if len(chunks) <= top_k:
            return chunks

        try:
            return self._cross_encoder_rerank(query, chunks, top_k)
        except ImportError:
            logger.warning("sentence-transformers not available, using score-based ranking")
            return self._score_based_rerank(chunks, top_k)

    def _cross_encoder_rerank(self, query: str, chunks: list, top_k: int) -> list:
        """Use cross-encoder model for re-ranking."""
        from sentence_transformers import CrossEncoder

        if self._model is None:
            device = "cuda" if self.use_gpu else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Cross-encoder loaded: {self.model_name}")

        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        # Combine original similarity score with cross-encoder score
        for chunk, ce_score in zip(chunks, scores):
            chunk.rerank_score = float(ce_score) * 0.7 + chunk.score * 0.3

        sorted_chunks = sorted(chunks, key=lambda c: getattr(c, "rerank_score", c.score), reverse=True)
        return sorted_chunks[:top_k]

    @staticmethod
    def _score_based_rerank(chunks: list, top_k: int) -> list:
        """Fallback: just return top-k by original similarity score."""
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


# Context Builder

class ContextBuilder:
    """
    Assembles retrieved chunks into a structured context string for the LLM.
    Prioritizes tables and figures, organizes by page order.
    """

    @staticmethod
    def build(chunks: list, query: str) -> str:
        """Build a rich context string from retrieved chunks."""
        if not chunks:
            return "Không tìm thấy thông tin liên quan trong tài liệu."

        # Sort by page number for logical reading order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk.metadata.get("page", 0))

        parts = []
        seen_tables = set()  # Avoid duplicate table content

        for i, rc in enumerate(sorted_chunks, 1):
            meta = rc.chunk.metadata
            page = meta.get("page", "?")
            content_type = meta.get("content_type", "text")
            page_summary = meta.get("page_summary", "")
            source = meta.get("source_file", "")

            # Deduplicate table chunks (same page + same start content)
            if content_type == "table":
                table_key = f"{page}:{rc.chunk.text[:50]}"
                if table_key in seen_tables:
                    continue
                seen_tables.add(table_key)

            header = f"--- Đoạn {i} | Trang {page} | Loại: {content_type.upper()} | Nguồn: {source} ---"
            if page_summary:
                header += f"\n[Tóm tắt trang: {page_summary}]"

            parts.append(f"{header}\n{rc.chunk.text}")

        return "\n\n".join(parts)

    @staticmethod
    def chunks_to_sources(chunks: list) -> list[Source]:
        """Convert retrieved chunks to Source citations."""
        sources = []
        for rc in chunks:
            meta = rc.chunk.metadata
            sources.append(Source(
                content=rc.chunk.text[:500] + ("..." if len(rc.chunk.text) > 500 else ""),
                content_type=meta.get("content_type", "text"),
                page=meta.get("page", 0),
                source_file=meta.get("source_file", "unknown"),
                score=rc.score,
            ))
        return sources


# Retrieval & Reasoning Layer

class RetrievalReasoningLayer:
    """
    Layer C: Orchestrates the full RAG pipeline.
    retrieve → rerank → build_context → generate
    """

    def __init__(self, config, knowledge_base, ollama_client):
        self.cfg = config
        self.kb = knowledge_base
        self.ollama = ollama_client
        self.reranker = CrossEncoderReranker(config)
        self.context_builder = ContextBuilder()

    # Main Query Interface

    def query(self, question: str, filter_source: Optional[str] = None) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → rerank → generate.
        Returns a complete RAGResponse with answer and sources.
        """
        # Step 1: Retrieve initial candidates
        raw_chunks = self.kb.retrieve(
            query=question,
            top_k=self.cfg.chroma.top_k,
            filter_source=filter_source,
        )
        logger.info(f"Retrieved {len(raw_chunks)} initial chunks")

        # Step 2: Re-rank for precision
        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=raw_chunks,
            top_k=self.cfg.chroma.rerank_top_k,
        )
        logger.info(f"Re-ranked to {len(reranked_chunks)} chunks")

        # Step 3: Build context
        context = self.context_builder.build(reranked_chunks, question)

        # Step 4: Generate answer
        answer = self._generate_answer(question, context)

        # Step 5: Assemble response
        sources = self.context_builder.chunks_to_sources(reranked_chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieved_count=len(raw_chunks),
            reranked_count=len(reranked_chunks),
        )

    def query_stream(self, question: str, filter_source: Optional[str] = None) -> Generator[str, None, list[dict]]:
        """
        Streaming version of query. Yields tokens as they're generated by the LLM.
        
        Sau khi stream xong, hàm sẽ return danh sách sources (citations) 
        để đảm bảo tính minh bạch của hệ thống RAG.
        
        Returns:
            Generator[str, None, list[dict]]: Generator yield từng token,
            và return sources khi generator kết thúc.
        """
        from src.config import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
        raw_chunks = self.kb.retrieve(
            query=question,
            top_k=self.cfg.chroma.top_k,
            filter_source=filter_source,
        )
        
        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=raw_chunks,
            top_k=self.cfg.chroma.rerank_top_k,
        )

        context = self.context_builder.build(reranked_chunks, question)

        language = self._detect_language(question)
        lang_instruction = self._get_language_instruction(language)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            language_instruction=lang_instruction,
        )
        for token in self.ollama.stream(prompt=prompt, system=SYSTEM_PROMPT):
            yield token

        return self.context_builder.chunks_to_sources(reranked_chunks)


    # Answer Generation
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Call LLM to generate the final answer based on retrieved context.
        
        Sử dụng language detection để đưa ra chỉ dẫn ngôn ngữ phù hợp.
        """
        from src.config import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

        language = self._detect_language(question)
        lang_instruction = self._get_language_instruction(language)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            language_instruction=lang_instruction,
        )

        return self.ollama.query(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.cfg.ollama.temperature,
            max_tokens=self.cfg.ollama.max_tokens,
        )

    # Convenience Methods
    def get_retrieval_debug(self, question: str) -> dict:
        """Debug info: show retrieved chunks and their scores."""
        raw = self.kb.retrieve(question, top_k=self.cfg.chroma.top_k)
        reranked = self.reranker.rerank(question, raw, self.cfg.chroma.rerank_top_k)

        return {
            "query": question,
            "initial_retrieval": [
                {
                    "text": c.chunk.text[:200],
                    "score": round(c.score, 4),
                    "type": c.chunk.metadata.get("content_type"),
                    "page": c.chunk.metadata.get("page"),
                }
                for c in raw
            ],
            "after_reranking": [
                {
                    "text": c.chunk.text[:200],
                    "original_score": round(c.score, 4),
                    "rerank_score": round(getattr(c, "rerank_score", c.score), 4),
                    "type": c.chunk.metadata.get("content_type"),
                    "page": c.chunk.metadata.get("page"),
                }
                for c in reranked
            ],
        }

    def _detect_language(self, text: str) -> str:
        vietnamese_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        vietnamese_chars += vietnamese_chars.upper()
        viet_count = sum(1 for c in text if c in vietnamese_chars)
        if viet_count >= 2:
            return "vietnamese"
        viet_words = ["là","và","của","có","được","không","này","cho","với","trong","một","các","tôi","bạn","hãy","theo","như","khi","về","từ","tại","bao","nhiêu","gì","nào","sao","tại sao"]
        text_lower = text.lower()
        if any(w in text_lower for w in viet_words):
            return "vietnamese"
        return "english"

    def _get_language_instruction(self, language: str) -> str:
        if language == "vietnamese":
            return "Câu hỏi được viết bằng TIẾNG VIỆT. Bạn BẮT BUỘC phải trả lời HOÀN TOÀN bằng tiếng Việt. Không được dùng bất kỳ từ tiếng Anh nào trong câu trả lời (trừ tên riêng và thuật ngữ kỹ thuật không có từ tương đương)."
        return "The question is in ENGLISH. You MUST answer entirely in English."
    
    