import re
import logging
from dataclasses import dataclass
from typing import Optional, Generator

logger = logging.getLogger(__name__)


@dataclass
class Source:
    content: str
    content_type: str
    page: int
    source_file: str
    score: float

    @property
    def type_label(self):
        return {"table": "TABLE", "figure": "FIGURE", "title": "TITLE", "text": "TEXT"}.get(
            self.content_type, "UNKNOWN"
        )

    @property
    def display_label(self):
        return f"[{self.type_label}] Page {self.page}"


@dataclass
class RAGResponse:
    answer: str
    sources: list
    query: str
    retrieved_count: int
    reranked_count: int


class CrossEncoderReranker:
    def __init__(self, config):
        self.model_name = config.reranker.model_name
        self.use_gpu = config.reranker.use_gpu
        self._model = None

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        if not chunks:
            return []
        if len(chunks) <= top_k:
            return sorted(chunks, key=lambda c: c.score, reverse=True)
        try:
            return self._cross_encoder_rerank(query, chunks, top_k)
        except ImportError:
            logger.warning("sentence-transformers not available, using score ranking")
            return self._score_based_rerank(chunks, top_k)

    def _cross_encoder_rerank(self, query: str, chunks: list, top_k: int) -> list:
        from sentence_transformers import CrossEncoder
        if self._model is None:
            device = "cuda" if self.use_gpu else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Cross-encoder loaded: {self.model_name}")

        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        for chunk, ce_score in zip(chunks, scores):
            chunk.rerank_score = float(ce_score) * 0.7 + chunk.score * 0.3

        return sorted(chunks, key=lambda c: getattr(c, "rerank_score", c.score), reverse=True)[:top_k]

    @staticmethod
    def _score_based_rerank(chunks: list, top_k: int) -> list:
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


class ContextBuilder:
    @staticmethod
    def build(chunks: list, query: str) -> str:
        if not chunks:
            return "No relevant information found in the document."

        tables_and_figures = [c for c in chunks if c.chunk.metadata.get("content_type") in ("table", "figure")]
        text_chunks = [c for c in chunks if c.chunk.metadata.get("content_type") not in ("table", "figure")]

        ordered = sorted(tables_and_figures, key=lambda c: c.chunk.metadata.get("page", 0))
        ordered += sorted(text_chunks, key=lambda c: c.chunk.metadata.get("page", 0))

        parts = []
        seen = set()

        for i, rc in enumerate(ordered, 1):
            meta = rc.chunk.metadata
            page = meta.get("page", "?")
            content_type = meta.get("content_type", "text")
            page_summary = meta.get("page_summary", "")
            source = meta.get("source_file", "")

            dedup_key = f"{page}:{rc.chunk.text[:80]}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            header = (
                f"--- Snippet {i} | Page {page} | Type: {content_type.upper()} "
                f"| Source: {source} ---"
            )
            if page_summary:
                header += f"\n[Page summary: {page_summary}]"

            if content_type == "table":
                header += "\n[READING INSTRUCTION: Read this table ROW BY ROW. Each row is one record. Do not mix values across rows or columns.]"

            parts.append(f"{header}\n{rc.chunk.text}")

        return "\n\n".join(parts)

    @staticmethod
    def chunks_to_sources(chunks: list) -> list:
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


class RetrievalReasoningLayer:
    def __init__(self, config, knowledge_base, ollama_client):
        self.cfg = config
        self.kb = knowledge_base
        self.ollama = ollama_client
        self.reranker = CrossEncoderReranker(config)
        self.context_builder = ContextBuilder()

    def query(self, question: str, filter_source: Optional[str] = None) -> RAGResponse:
        raw_chunks = self.kb.retrieve(
            query=question,
            top_k=self.cfg.chroma.top_k,
            filter_source=filter_source,
        )
        logger.info(f"Retrieved {len(raw_chunks)} initial chunks")

        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=raw_chunks,
            top_k=self.cfg.chroma.rerank_top_k,
        )
        logger.info(f"Reranked to {len(reranked_chunks)} chunks")

        context = self.context_builder.build(reranked_chunks, question)
        answer = self._generate_answer(question, context)
        sources = self.context_builder.chunks_to_sources(reranked_chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieved_count=len(raw_chunks),
            reranked_count=len(reranked_chunks),
        )

    def query_stream(self, question: str, filter_source: Optional[str] = None) -> Generator:
        from src.config import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
        raw_chunks = self.kb.retrieve(question, top_k=self.cfg.chroma.top_k, filter_source=filter_source)
        reranked_chunks = self.reranker.rerank(question, raw_chunks, self.cfg.chroma.rerank_top_k)
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

    def _generate_answer(self, question: str, context: str) -> str:
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

    def _detect_language(self, text: str) -> str:
        vietnamese_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        vietnamese_chars += vietnamese_chars.upper()
        if sum(1 for c in text if c in vietnamese_chars) >= 2:
            return "vietnamese"
        viet_words = [
            "là","và","của","có","được","không","này","cho","với","trong",
            "một","các","tôi","bạn","hãy","theo","như","khi","về","từ",
            "tại","bao","nhiêu","gì","nào","sao","tại sao","doanh thu",
            "bảng","biểu","quý","năm","tăng","giảm","tổng","chi phí",
        ]
        text_lower = text.lower()
        if any(w in text_lower for w in viet_words):
            return "vietnamese"
        return "english"

    def _get_language_instruction(self, language: str) -> str:
        if language == "vietnamese":
            return (
                "Câu hỏi được viết bằng TIẾNG VIỆT. "
                "Bạn BẮT BUỘC phải trả lời HOÀN TOÀN bằng tiếng Việt. "
                "Không được dùng câu tiếng Anh trong phần trả lời. "
                "Tên riêng và số liệu giữ nguyên, nhưng tất cả văn bản giải thích phải bằng tiếng Việt."
            )
        return (
            "The question is in ENGLISH. "
            "You MUST answer entirely in English."
        )

    def get_retrieval_debug(self, question: str) -> dict:
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