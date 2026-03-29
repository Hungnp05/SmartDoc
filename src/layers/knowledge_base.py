"""
Layer B: Knowledge Base Layer
──────────────────────────────
Manages the vector store with enriched metadata:
  - Smart chunking that respects document structure
  - Hybrid metadata: page, content_type, page_summary, source
  - ChromaDB with cosine similarity
  - Embedding via nomic-embed-text (Ollama)
"""

import uuid
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# Data Models 

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict
    embedding: Optional[list[float]] = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# Smart Chunker

class SmartChunker:
    """
    Chunking strategy that preserves document structure:
    - Tables → stored as single chunks (never split)
    - Figures → stored as single chunks
    - Text/Title → split with overlap, respecting sentence boundaries
    """

    def __init__(self, config):
        self.cfg = config.chunk

    def chunk_document(self, processed_doc) -> list[Chunk]:
        """Convert a ProcessedDocument into a flat list of Chunks."""
        chunks = []

        for page in processed_doc.pages:
            page_meta_base = {
                "source_file": processed_doc.source_file,
                "page": page.page_num + 1,
                "page_summary": page.page_summary,
                "total_pages": processed_doc.total_pages,
            }

            for region in page.regions:
                if not region.content or len(region.content.strip()) < self.cfg.min_chunk_size:
                    continue

                region_meta = {
                    **page_meta_base,
                    "content_type": region.region_type,
                    "confidence": region.confidence,
                    "bbox": f"{region.bbox.x1},{region.bbox.y1},{region.bbox.x2},{region.bbox.y2}",
                }

                if region.region_type in ("table", "figure"):
                    # Structural content: keep whole, add type prefix
                    chunks.append(self._make_chunk(
                        text=self._format_structured_content(region),
                        metadata=region_meta,
                    ))
                else:
                    # Text/title: split with overlap
                    text_chunks = self._split_text(region.content)
                    for i, text in enumerate(text_chunks):
                        chunks.append(self._make_chunk(
                            text=text,
                            metadata={**region_meta, "chunk_index": i},
                        ))

        logger.info(f"Chunked '{processed_doc.source_file}' → {len(chunks)} chunks")
        return chunks

    def _format_structured_content(self, region) -> str:
        """Add context prefix to tables and figures for better retrieval."""
        type_labels = {
            "table": "📊 BẢNG DỮ LIỆU",
            "figure": "📈 MÔ TẢ BIỂU ĐỒ",
        }
        label = type_labels.get(region.region_type, region.region_type.upper())
        return f"[{label} - Trang {region.page_num + 1}]\n\n{region.content}"

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks, trying to respect sentence boundaries.
        """
        size = self.cfg.text_chunk_size
        overlap = self.cfg.text_chunk_overlap

        if len(text) <= size:
            return [text]

        chunks = []
        # Try to split on newlines or periods first
        sentences = self._sentence_split(text)
        current = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > size and current:
                chunks.append(" ".join(current))
                # Overlap: keep last few sentences
                overlap_tokens = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= overlap:
                        overlap_tokens.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_tokens
                current_len = overlap_len

            current.append(sentence)
            current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if len(c.strip()) >= self.cfg.min_chunk_size]

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Simple sentence splitter."""
        import re
        parts = re.split(r'(?<=[.!?।\n])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _make_chunk(text: str, metadata: dict) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            metadata=metadata,
        )


# Embedding Handle

class EmbeddingHandler:
    """Wrapper around Ollama's nomic-embed-text for generating embeddings."""

    def __init__(self, ollama_client, config):
        self.ollama = ollama_client
        self.model = config.ollama.embed_model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.ollama.embed(batch, model=self.model)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = self.ollama.embed([query], model=self.model)
        return results[0]


# Knowledge Base Layer

class KnowledgeBaseLayer:
    """
    Layer B: Manages ChromaDB vector store with enriched document chunks.
    """

    def __init__(self, config, ollama_client):
        self.cfg = config
        self.chunker = SmartChunker(config)
        self.embedder = EmbeddingHandler(ollama_client, config)
        self._collection = None
        self._init_chroma()

    # Initialization

    def _init_chroma(self):
        """Initialize ChromaDB persistent client."""
        client = chromadb.PersistentClient(
            path=self.cfg.chroma.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = client.get_or_create_collection(
            name=self.cfg.chroma.collection_name,
            metadata={"hnsw:space": self.cfg.chroma.distance_metric},
        )
        logger.info(f"ChromaDB initialized: {self._collection.count()} existing chunks")

    # Ingestion

    def ingest_document(self, processed_doc, progress_callback=None) -> int:
        """
        Chunk → embed → store a processed document.
        Returns the number of chunks added.
        """
        # Step 1: Chunk
        if progress_callback:
            progress_callback("Đang tạo chunks thông minh...")
        chunks = self.chunker.chunk_document(processed_doc)

        # Step 2: Check for duplicates (by source file)
        self._remove_existing(processed_doc.source_file)

        # Step 3: Embed in batches
        if progress_callback:
            progress_callback(f"Đang tạo embeddings cho {len(chunks)} chunks...")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Step 4: Store in ChromaDB
        if progress_callback:
            progress_callback("Đang lưu vào Vector Database...")

        self._store_chunks(chunks)

        logger.info(f"Ingested {len(chunks)} chunks from '{processed_doc.source_file}'")
        return len(chunks)

    def _remove_existing(self, source_file: str):
        """Remove all chunks from a previously ingested file."""
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} existing chunks for '{source_file}'")
        except Exception as e:
            logger.warning(f"Could not remove existing chunks: {e}")

    def _store_chunks(self, chunks: list[Chunk], batch_size: int = 100):
        """Store chunks in ChromaDB in batches."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[c.embedding for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    # Retrieval

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 filter_source: Optional[str] = None) -> list[RetrievedChunk]:
        """
        Retrieve top-k most relevant chunks via cosine similarity.
        """
        k = top_k or self.cfg.chroma.top_k
        query_embedding = self.embedder.embed_query(query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_source:
            query_kwargs["where"] = {"source_file": filter_source}

        results = self._collection.query(**query_kwargs)

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1]
            similarity = 1 - (dist / 2)

            chunk = Chunk(
                chunk_id=results["ids"][0][len(retrieved)],
                text=doc,
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

        return retrieved

    # Utility

    def list_documents(self) -> list[dict]:
        """Return metadata about all indexed documents."""
        try:
            results = self._collection.get(include=["metadatas"])
            docs = {}
            for meta in results["metadatas"]:
                src = meta.get("source_file", "unknown")
                if src not in docs:
                    docs[src] = {
                        "source_file": src,
                        "total_pages": meta.get("total_pages", "?"),
                        "chunk_count": 0,
                    }
                docs[src]["chunk_count"] += 1
            return list(docs.values())
        except Exception:
            return []

    def get_stats(self) -> dict:
        """Return database statistics."""
        count = self._collection.count()
        docs = self.list_documents()
        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "documents": docs,
        }

    def clear_all(self):
        """Delete all data from the knowledge base."""
        self._collection.delete(where={"source_file": {"$ne": ""}})
        logger.info("Knowledge base cleared")
