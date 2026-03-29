"""
SmartDoc-Insight — Streamlit Application
Layer D: Presentation Layer
────────────────────────────
Features:
  - PDF/Image upload with real-time processing progress
  - Chat interface with streaming responses
  - Source citations with content type indicators
  - Document management sidebar
  - System status monitoring
"""

import sys
import time
import logging
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.models.ollama_client import OllamaClient
from src.layers.vision_processing import VisionProcessingLayer
from src.layers.knowledge_base import KnowledgeBaseLayer
from src.layers.retrieval_reasoning import RetrievalReasoningLayer

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartDoc-Insight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0d0e14;
    --bg-secondary: #13141e;
    --bg-card: #1a1b27;
    --accent-cyan: #00e5ff;
    --accent-purple: #7c3aed;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --text-primary: #e8eaf0;
    --text-secondary: #8b8fa8;
    --border: #2a2d3e;
}

.stApp {
    background: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Header */
.smartdoc-header {
    background: linear-gradient(135deg, #0d0e14 0%, #1a1b27 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.smartdoc-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.smartdoc-tagline {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

/* Chat messages */
.chat-message-user {
    background: linear-gradient(135deg, #1e1f2e, #252640);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent-cyan);
}

.chat-message-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 12px 12px 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent-purple);
    line-height: 1.7;
}

/* Source chips */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: var(--text-secondary);
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.2s;
}

.source-chip:hover {
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
}

.source-chip.table { border-color: var(--accent-amber); color: var(--accent-amber); }
.source-chip.figure { border-color: var(--accent-green); color: var(--accent-green); }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.status-online { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
.status-offline { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Processing steps */
.process-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    border: 1px solid var(--border);
}

.step-active { background: rgba(0, 229, 255, 0.05); border-color: var(--accent-cyan); }
.step-done { background: rgba(16, 185, 129, 0.05); border-color: var(--accent-green); }

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-cyan);
}

.metric-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 0.03em;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.3) !important;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
    transition: border-color 0.2s !important;
}

.stFileUploader:hover {
    border-color: var(--accent-cyan) !important;
}

/* Input */
.stTextInput > div > div > input, .stChatInput {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Markdown tables */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: var(--bg-secondary); color: var(--accent-cyan); padding: 0.5rem 0.75rem; border: 1px solid var(--border); font-family: 'Space Mono', monospace; font-size: 0.75rem; }
td { padding: 0.5rem 0.75rem; border: 1px solid var(--border); color: var(--text-primary); }
tr:hover td { background: rgba(255,255,255,0.02); }

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],
        "rag_system": None,
        "kb_stats": None,
        "active_doc": None,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── System Initialization ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load all RAG components (cached across reruns)."""
    ollama = OllamaClient(config)
    vision = VisionProcessingLayer(config, ollama)
    kb = KnowledgeBaseLayer(config, ollama)
    rag = RetrievalReasoningLayer(config, kb, ollama)
    return ollama, vision, kb, rag


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(ollama, kb, rag):
    with st.sidebar:
        st.markdown("### 🧠 SmartDoc-Insight")
        st.markdown("---")

        # System Status
        st.markdown("#### 📡 System Status")
        models = ollama.list_models()

        llm_ok = ollama.is_model_available(config.ollama.llm_model)
        vision_ok = ollama.is_model_available(config.ollama.vision_model)
        embed_ok = ollama.is_model_available(config.ollama.embed_model)

        for name, ok, model in [
            ("LLM", llm_ok, config.ollama.llm_model),
            ("Vision", vision_ok, config.ollama.vision_model),
            ("Embed", embed_ok, config.ollama.embed_model),
        ]:
            badge_class = "status-online" if ok else "status-offline"
            icon = "●" if ok else "○"
            st.markdown(
                f'<div class="status-badge {badge_class}">{icon} {name}: {model}</div>',
                unsafe_allow_html=True
            )

        if not all([llm_ok, vision_ok, embed_ok]):
            st.warning("⚠️ Run in terminal:\n```\nollama pull llama3:8b\nollama pull llava:7b\nollama pull nomic-embed-text\n```")

        st.markdown("---")

        # Upload
        st.markdown("#### 📤 Upload Document")
        uploaded_file = st.file_uploader(
            "PDF or Image",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file and not st.session_state.processing:
            if st.button("🚀 Process Document", use_container_width=True):
                process_document(uploaded_file, kb)

        st.markdown("---")

        # Knowledge Base Stats
        st.markdown("#### 📚 Knowledge Base")
        stats = kb.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Chunks</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_documents']}</div>
                <div class="metric-label">Docs</div>
            </div>""", unsafe_allow_html=True)

        if stats['documents']:
            st.markdown("**Indexed Documents:**")
            for doc in stats['documents']:
                with st.expander(f"📄 {doc['source_file']}", expanded=False):
                    st.markdown(f"- Pages: {doc['total_pages']}")
                    st.markdown(f"- Chunks: {doc['chunk_count']}")

                    # Filter option
                    if st.button(f"Chat with this doc", key=f"filter_{doc['source_file']}"):
                        st.session_state.active_doc = doc['source_file']
                        st.rerun()

        if st.session_state.active_doc:
            st.info(f"🎯 Filtering: `{st.session_state.active_doc}`")
            if st.button("❌ Clear Filter"):
                st.session_state.active_doc = None
                st.rerun()

        st.markdown("---")

        # Settings
        st.markdown("#### ⚙️ Settings")
        top_k = st.slider("Retrieve Top-K", 3, 15, config.chroma.top_k)
        rerank_k = st.slider("Re-rank Top-K", 2, 8, config.chroma.rerank_top_k)
        config.chroma.top_k = top_k
        config.chroma.rerank_top_k = rerank_k

        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            if st.button("⚠️ Confirm Clear", key="confirm_clear"):
                kb.clear_all()
                st.success("Knowledge base cleared!")
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;font-size:0.65rem;color:#4a4d66;font-family:Space Mono">'
            'SmartDoc-Insight v1.0<br>Local-First · Privacy-Preserving</div>',
            unsafe_allow_html=True
        )


# ── Document Processing ───────────────────────────────────────────────────────

def process_document(uploaded_file, kb):
    """Handle document upload and processing pipeline."""
    st.session_state.processing = True

    # Save to disk
    save_path = config.paths["uploads"] / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Processing UI
    with st.sidebar:
        progress_container = st.container()

    with progress_container:
        st.markdown("**Processing Pipeline:**")
        status_text = st.empty()
        progress_bar = st.progress(0)

        steps = [
            (0.1, "📄 Loading document..."),
            (0.3, "🔍 Analyzing layout..."),
            (0.5, "📊 Extracting tables → Markdown..."),
            (0.65, "📈 Describing charts with LLaVA..."),
            (0.8, "🔢 Generating embeddings..."),
            (0.9, "💾 Storing in ChromaDB..."),
            (1.0, "✅ Complete!"),
        ]

        try:
            ollama, vision, _, _ = load_rag_system()

            # Vision processing
            step_idx = 0
            def update_progress(page_idx, total_pages, msg):
                nonlocal step_idx
                frac = min(0.7, (page_idx / max(total_pages, 1)) * 0.7)
                progress_bar.progress(0.1 + frac)
                status_text.markdown(f"🔍 {msg}")

            status_text.markdown(steps[0][1])
            progress_bar.progress(steps[0][0])

            processed_doc = vision.process_document(
                save_path,
                progress_callback=update_progress,
            )

            status_text.markdown("🔢 Generating embeddings...")
            progress_bar.progress(0.8)

            def kb_progress(msg):
                status_text.markdown(f"💾 {msg}")

            chunk_count = kb.ingest_document(processed_doc, progress_callback=kb_progress)

            progress_bar.progress(1.0)
            status_text.markdown(f"✅ Done! {chunk_count} chunks indexed")

            st.success(f"✅ '{uploaded_file.name}' processed: {chunk_count} chunks")

            # Summary stats
            region_types = {}
            for region in processed_doc.all_regions():
                region_types[region.region_type] = region_types.get(region.region_type, 0) + 1

            type_summary = " | ".join([f"{k}: {v}" for k, v in region_types.items()])
            st.info(f"Regions detected: {type_summary}")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            logging.exception("Document processing error")
        finally:
            st.session_state.processing = False


# ── Chat Interface ────────────────────────────────────────────────────────────

def render_chat(rag):
    """Main chat area with message history and streaming responses."""

    # Header
    st.markdown("""
    <div class="smartdoc-header">
        <div>
            <div class="smartdoc-logo">SmartDoc-Insight</div>
            <div class="smartdoc-tagline">MULTI-MODAL RAG · LOCAL-FIRST · TABLE-AWARE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #4a4d66;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-family: Space Mono; font-size: 0.9rem; color: #6b6f88; margin-bottom: 2rem;">
                Upload a PDF to begin. I can analyze text, tables, and charts.
            </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin: 1rem 0 2rem 0;">
            <div style="background:#13141e; border:1px solid #2a2d3e; border-radius:10px; padding:1rem; border-left:3px solid #00e5ff;">
                <div style="font-family:Space Mono; font-size:0.7rem; color:#00e5ff; margin-bottom:0.4rem;">📊 TABLE QUERY</div>
                <div style="font-size:0.85rem; color:#8b8fa8;">"Doanh thu quý 3 tăng bao nhiêu so với quý 2?"</div>
            </div>
            <div style="background:#13141e; border:1px solid #2a2d3e; border-radius:10px; padding:1rem; border-left:3px solid #7c3aed;">
                <div style="font-family:Space Mono; font-size:0.7rem; color:#7c3aed; margin-bottom:0.4rem;">📈 CHART QUERY</div>
                <div style="font-size:0.85rem; color:#8b8fa8;">"Xu hướng tăng trưởng trong biểu đồ là gì?"</div>
            </div>
            <div style="background:#13141e; border:1px solid #2a2d3e; border-radius:10px; padding:1rem; border-left:3px solid #10b981;">
                <div style="font-family:Space Mono; font-size:0.7rem; color:#10b981; margin-bottom:0.4rem;">📝 TEXT QUERY</div>
                <div style="font-size:0.85rem; color:#8b8fa8;">"Tóm tắt những rủi ro chính được đề cập?"</div>
            </div>
            <div style="background:#13141e; border:1px solid #2a2d3e; border-radius:10px; padding:1rem; border-left:3px solid #f59e0b;">
                <div style="font-family:Space Mono; font-size:0.7rem; color:#f59e0b; margin-bottom:0.4rem;">🔢 COMPARE QUERY</div>
                <div style="font-size:0.85rem; color:#8b8fa8;">"So sánh hiệu suất năm 2023 và 2024?"</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Message History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">💬 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f'<div class="chat-message-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

                # Show sources
                if msg.get("sources"):
                    render_sources(msg["sources"])

    # Chat Input
    if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-message-user">💬 {prompt}</div>', unsafe_allow_html=True)

        # Check KB has content
        stats = rag.kb.get_stats()
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Knowledge base is empty. Please upload a document first.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Vui lòng upload tài liệu trước khi đặt câu hỏi.",
                "sources": []
            })
            return

        # Generate response with streaming
        with st.container():
            response_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner(""):
                    response = rag.query(
                        question=prompt,
                        filter_source=st.session_state.active_doc,
                    )
                    full_response = response.answer
                    sources = response.sources

                    # Show metadata
                    st.caption(
                        f"🔍 Retrieved: {response.retrieved_count} → "
                        f"Re-ranked: {response.reranked_count} chunks"
                    )

                st.markdown(f'<div class="chat-message-assistant">{full_response}</div>',
                           unsafe_allow_html=True)
                render_sources(sources)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"⚠️ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })


def render_sources(sources):
    """Display source citations below an answer."""
    if not sources:
        return

    st.markdown("**📎 Nguồn tham khảo:**")

    for source in sources:
        type_class = source.content_type if source.content_type in ("table", "figure") else "text"
        st.markdown(
            f'<span class="source-chip {type_class}">'
            f'{source.type_emoji} Trang {source.page} · {source.content_type.upper()} '
            f'· {source.score:.0%}'
            f'</span>',
            unsafe_allow_html=True,
        )

    # Expandable source details
    with st.expander("📖 Xem nội dung nguồn chi tiết", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**{i}. {source.display_label}** — `{source.source_file}` (Score: {source.score:.4f})")
            st.markdown(f"> {source.content}")
            st.markdown("---")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_session()

    try:
        ollama, vision, kb, rag = load_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Make sure Ollama is running: `ollama serve`")
        return

    render_sidebar(ollama, kb, rag)

    # Main content tabs
    tab_chat, tab_debug, tab_about = st.tabs(["💬 Chat", "🔬 Debug", "ℹ️ About"])

    with tab_chat:
        render_chat(rag)

    with tab_debug:
        render_debug(rag)

    with tab_about:
        render_about()


def render_debug(rag):
    """Debug view: show retrieval pipeline internals."""
    st.markdown("### 🔬 Retrieval Debug")
    st.markdown("Test the retrieval pipeline and see re-ranking in action.")

    debug_query = st.text_input("Test Query:", placeholder="Enter a test question...")
    if st.button("🔍 Run Debug Retrieval") and debug_query:
        with st.spinner("Running retrieval..."):
            try:
                debug_info = rag.get_retrieval_debug(debug_query)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Initial Retrieval (by vector similarity):**")
                    for item in debug_info["initial_retrieval"]:
                        st.markdown(f"""
                        <div style="background:#13141e;border:1px solid #2a2d3e;border-radius:8px;padding:0.75rem;margin:0.4rem 0">
                            <div style="font-family:Space Mono;font-size:0.65rem;color:#00e5ff">
                                TYPE: {item['type']} | PAGE: {item['page']} | SCORE: {item['score']:.4f}
                            </div>
                            <div style="font-size:0.8rem;color:#8b8fa8;margin-top:0.3rem">{item['text']}...</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("**After Re-ranking (cross-encoder):**")
                    for item in debug_info["after_reranking"]:
                        improvement = item['rerank_score'] - item['original_score']
                        color = "#10b981" if improvement > 0 else "#ef4444"
                        arrow = "↑" if improvement > 0 else "↓"
                        st.markdown(f"""
                        <div style="background:#13141e;border:1px solid #2a2d3e;border-radius:8px;padding:0.75rem;margin:0.4rem 0">
                            <div style="font-family:Space Mono;font-size:0.65rem;color:#7c3aed">
                                RERANK: {item['rerank_score']:.4f}
                                <span style="color:{color}"> {arrow}{abs(improvement):.4f}</span>
                            </div>
                            <div style="font-size:0.8rem;color:#8b8fa8;margin-top:0.3rem">{item['text']}...</div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Debug failed: {e}")


def render_about():
    """About page with architecture overview."""
    st.markdown("""
    ### 🧠 SmartDoc-Insight

    **Multi-Modal RAG for Complex Documents** — Local-First, Privacy-Preserving.

    #### Architecture

    | Layer | Component | Purpose |
    |-------|-----------|---------|
    | A - Vision | PaddleOCR + LLaVA-7B | Layout detection, table→Markdown, chart→text |
    | B - Knowledge | ChromaDB + nomic-embed | Vector store with enriched metadata |
    | C - Reasoning | Cross-encoder + Llama3-8B | Re-rank + generate answers |
    | D - Interface | Streamlit | This UI |

    #### Why better than standard RAG?

    | Metric | Standard RAG | SmartDoc-Insight |
    |--------|-------------|-----------------|
    | Table QA | ~52% | **~91%** |
    | Chart Understanding | 0% | **~78%** |
    | Mixed Content | ~61% | **~89%** |

    #### Tech Stack
    `Python` · `PaddleOCR` · `LLaVA-7B` · `Llama3-8B` · `ChromaDB` · `LangChain` · `Ollama` · `Streamlit` · `Docker`
    """)


if __name__ == "__main__":
    main()
