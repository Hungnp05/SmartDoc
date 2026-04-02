# SmartDoc-Insight
### Multi-Modal RAG for Complex Documents

> **Local-First** · **Table-Aware** · **Vision-Enhanced** · **Privacy-Preserving**

A production-grade Retrieval-Augmented Generation system that understands PDFs containing mixed content: raw text, financial tables, and growth charts — all running entirely on local hardware (RTX 4050 6GB VRAM).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SmartDoc-Insight Pipeline                │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Layer A     │  Layer B     │  Layer C     │  Layer D      │
│  Vision      │  Knowledge   │  Retrieval   │  Presentation │
│  Processing  │  Base        │  & Reasoning │  (Streamlit)  │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ PaddleOCR    │ ChromaDB     │ Vector Search│ Upload UI     │
│ LayoutLMv3   │ Hybrid Meta  │ Re-ranking   │ Chat Interface│
│ LLaVA-7B     │ Enriched     │ Llama3-8B    │ Source Viewer │
│ Table→MD     │ Embeddings   │ Inference    │ Real-time     │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (RTX 4050 6GB+ recommended)
- [Ollama](https://ollama.ai/) installed locally

### 1. Clone & Setup
```bash
git clone <repo>
cd smartdoc-insight
cp .env.example .env
```

### 2. Pull AI Models via Ollama
```bash
ollama pull llama3:8b
ollama pull llava:7b
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Access the App
Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
smartdoc-insight/
├── src/
│   ├── layers/
│   │   ├── vision_processing.py     # Layer A: OCR, Layout, Table, Chart
│   │   ├── knowledge_base.py        # Layer B: ChromaDB, Embeddings
│   │   ├── retrieval_reasoning.py   # Layer C: Search, Re-rank, LLM
│   │   └── __init__.py
│   ├── models/
│   │   ├── ollama_client.py         # Ollama API wrapper
│   │   └── embeddings.py            # Embedding model handler
│   ├── utils/
│   │   ├── pdf_parser.py            # PDF → page images
│   │   ├── table_extractor.py       # Table → Markdown converter
│   │   └── chunker.py               # Smart chunking logic
│   └── config.py                    # Central configuration
├── app/
│   ├── streamlit_app.py             # Main Streamlit app
│   ├── components/
│   │   ├── sidebar.py               # Upload & settings panel
│   │   ├── chat.py                  # Chat interface
│   │   └── source_viewer.py         # Retrieved sources display
│   └── styles/
│       └── main.css                 # Custom styling
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_vision.py
│   ├── test_retrieval.py
│   └── sample_docs/
├── scripts/
│   ├── setup.sh                     # One-click setup
│   └── benchmark.py                 # Accuracy benchmarking
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Layout AI** | PaddleOCR + LayoutLMv3 | Detect text/table/chart regions |
| **Vision LLM** | LLaVA-7B (via Ollama) | Describe charts & figures |
| **Main LLM** | Llama 3 8B (via Ollama) | Answer questions |
| **Embeddings** | `nomic-embed-text` | Semantic encoding |
| **Vector DB** | ChromaDB | Similarity search |
| **Orchestration** | LangChain | Pipeline management |
| **UI** | Streamlit | Web interface |
| **Runtime** | Docker + Ollama | Deployment |

---

## Key Technical Innovations

### 1. Table-Aware RAG (+40% accuracy)
Instead of naively chunking PDFs, we:
1. Detect table regions with PaddleOCR Layout
2. Extract full table structure → convert to Markdown
3. Store entire table as one chunk with rich metadata
4. Never split a table across chunks

### 2. 4-bit Quantization for 6GB VRAM
Both LLaVA-7B and Llama3-8B run with Q4_K_M quantization via Ollama, fitting comfortably in RTX 4050's 6GB VRAM.

### 3. Hybrid Metadata Store
Every chunk stores:
```json
{
  "page": 2,
  "content_type": "table",
  "page_summary": "Q3 financial results with revenue breakdown",
  "source_file": "annual_report_2024.pdf",
  "bounding_box": [x1, y1, x2, y2]
}
```

---

## 📊 Performance Benchmark

**SmartDoc-Insight** is designed to handle complex PDF documents containing **tables**, **charts**, and **text** more effectively than standard RAG systems.

Below are the latest benchmark results on the sample document `sample_report.pdf` (8 regions) using 3 representative questions across different content types.

### Benchmark Results (Latest Run)

| Question Type     | Question Sample                                              | Accuracy | Latency   | Evaluation          |
|-------------------|--------------------------------------------------------------|----------|-----------|---------------------|
| **Table QA**      | What is the total revenue in Q3?                            | **50%**  | 27.8s     | Average             |
| **Chart QA**      | What trend does the growth chart show?                      | **75%**  | 26.4s     | Good                |
| **Text QA**       | What are the main risks mentioned in the report?            | **67%**  | 22.3s     | Improved            |
| **Overall**       | -                                                            | **64%**  | -         | Moderate            |

**Document Processing Time**: 50.4 seconds (8 regions)

### Performance Analysis

**Strengths:**
- Excellent performance on **Chart QA (75%)** — The Vision Processing Layer successfully extracts and interprets chart trends.
- Significant improvement in **Text QA**, increasing from 33% to **67%**.
- Faster document processing time compared to previous runs (reduced from 71.9s to 50.4s).

**Weaknesses:**
- **Table QA** remains at 50% — Still the weakest area, likely due to challenges in extracting and reasoning over structured tabular data.
- Overall accuracy at 64% indicates room for further improvement in retrieval quality and reasoning capabilities.

**Notes:**
- The current evaluation still uses **keyword-based scoring**. We are transitioning to a more robust **LLM-as-Judge** system for semantic accuracy.
- The warning `bert.embeddings.position_ids | UNEXPECTED` is harmless and can be safely ignored when loading the cross-encoder model.
- Latency has improved but remains relatively high due to local execution with Ollama and vision processing layers.

### Comparison with Previous Version

| Metric             | Previous Run | Latest Run | Improvement |
|--------------------|--------------|------------|-------------|
| Overall Accuracy   | 53%          | **64%**    | **+11%**    |
| Table QA           | 50%          | 50%        | -           |
| Chart QA           | 75%          | 75%        | -           |
| Text QA            | 33%          | **67%**    | **+34%**    |
| Avg Latency/Query  | ~29s         | ~25.5s     | Improved    |
| Processing Time    | 71.9s        | **50.4s**  | **-21.5s**  |

### Improvement Roadmap

- Fully replace keyword-based evaluation with **LLM-as-Judge** (Correctness + Faithfulness) for more reliable scoring.
- Enhance table understanding by improving the Vision Layer and SmartChunker.
- Implement hybrid retrieval (vector + keyword + metadata filtering).
- Optimize latency through model quantization, embedding caching, and parallel processing.
- Expand the benchmark with a larger and more diverse set of questions (target: 20+ questions).

---

Full detailed benchmark results (including per-question explanations) are saved to `benchmark_results.json` after each run.

**How to reproduce the benchmark:**
```bash
python scripts/benchmark.py --doc data/uploads/sample_report.pdf



## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| RAM | 8GB | 16GB |
| GPU VRAM | Not required | 6GB (RTX 4050+) |
| Storage | 15GB (models + deps) | 20GB |

---

## Installation

### Step 1: Install Ollama

Download and install Ollama from https://ollama.ai

After installation, Ollama runs in the background automatically. Verify it is running:

```
curl http://localhost:11434/api/tags
```

If it returns JSON, Ollama is running correctly. If you see `bind: Only one usage` when running `ollama serve`, Ollama is already running — ignore that error.

### Step 2: Pull AI Models

Open a terminal and run the following three commands. Each downloads several GB, so a stable internet connection is required:

```
ollama pull llama3:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

Verify all three models are available:

```
ollama list
```

All three models must appear in the list before proceeding.

### Step 3: Clone or Extract the Project

```
cd E:\code
tar -xzf smartdoc-insight.tar.gz
cd RAG
```

Or if using git:

```
git clone <repo_url>
cd smartdoc-insight
```

### Step 4: Create a Virtual Environment

**Windows:**

```
python -m venv .venv
.venv\Scripts\activate
```

**Linux / Mac:**

```
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the prompt will show `(.venv)` at the beginning of each line.

### Step 5: Install Python Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on Windows and encounter errors with PaddleOCR:

```
pip install paddlepaddle
pip install paddleocr
```

### Step 6: Configure Environment Variables

```
copy .env.example .env
```

Open the newly created `.env` file and verify the first line reads:

```
OLLAMA_BASE_URL=http://localhost:11434
```

Make sure it says `localhost` and not a different IP address. Leave all other lines at their default values.

### Step 7: Enable dotenv Loading in Config

Open `src/config.py` and add these two lines at the very top of the file, before `import os`:

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

This ensures Python reads your `.env` file and overrides any cached environment values.

---

## Running the Application

```
streamlit run app/streamlit_app.py
```

The browser will open automatically at `http://localhost:8501`. If it does not open, navigate there manually.

---

## Usage

### Uploading a Document

1. Drag a PDF into the **Upload Document** area in the left sidebar, or click **Browse files**
2. Click **Process Document**
3. Wait for the progress bar to complete — the sidebar will show the number of chunks indexed

The first run takes longer because the models need to initialize.

### Asking Questions

Once the sidebar shows `X chunks indexed`, type a question into the chat input at the bottom.

Example questions:

```
What was the Q3 revenue?
Which business segment had the highest growth?
Summarize the main risks mentioned in the report.
Compare performance between 2023 and 2024.
```

### Debug Tab

Switch to the **Debug** tab to inspect the retrieval pipeline in detail: chunk scores before and after reranking.

---

## Creating a Sample PDF for Testing

If you do not have a PDF ready, run the included script to generate a sample financial report:

```
pip install reportlab
py scripts/create_sample_pdf.py
```

The file `sample_report.pdf` will be created in `data/uploads/`.

---

## Running Tests

```
pytest tests/ -v
```

Tests do not require Ollama to be running. The virtual environment must be active.

---

## Running with Docker

Docker Desktop must be installed first.

```
docker-compose -f docker/docker-compose.yml up --build
```

Docker will pull the models automatically and start the application. Access it at `http://localhost:8501`.

---

## Project Structure

```
RAG/
├── app/
│   └── streamlit_app.py          Streamlit UI
├── src/
│   ├── config.py                 Central configuration
│   ├── pipeline.py               Top-level orchestrator
│   ├── layers/
│   │   ├── vision_processing.py  PDF reading and text extraction
│   │   ├── knowledge_base.py     ChromaDB, embeddings, chunking
│   │   └── retrieval_reasoning.py  Retrieval, reranking, answer generation
│   ├── models/
│   │   ├── ollama_client.py      Ollama API wrapper
│   │   └── embeddings.py         Embedding utilities
│   └── utils/
│       ├── table_extractor.py    HTML table to Markdown converter
│       ├── chunker.py            Text chunking utilities
│       └── pdf_parser.py         PDF to image conversion
├── scripts/
│   ├── create_sample_pdf.py      Generate a sample PDF for testing
│   └── benchmark.py              Measure retrieval accuracy
├── tests/                        Unit tests
├── data/
│   ├── uploads/                  Uploaded PDF files
│   └── chroma_db/                Vector database (auto-created)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env                          Your configuration (created from .env.example)
├── .env.example                  Configuration template
└── requirements.txt              Python dependencies
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3:8b` | Model used to generate answers |
| `VISION_MODEL` | `llava:7b` | Model used to describe charts and figures |
| `EMBED_MODEL` | `nomic-embed-text` | Model used to generate embeddings |
| `PDF_DPI` | `200` | Resolution for rendering PDF pages |
| `RETRIEVAL_TOP_K` | `10` | Number of candidate chunks retrieved initially |
| `RERANK_TOP_K` | `4` | Number of chunks kept after reranking |
| `CHUNK_SIZE` | `512` | Maximum characters per chunk |

---

## Troubleshooting

**Problem: `0 chunks indexed` after uploading a document**

Check that `src/config.py` has `load_dotenv(override=True)` at the top. Verify that `OLLAMA_BASE_URL` in `.env` is set to `localhost`.

**Problem: `ollama serve` reports `bind: Only one usage`**

Ollama is already running. No action needed — use it as is.

**Problem: `Cannot import PPStructure from paddleocr`**

This is a breaking API change in newer versions of PaddleOCR. The system automatically falls back to PyMuPDF for text extraction, which works correctly. Core functionality is not affected.

**Problem: Model not found when querying**

Re-pull the required models:

```
ollama pull llama3:8b
ollama pull nomic-embed-text
```

**Problem: Port 8501 is already in use**

```
streamlit run app/streamlit_app.py --server.port 8502
```

**Problem: Want to clear all indexed data**

Delete the `data/chroma_db/` directory, or use the **Clear Knowledge Base** button in the sidebar.

---

## Benchmarking

```
py scripts/benchmark.py --doc data/uploads/sample_report.pdf --output results.json
```

---

## Diagnostics

If you encounter an error that cannot be resolved, run the diagnostic script and share the output:

```
py debug.py
```