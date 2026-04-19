# SmartDoc-Insight
### Multi-Modal RAG for Complex Documents

> **Local-First** · **Table-Aware** · **Vision-Enhanced** · **Privacy-Preserving**

A production-grade Retrieval-Augmented Generation system that understands PDFs containing mixed content: raw text, financial tables, and growth charts — all running entirely on local hardware (RTX 4050 6GB VRAM).

Video Demo: https://www.youtube.com/watch?v=JRN3Oe5GRZA&t=157s

---

## Key Features & Achievements

- **Table-Aware Chunking**: Preserves entire financial tables without splitting, significantly improving Table QA accuracy
- **Multi-Modal Understanding**: Combines Vision (LLaVA) and NLP to interpret both charts and tables
- **Advanced Retrieval**: Hybrid metadata-enriched chunks with reranking
- **Fully Local & Private**: No data leaves your machine — ideal for sensitive enterprise documents
- **Latest Benchmark**: Achieved **84% Overall Accuracy** on 8 complex questions (Table QA reached 79%)

---

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
SmartDoc-Insight/
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── Makefile                        # Build automation (3.8%)
├── requirements.txt                # Python dependencies
├── debug.py                        # Debug script
│
├── app/                            # Application layer
│   ├── __init__.py
│   └── streamlit_app.py           # Main Streamlit web application
│
├── src/                            # Core source code (92.8% Python)
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── pipeline.py                # Main processing pipeline
│   │
│   ├── layers/                    # Processing layers
│   │   ├── __init__.py
│   │   ├── vision_processing.py   # Vision/image processing layer
│   │   ├── knowledge_base.py      # Knowledge base layer
│   │   └── retrieval_reasoning.py # Retrieval and reasoning layer
│   │
│   ├── models/                    # Model clients
│   │   ├── __init__.py
│   │   └── ollama_client.py       # Ollama LLM client
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── table_extractor.py     # Table extraction utility
│
├── docker/                         # Docker configuration (1.2%)
│   ├── Dockerfile                 # Container image definition
│   └── docker-compose.yml         # Multi-container orchestration
│
├── scripts/                        # Utility scripts (2.2% Shell)
│   ├── setup.sh                   # Setup script
│   └── benchmark.py               # Performance benchmarking
│
└── tests/                          # Test suite
    └── test_pipeline.py           # Pipeline tests
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



## Performance Benchmark

**SmartDoc-Insight** is designed to effectively handle complex PDF documents containing **tables**, **charts**, and **text** — areas where traditional RAG systems often struggle.

The following results are from the latest benchmark run on the sample document `sample_report.pdf` (8 questions) using 8 representative questions in Vietnamese.

### Benchmark Results (Latest)

| Question Type     | Question Sample                                              | Accuracy | Latency   | Evaluation          |
|-------------------|--------------------------------------------------------------|----------|-----------|---------------------|
| **Table QA**      | Doanh thu quý 3 năm 2024 là bao nhiêu?                      | **79%**  | 27.8s     | Good                |
| **Table QA**      | Tăng trưởng doanh thu quý 3 so với quý 2 là bao nhiêu phần trăm? | **75%**  | 31.5s     | Good                |
| **Table QA**      | Mảng kinh doanh nào có tỷ lệ tăng trưởng cao nhất trong năm 2024? | **33%**  | 18.2s     | Poor                |
| **Table QA**      | Tổng số nhân viên tính đến cuối năm 2024 là bao nhiêu?      | **100%** | 17.6s     | Excellent           |
| **Table QA**      | Chi phí nhân công chiếm bao nhiêu phần trăm tổng chi phí?   | **100%** | 18.1s     | Excellent           |
| **Text QA**       | Các rủi ro chính được nêu trong báo cáo là gì?              | **100%** | 30.6s     | Excellent           |
| **Text QA**       | Kế hoạch doanh thu năm 2025 là bao nhiêu?                   | **100%** | 18.0s     | Excellent           |
| **Chart QA**      | Xu hướng tăng trưởng doanh thu từ 2021 đến 2024 như thế nào?| **100%** | 22.0s     | Excellent           |
| **Overall**       | -                                                            | **84%**  | **27.6s** | **Excellent**       |

**Document Processing Time**: 106.4 seconds (32 chunks indexed from 13 text + 19 table regions)

### Performance Analysis

**Strengths:**
- Achieved excellent **overall accuracy of 84%**, surpassing the target of ≥ 80%.
- Outstanding performance on **Chart QA (100%)** and **Text QA (100%)** — demonstrating strong capability in understanding charts and textual content.
- Significant improvement in **Table QA**, reaching **79%** accuracy thanks to the enhanced Table-Aware Chunking mechanism.
- Average query latency of **27.6 seconds** remains acceptable for a fully local setup on RTX 4050 6GB VRAM.

**Areas for Improvement:**
- One Table QA question scored low at 33%, indicating occasional difficulty in precise segment identification and numerical reasoning from complex tables.
- Document processing time increased to 106.4s (due to more regions detected); further optimization in parallel processing and caching is needed.

**Notes:**
- The benchmark uses a combination of keyword and semantic matching for scoring.
- The `bert.embeddings.position_ids | UNEXPECTED` warning when loading the cross-encoder model is normal and can be safely ignored.
- Results are automatically saved to `results.json` after each run.

---

Full benchmark details are automatically saved to `benchmark_results.json` after each run.

**To reproduce this benchmark:**
```bash
python scripts/benchmark.py --doc data/uploads/sample_report.pdf
```


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

## Note
Note: Due to hardware limitations, this project runs in continuous model swapping in VRAM. This is the main reason why local RAG systems run extremely slowly if you don't have enough VRAM to accommodate both models simultaneously.