# SmartDoc-Insight Makefile
# ─────────────────────────

.PHONY: setup run test docker-up docker-down pull-models clean lint

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "Setting up SmartDoc-Insight..."
	bash scripts/setup.sh

venv:
	python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	streamlit run app/streamlit_app.py \
		--server.port 8501 \
		--server.maxUploadSize 50 \
		--browser.gatherUsageStats false

run-dev:
	streamlit run app/streamlit_app.py \
		--server.port 8501 \
		--server.maxUploadSize 50 \
		--server.runOnSave true \
		--browser.gatherUsageStats false

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d
	@echo "SmartDoc-Insight running at http://localhost:8501"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f app

# ── Models ────────────────────────────────────────────────────────────────────
pull-models:
	@echo "Pulling AI models via Ollama..."
	ollama pull llama3:8b
	ollama pull llava:7b
	ollama pull nomic-embed-text
	@echo "All models ready!"

check-models:
	ollama list

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

benchmark:
	@echo "Running benchmark (requires a PDF document)..."
	python scripts/benchmark.py --doc $(DOC) --output benchmark_results.json

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ app/ tests/
	ruff format --check src/ app/ tests/

format:
	ruff format src/ app/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true

clean-db:
	@echo "Clearing ChromaDB..."
	rm -rf data/chroma_db/*
	@echo "Database cleared."

clean-uploads:
	@echo "Clearing uploaded files..."
	rm -rf data/uploads/*
	@echo "Uploads cleared."

# ── Info ──────────────────────────────────────────────────────────────────────
info:
	@echo "SmartDoc-Insight"
	@echo "================"
	@echo "App:      http://localhost:8501"
	@echo "Ollama:   http://localhost:11434"
	@echo ""
	@echo "Models:"
	@echo "  LLM:    llama3:8b"
	@echo "  Vision: llava:7b"
	@echo "  Embed:  nomic-embed-text"
	@echo ""
	@echo "Tech Stack: Python · PaddleOCR · ChromaDB · LangChain · Ollama · Streamlit"
