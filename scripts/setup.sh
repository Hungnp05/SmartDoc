#!/bin/bash
# SmartDoc-Insight One-Click Setup Script
# ─────────────────────────────────────────
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗██████╗  ██████╗  ██████╗"
echo "  ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗██╔════╝"
echo "  ███████╗██╔████╔██║███████║██████╔╝   ██║   ██║  ██║██║   ██║██║     "
echo "  ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║   ██║  ██║██║   ██║██║     "
echo "  ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║   ██████╔╝╚██████╔╝╚██████╗"
echo "  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝  ╚═════╝"
echo -e "${NC}"
echo -e "  ${CYAN}SmartDoc-Insight${NC} — Multi-Modal RAG Setup"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

check_cmd() {
    if command -v "$1" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 found"
    else
        echo -e "  ${RED}✗ $1 not found${NC} — Please install $1 first"
        [ "$2" = "required" ] && exit 1
    fi
}

check_cmd python3 required
check_cmd pip3 required
check_cmd docker optional
check_cmd ollama optional

# Setup Python environment
echo -e "\n${YELLOW}[2/5] Setting up Python environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "  ${GREEN}✓${NC} Python dependencies installed"

# Copy env file
echo -e "\n${YELLOW}[3/5] Setting up configuration...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "  ${GREEN}✓${NC} .env created from .env.example"
else
    echo -e "  ${GREEN}✓${NC} .env already exists"
fi

# Pull Ollama models
echo -e "\n${YELLOW}[4/5] Pulling AI models via Ollama...${NC}"
if command -v ollama &>/dev/null; then
    echo -e "  Pulling llama3:8b (requires ~5GB disk)..."
    ollama pull llama3:8b

    echo -e "  Pulling llava:7b (requires ~4GB disk)..."
    ollama pull llava:7b

    echo -e "  Pulling nomic-embed-text (requires ~270MB disk)..."
    ollama pull nomic-embed-text

    echo -e "  ${GREEN}✓${NC} All models pulled"
else
    echo -e "  ${YELLOW}⚠${NC} Ollama not found. Please install from https://ollama.ai"
    echo -e "  Then run:"
    echo -e "    ollama pull llama3:8b"
    echo -e "    ollama pull llava:7b"
    echo -e "    ollama pull nomic-embed-text"
fi

# Done
echo -e "\n${YELLOW}[5/5] Setup complete!${NC}"
echo ""
echo -e "  ${GREEN}To start SmartDoc-Insight:${NC}"
echo -e "  ${CYAN}  source .venv/bin/activate${NC}"
echo -e "  ${CYAN}  streamlit run app/streamlit_app.py${NC}"
echo ""
echo -e "  ${GREEN}Or with Docker:${NC}"
echo -e "  ${CYAN}  docker-compose -f docker/docker-compose.yml up --build${NC}"
echo ""
echo -e "  ${GREEN}Access at:${NC} ${CYAN}http://localhost:8501${NC}"
echo ""
