#!/usr/bin/env bash
# Set up the development environment (macOS). Needs uv and Ollama installed and running. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "Installing Python dependencies ..."
uv sync
chflags nohidden .venv/lib/python3.*/site-packages/*.pth 2>/dev/null || true   # iCloud hides these

echo "Setting up .env ..."
[[ -f .env ]] || cp .env.example .env   # then add your API keys to .env
source .env

# The Ollama app reads launchd's environment, not .env. Restart Ollama after changing this.
echo "Setting OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1} (restart Ollama if it changed) ..."
launchctl setenv OLLAMA_NUM_PARALLEL "${OLLAMA_NUM_PARALLEL:-1}"

echo "Pulling Ollama models ..."
ollama pull ministral-3:8b   # variation generator
ollama pull qwen3:8b         # LLM judges

echo "Done."
