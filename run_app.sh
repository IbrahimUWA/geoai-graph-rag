#!/usr/bin/env bash
# Start GraphRAG Explorer (opens http://127.0.0.1:7861 in your browser).
cd "$(dirname "$0")"
. .venv/bin/activate
if command -v ollama >/dev/null && ! curl -s http://localhost:11434/api/tags >/dev/null; then
  (ollama serve >/dev/null 2>&1 &) ; sleep 2
fi
python -m graphrag.app
