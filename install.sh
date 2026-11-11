#!/usr/bin/env bash
# GraphRAG Explorer - one-step installer for Linux (Ubuntu, Debian, Fedora, ...).
#   bash install.sh
# Creates .venv, installs PyTorch (GPU if an NVIDIA card is found, otherwise CPU), the app's packages,
# the ready-built knowledge graph, a local language model through Ollama, and a menu shortcut.
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then echo "Python 3.10-3.13 is needed: sudo apt install python3 python3-venv"; exit 1; fi
"$PY" - <<'EOF' || { echo "Python 3.10 to 3.13 is needed"; exit 1; }
import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)
EOF

say "Creating the virtual environment (.venv)"
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  say "NVIDIA GPU found: installing PyTorch with CUDA"
  pip install -q torch --index-url https://download.pytorch.org/whl/cu128
else
  say "No NVIDIA GPU: installing the CPU build of PyTorch (the app works fine on CPU)"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

say "Installing the app's packages"
pip install -q -r requirements.txt

say "Downloading the ready-built knowledge graph"
python scripts/get_data.py

say "Language model (Ollama)"
if ! command -v ollama >/dev/null; then
  read -r -p "Ollama is not installed. Install it now (needs sudo)? [Y/n] " yn
  if [[ "${yn:-Y}" =~ ^[Yy]$ ]]; then curl -fsSL https://ollama.com/install.sh | sh; fi
fi
if command -v ollama >/dev/null; then
  MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  MODEL=qwen3:8b; [ "$MEM_GB" -lt 16 ] && MODEL=qwen3:4b
  echo "Pulling $MODEL (about 5 GB the first time)"
  ollama pull "$MODEL"
  printf '{\n "llm_backend": "ollama",\n "llm_model": "%s"\n}\n' "$MODEL" > settings.json
else
  echo "Skipped. Install Ollama later from https://ollama.com and run: ollama pull qwen3:8b"
fi

say "Adding a menu shortcut"
APPS="$HOME/.local/share/applications"; mkdir -p "$APPS"
cat > "$APPS/graphrag-explorer.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=GraphRAG Explorer
Comment=Ask questions over a knowledge graph of your documents
Exec=bash "$PWD/run_app.sh"
Icon=$PWD/graphrag/assets/icon.png
Terminal=true
Categories=Education;Science;
EOF
chmod +x run_app.sh

say "Done. Start the app with:  ./run_app.sh   (or 'GraphRAG Explorer' in your applications menu)"
