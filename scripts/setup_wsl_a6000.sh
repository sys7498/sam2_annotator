#!/usr/bin/env bash
# One-command setup for Ubuntu WSL2 with an NVIDIA RTX A6000.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="sam2-annotator"
CHECKPOINT_DIR="$PROJECT_ROOT/vendor/sam2_realtime/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/sam2.1_hiera_large.pt"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CHECKPOINT_SHA256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda in WSL first, then rerun this script." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable in WSL. Install/update the NVIDIA Windows WSL driver first." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download the SAM2 checkpoint. Install it with: sudo apt-get install -y curl" >&2
  exit 1
fi

echo "== GPU detected =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  echo "== Installing OpenCV runtime libraries =="
  sudo apt-get update
  sudo apt-get install -y libgl1 libglib2.0-0
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "== Creating conda environment: $ENV_NAME =="
  conda env create -f "$PROJECT_ROOT/environment.yml"
fi

echo "== Installing Python packages =="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements.txt"

mkdir -p "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "== Downloading SAM2.1 large checkpoint (about 0.9 GB) =="
  curl -fL --retry 3 --retry-delay 2 -o "$CHECKPOINT_PATH" "$CHECKPOINT_URL"
fi
echo "$CHECKPOINT_SHA256  $CHECKPOINT_PATH" | sha256sum -c -

echo "== Verifying CUDA and SAM2 realtime =="
cd "$PROJECT_ROOT"
conda run -n "$ENV_NAME" python scripts/verify_a6000.py

cat <<'EOF'

Setup complete.
Run the annotator:
  conda activate sam2-annotator
  cd /path/to/sam2_annotator
  python desktop_annotator.py --pick
EOF
