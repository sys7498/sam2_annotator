#!/usr/bin/env bash
# One-command setup for Ubuntu WSL2 with an NVIDIA RTX A6000.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="sam2-annotator"
CHECKPOINT_DIR="$PROJECT_ROOT/vendor/sam2_realtime/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/sam2.1_hiera_small.pt"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_SHA256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"

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
  sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg fontconfig fonts-dejavu-core
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

# Some OpenCV wheels look for ``cv2/qt/fonts`` even on a WSLg desktop. Link
# the system DejaVu font directory so Qt does not emit a warning for every
# window it opens. Existing packaged fonts are left unchanged.
CV2_QT_DIR="$(conda run -n "$ENV_NAME" python -c 'import cv2; from pathlib import Path; print(Path(cv2.__file__).resolve().parent / "qt")')"
if [[ -d "$CV2_QT_DIR" && -d /usr/share/fonts/truetype/dejavu && ! -e "$CV2_QT_DIR/fonts" ]]; then
  ln -s /usr/share/fonts/truetype/dejavu "$CV2_QT_DIR/fonts"
fi

mkdir -p "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "== Downloading SAM2.1 small checkpoint (about 176 MB) =="
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
