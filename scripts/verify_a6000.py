#!/usr/bin/env python3
"""Fail-fast verification for the WSL RTX A6000 installation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "sam2_realtime"
CHECKPOINT = VENDOR / "checkpoints" / "sam2.1_hiera_large.pt"

for path in (ROOT, VENDOR.parent, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable inside WSL. Check the NVIDIA WSL driver with nvidia-smi.")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint missing: {CHECKPOINT}. Run scripts/setup_wsl_a6000.sh.")

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}, CUDA wheel: {torch.version.cuda}")
    print(f"GPU: {props.name}, compute capability: {props.major}.{props.minor}")
    print(f"OpenCV: {cv2.__version__}")

    from sam2_realtime.sam2.build_sam import build_sam2_realtime_predictor

    predictor = build_sam2_realtime_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(CHECKPOINT),
        device="cuda",
    )
    print(f"SAM2 realtime: {type(predictor).__name__} loaded successfully")


if __name__ == "__main__":
    main()
