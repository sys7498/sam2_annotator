#!/usr/bin/env python3
"""Export an existing YTVIS annotation JSON as a mask-overlay MP4."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
NATIVE_UI_ROOT = ROOT / "native_ui"
for candidate in (str(NATIVE_UI_ROOT), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from interactive_sam2_gui_ytvis import (  # noqa: E402
    StreamFrameSource,
    YTVISTrackStore,
    render_annotation_overlay,
)


def _default_output(annotation_path: Path) -> Path:
    stem = annotation_path.stem
    suffix = ""
    if stem.startswith("annotations_ytvis_"):
        suffix = stem.removeprefix("annotations_ytvis")
    return annotation_path.with_name(f"annotation_overlay{suffix}.mp4")


def _load_store(annotation_path: Path) -> YTVISTrackStore:
    import json

    with annotation_path.open("r", encoding="utf-8") as handle:
        predictions: Any = json.load(handle)
    if not isinstance(predictions, list):
        raise ValueError("annotation JSON must be a list of YTVIS predictions")

    num_frames = 0
    for prediction in predictions:
        if isinstance(prediction, dict) and isinstance(prediction.get("segmentations"), list):
            num_frames = max(num_frames, len(prediction["segmentations"]))
    store = YTVISTrackStore(num_frames=num_frames, category_id=1)
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        try:
            track_id = int(prediction["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        segmentations = prediction.get("segmentations", [])
        if not isinstance(segmentations, list):
            continue
        bboxes = prediction.get("bboxes", [])
        areas = prediction.get("areas", [])
        frames: dict[int, dict[str, Any]] = {}
        for frame_idx, segmentation in enumerate(segmentations):
            if not isinstance(segmentation, dict):
                continue
            bbox = bboxes[frame_idx] if isinstance(bboxes, list) and frame_idx < len(bboxes) else [0, 0, 0, 0]
            area = areas[frame_idx] if isinstance(areas, list) and frame_idx < len(areas) else 0
            frames[frame_idx] = {"segmentation": segmentation, "bbox": bbox, "area": int(area)}
        store.tracks[track_id] = {
            "frames": frames,
            "score": float(prediction.get("score", 1.0)),
            "category_id": int(prediction.get("category_id", 1)),
        }
    return store


def export(input_path: Path, annotation_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing MP4: {output_path} (pass --overwrite)")
    store = _load_store(annotation_path)
    if store.num_frames <= 0:
        raise ValueError("annotation has no frames")

    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.mp4")
    source = StreamFrameSource(str(input_path), max_frames=store.num_frames)
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        item = source.read_next()
        if item is None:
            raise RuntimeError(f"No readable frames: {input_path}")
        frame, _ = item
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(source.fps)), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open an MP4 writer (mp4v)")
        for frame_idx in range(store.num_frames):
            if frame_idx > 0:
                item = source.read_next()
                if item is None:
                    break
                frame, _ = item
            writer.write(render_annotation_overlay(frame, store.masks_at_frame(frame_idx)))
            written += 1
        writer.release()
        writer = None
        if written == 0:
            raise RuntimeError("No frames written")
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.release()
        source.close()
    print(f"Saved {output_path} ({written} frames, {source.fps:g} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video or frame-sequence directory")
    parser.add_argument("--annotations", type=Path, required=True, help="annotations_ytvis*.json")
    parser.add_argument("--output", type=Path, help="Output MP4 (default: next to annotations)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the requested output MP4")
    args = parser.parse_args()
    output = args.output if args.output is not None else _default_output(args.annotations)
    export(args.input.resolve(), args.annotations.resolve(), output.resolve(), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
