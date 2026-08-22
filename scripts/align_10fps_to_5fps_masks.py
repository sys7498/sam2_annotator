#!/usr/bin/env python3
"""Rebuild 10fps inputs so every even frame is the exact 5fps mask image.

Legacy 5fps and 10fps extractions can have different FFmpeg sampling phases.
This script makes the 5fps RGB JPEGs the immutable even frames of the 10fps
model input and fills odd frames from the corresponding 30fps source timeline.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def image_files(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.jpg"), *directory.glob("*.jpeg")], key=lambda path: path.name)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def discover_fivefps(dataset_root: Path) -> list[Path]:
    found: list[Path] = []
    for root_text, dir_names, file_names in os.walk(dataset_root):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith(".") and name.lower() != "history")
        root = Path(root_text)
        if root.name.lower() != "rgb":
            continue
        if any(Path(name).suffix.lower() in {".jpg", ".jpeg"} for name in file_names):
            found.append(root)
            dir_names[:] = []
    return sorted(found)


def discover_output_10fps(output_root: Path) -> list[Path]:
    found: list[Path] = []
    for root_text, dir_names, file_names in os.walk(output_root):
        dir_names[:] = sorted(name for name in dir_names if not name.startswith(".") and name.lower() != "history")
        root = Path(root_text)
        if root.name.lower() == "rgb" and any(Path(name).suffix.lower() in {".jpg", ".jpeg"} for name in file_names):
            found.append(root)
            dir_names[:] = []
    return sorted(found)


def current_is_aligned(mask_frames: list[Path], current_frames: list[Path]) -> bool:
    return (
        len(current_frames) in {2 * len(mask_frames) - 1, 2 * len(mask_frames)}
        and all(digest(current_frames[2 * index]) == digest(mask) for index, mask in enumerate(mask_frames))
    )


def extract_phase(source_video: Path, destination: Path, phase: int) -> list[Path]:
    destination.mkdir()
    expression = f"fps=30,select='eq(mod(n\\,6)\\,{phase})'"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_video), "-map", "0:v:0", "-an", "-vf", expression,
            "-vsync", "0", "-q:v", "2", "-start_number", "0", str(destination / "%06d.jpg"),
        ],
        check=True,
    )
    return image_files(destination)


def desired_sources(mask_frames: list[Path], raw_video: Path, temp_root: Path) -> list[Path]:
    # Legacy `fps=5` extraction is equivalent to FPS-30 frames 2,8,14,... .
    # Re-extract these frames from the source video and verify that every mask
    # image is truly from that exact time before taking 5,11,17,... as odds.
    reconstructed_even = extract_phase(raw_video, temp_root / "legacy_even", phase=2)
    if len(reconstructed_even) != len(mask_frames) or any(
        digest(reconstructed_even[index]) != digest(mask)
        for index, mask in enumerate(mask_frames)
    ):
        raise RuntimeError(f"Legacy 5fps phase verification failed: {raw_video}")
    reconstructed_odd = extract_phase(raw_video, temp_root / "legacy_odd", phase=5)
    if len(reconstructed_odd) not in {len(mask_frames) - 1, len(mask_frames)}:
        raise RuntimeError(f"Legacy 10fps odd-frame count is inconsistent: {raw_video}")
    sources: list[Path] = []
    for index, mask in enumerate(mask_frames):
        sources.append(mask)  # exact JPEG whose mask was annotated
        if index < len(reconstructed_odd):
            sources.append(reconstructed_odd[index])
    return sources


def legacy_sources(raw_video: Path, temp_root: Path) -> list[Path]:
    even = extract_phase(raw_video, temp_root / "legacy_even", phase=2)
    odd = extract_phase(raw_video, temp_root / "legacy_odd", phase=5)
    if len(odd) not in {len(even) - 1, len(even)}:
        raise RuntimeError(f"Legacy 10fps frame count is inconsistent: {raw_video}")
    sources: list[Path] = []
    for index, frame in enumerate(even):
        sources.append(frame)
        if index < len(odd):
            sources.append(odd[index])
    return sources


def participant_from_sequence(sequence: str) -> str:
    participant = sequence.split("_", 1)[0]
    if not participant:
        raise ValueError(f"Cannot infer participant from sequence: {sequence}")
    return participant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aria-root", type=Path, default=Path("/workspace/shared/aria_recording"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--sequence", help="Only rebuild this sequence name")
    parser.add_argument("--all-output", action="store_true", help="Align every existing 10fps sequence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    aria_root = args.aria_root.resolve()
    dataset_root = (args.dataset_root or aria_root / "split_5fps_sampled_sequences" / "join").resolve()
    output_root = (args.output_root or aria_root / "10fps_sampled_sequences").resolve()
    raw_root = (args.raw_root or aria_root / "raw").resolve()
    archive_root = output_root.parent / ".archive_pre_aligned_10fps"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    mask_dirs = {path.parent.name: path for path in discover_fivefps(dataset_root)}
    if args.all_output:
        targets = [(path.parent.name, path.parent.parent.name, mask_dirs.get(path.parent.name), path) for path in discover_output_10fps(output_root)]
    else:
        targets = [(path.parent.name, participant_from_sequence(path.parent.name), path, output_root / participant_from_sequence(path.parent.name) / path.parent.name / "rgb") for path in mask_dirs.values()]
    updated = skipped = 0
    for sequence, participant, mask_dir, output_dir in targets:
        raw_video = raw_root / participant / sequence / "rgb.mp4"
        if args.sequence and sequence != args.sequence:
            continue
        if not raw_video.is_file() or not output_dir.is_dir():
            print(f"[skip] missing raw or 10fps output for {mask_dir}")
            skipped += 1
            continue
        current = image_files(output_dir)
        mask_frames = image_files(mask_dir) if mask_dir is not None else []
        if mask_frames and current_is_aligned(mask_frames, current):
            print(f"[skip] already aligned: {participant}/{sequence} ({len(current)} frames)")
            skipped += 1
            continue
        with tempfile.TemporaryDirectory(prefix=f".{sequence}.align-", dir=str(output_dir.parent)) as temp_text:
            temp_root = Path(temp_text)
            sources = desired_sources(mask_frames, raw_video, temp_root) if mask_frames else legacy_sources(raw_video, temp_root)
            print(f"[align] {participant}/{sequence}: masks={len(mask_frames)} -> 10fps={len(sources)}")
            if args.dry_run:
                updated += 1
                continue
            replacement = temp_root / "rgb"
            replacement.mkdir()
            for index, source in enumerate(sources):
                link_or_copy(source, replacement / f"{index:06d}.jpg")
            rebuilt = image_files(replacement)
            if len(rebuilt) != len(sources) or any(
                digest(rebuilt[index]) != digest(source) for index, source in enumerate(sources)
            ):
                raise RuntimeError(f"Verification failed before replacing: {output_dir}")
            archived = archive_root / timestamp / participant / sequence / "rgb"
            archived.parent.mkdir(parents=True, exist_ok=True)
            if archived.exists():
                raise FileExistsError(f"Archive already exists: {archived}")
            os.replace(output_dir, archived)
            os.replace(replacement, output_dir)
        updated += 1
    print(f"Done. updated={updated} skipped={skipped} archive={archive_root / timestamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
