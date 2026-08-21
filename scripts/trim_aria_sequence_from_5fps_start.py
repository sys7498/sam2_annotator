#!/usr/bin/env python3
"""Make one Aria sequence start at an existing 5fps frame, transactionally.

The VRS is retained as capture provenance.  RGB video, timestamp sidecars,
hand sidecars, sampled frame trees, encoded video, and the empty annotation
bundle are rebuilt so they describe the same cropped clip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jpeg_files(directory: Path) -> list[Path]:
    return sorted(
        [*directory.glob("*.jpg"), *directory.glob("*.jpeg")],
        key=lambda path: path.name,
    )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def frame_tree(source_frames: list[Path], destination: Path, *, start: int, step: int) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    selected = source_frames[start::step]
    if not selected:
        raise ValueError(f"No frames selected from {source_frames[0].parent}")
    for new_index, source in enumerate(selected):
        hardlink_or_copy(source, destination / f"{new_index:06d}.jpg")


def run_ffmpeg(source: Path, destination: Path, *, start_frame: int, scale: str | None) -> None:
    filters = [f"select='gte(n\\,{start_frame})'", "setpts=N/(30*TB)"]
    if scale:
        filters.append(scale)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-vf", ",".join(filters),
        "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", str(destination),
    ]
    subprocess.run(command, check=True)


def video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_raw_sidecars(
    raw_dir: Path,
    temp_dir: Path,
    *,
    start_rgb_index: int,
    expected_frames: int,
) -> tuple[int, int, int]:
    with (raw_dir / "rgb_timestamps.csv").open(newline="", encoding="utf-8") as handle:
        timestamp_rows = list(csv.DictReader(handle))
        timestamp_fields = list(timestamp_rows[0])
    selected_timestamps = timestamp_rows[start_rgb_index:]
    if len(selected_timestamps) != expected_frames:
        raise ValueError("RGB timestamp row count does not match selected video frames")
    first_mp4_time = float(selected_timestamps[0]["mp4_time_ns"])
    with (temp_dir / "rgb_timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timestamp_fields)
        writer.writeheader()
        for row in selected_timestamps:
            updated = dict(row)
            updated["mp4_time_ns"] = str(float(row["mp4_time_ns"]) - first_mp4_time)
            writer.writerow(updated)

    start_device_time = int(selected_timestamps[0]["vrs_device_time_ns"])
    with (raw_dir / "hand_tracking.jsonl").open(encoding="utf-8") as handle:
        hand_rows = [json.loads(line) for line in handle if line.strip()]
    selected_hands = [row for row in hand_rows if int(row["device_time_ns"]) >= start_device_time]
    if not selected_hands:
        raise ValueError("No hand-tracking records remain after crop")
    hand_start_index = int(selected_hands[0]["sample_index"])
    for row in selected_hands:
        row["sample_index"] = int(row["sample_index"]) - hand_start_index
    with (temp_dir / "hand_tracking.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_hands:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (raw_dir / "hand_landmarks.csv").open(newline="", encoding="utf-8") as handle:
        landmark_reader = csv.DictReader(handle)
        landmark_fields = list(landmark_reader.fieldnames or [])
        landmark_rows = [
            row for row in landmark_reader if int(row["device_time_ns"]) >= start_device_time
        ]
    for row in landmark_rows:
        row["sample_index"] = str(int(row["sample_index"]) - hand_start_index)
    with (temp_dir / "hand_landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=landmark_fields)
        writer.writeheader()
        writer.writerows(landmark_rows)

    conversion = json.loads((raw_dir / "rgb_conversion.json").read_text(encoding="utf-8"))
    conversion.update(
        {
            "num_mp4_frames": expected_frames,
            "first_video_timestamp_ns": start_device_time,
            "end_video_timestamp_ns": int(selected_timestamps[-1]["vrs_device_time_ns"]),
            "video_duration_ns": int(round((expected_frames / 30.0) * 1_000_000_000)),
            "clip_source_rgb_start_index": start_rgb_index,
        }
    )
    write_json(temp_dir / "rgb_conversion.json", conversion)

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    rgb = manifest.setdefault("rgb", {})
    rgb["frames_in_vrs"] = int(len(timestamp_rows))
    rgb["frames_in_clip"] = expected_frames
    rgb["clip_source_rgb_start_index"] = start_rgb_index
    rgb["clip_start_vrs_device_time_ns"] = start_device_time
    hand = manifest.setdefault("hand_tracking", {})
    hand["samples"] = len(selected_hands)
    hand["detected_sides"] = sum(
        int(row.get("left") is not None) + int(row.get("right") is not None)
        for row in selected_hands
    )
    hand["landmark_rows"] = len(landmark_rows)
    write_json(temp_dir / "manifest.json", manifest)

    return start_device_time, hand_start_index, len(landmark_rows)


def replace_path(target: Path, replacement: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / target.name
    if archived.exists():
        raise FileExistsError(f"Archive target already exists: {archived}")
    os.replace(target, archived)
    os.replace(replacement, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-5fps-index", type=int, required=True)
    parser.add_argument(
        "--aria-root",
        type=Path,
        default=Path("/workspace/shared/aria_recording"),
    )
    args = parser.parse_args()
    if args.start_5fps_index < 0:
        raise SystemExit("--start-5fps-index must be non-negative")

    root = args.aria_root.resolve()
    participant = str(args.participant)
    sequence = str(args.sequence)
    active_5fps = root / "split_5fps_sampled_sequences" / "join" / sequence / "rgb"
    raw_dir = root / "raw" / participant / sequence
    raw_frames_dir = root / "30fps_raw" / participant / sequence / "rgb"
    ten_frames_dir = root / "10fps_sampled_sequences" / participant / sequence / "rgb"
    encoded_video = root / "encoded_videos" / participant / sequence / "rgb.mp4"
    mask_dir = root / "split_5fps_sampled_masks" / "join" / sequence / "rgb"
    for path in (active_5fps, raw_dir, raw_frames_dir, ten_frames_dir, encoded_video, mask_dir):
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")
    annotation_path = mask_dir / "annotations_ytvis.json"
    if annotation_path.is_file():
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if isinstance(annotations, list) and annotations:
            raise SystemExit(
                "Refusing to reset a non-empty annotation bundle. "
                "Migrate its frame indices explicitly first."
            )

    source_5fps = jpeg_files(active_5fps)
    source_30fps = jpeg_files(raw_frames_dir)
    if args.start_5fps_index >= len(source_5fps):
        raise SystemExit("5fps start index is outside the available frames")
    start_hash = sha256(source_5fps[args.start_5fps_index])
    matches = [index for index, path in enumerate(source_30fps) if sha256(path) == start_hash]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one matching 30fps frame, found {matches}")
    start_30fps_index = matches[0]
    expected_30fps = len(source_30fps) - start_30fps_index
    expected_5fps = len(source_5fps) - args.start_5fps_index
    stamp = datetime.now(timezone.utc).strftime("pretrim_%Y%m%dT%H%M%SZ")

    split_frame_dirs = [active_5fps]
    for split_root in root.glob("split_5fps_sampled_sequences*"):
        candidate = split_root / "join" / sequence / "rgb"
        if candidate.exists() and candidate not in split_frame_dirs:
            split_frame_dirs.append(candidate)
    deprecated_root = root / "5fps_sampled_sequences_deprecated"
    if deprecated_root.exists():
        split_frame_dirs.extend(
            path for path in deprecated_root.glob(f"**/{sequence}/rgb") if path not in split_frame_dirs
        )

    with tempfile.TemporaryDirectory(prefix=f".{sequence}.trim-", dir=str(root)) as temp_text:
        temp_root = Path(temp_text)
        raw_temp = temp_root / "raw_sidecars"
        raw_temp.mkdir()
        raw_video_temp = raw_temp / "rgb.mp4"
        run_ffmpeg(raw_dir / "rgb.mp4", raw_video_temp, start_frame=start_30fps_index, scale=None)
        if video_frame_count(raw_video_temp) != expected_30fps:
            raise RuntimeError("Trimmed raw MP4 has an unexpected frame count")
        start_device_time, hand_start_index, landmark_count = build_raw_sidecars(
            raw_dir,
            raw_temp,
            start_rgb_index=start_30fps_index,
            expected_frames=expected_30fps,
        )

        raw_frame_temp = temp_root / "30fps_rgb"
        frame_tree(source_30fps, raw_frame_temp, start=start_30fps_index, step=1)
        ten_frame_temp = temp_root / "10fps_rgb"
        frame_tree(source_30fps, ten_frame_temp, start=start_30fps_index, step=3)
        split_temps: dict[Path, Path] = {}
        for index, source_dir in enumerate(split_frame_dirs):
            source = jpeg_files(source_dir)
            if len(source) != len(source_5fps) or sha256(source[args.start_5fps_index]) != start_hash:
                raise RuntimeError(f"5fps duplicate is not aligned with active source: {source_dir}")
            replacement = temp_root / f"fivefps_{index}"
            frame_tree(source, replacement, start=args.start_5fps_index, step=1)
            split_temps[source_dir] = replacement

        encoded_temp = temp_root / "encoded_rgb.mp4"
        run_ffmpeg(raw_dir / "rgb.mp4", encoded_temp, start_frame=start_30fps_index, scale="scale=672:504")
        if video_frame_count(encoded_temp) != expected_30fps:
            raise RuntimeError("Trimmed encoded MP4 has an unexpected frame count")

        # Validate every replacement before changing an existing path.
        if sha256(raw_frame_temp / "000000.jpg") != start_hash:
            raise RuntimeError("New 30fps first JPEG does not match the selected 5fps start")
        for replacement in split_temps.values():
            if len(jpeg_files(replacement)) != expected_5fps or sha256(replacement / "000000.jpg") != start_hash:
                raise RuntimeError("New 5fps sequence validation failed")
        if len(jpeg_files(ten_frame_temp)) != (expected_30fps + 2) // 3:
            raise RuntimeError("New 10fps sequence frame count is incorrect")
        with (raw_temp / "rgb_timestamps.csv").open(newline="", encoding="utf-8") as handle:
            timestamps = list(csv.DictReader(handle))
        if len(timestamps) != expected_30fps or float(timestamps[0]["mp4_time_ns"]) != 0.0:
            raise RuntimeError("New RGB timestamp sidecar validation failed")

        clip_transform = {
            "schema_version": "aria-clip-transform-1.0",
            "source_vrs": (raw_dir / f"{sequence}.vrs").name,
            "source_rgb_start_index": start_30fps_index,
            "source_5fps_start_index": args.start_5fps_index,
            "clip_start_vrs_device_time_ns": start_device_time,
            "rgb_frames": expected_30fps,
            "five_fps_frames": expected_5fps,
            "ten_fps_frames": len(jpeg_files(ten_frame_temp)),
            "hand_tracking_source_start_index": hand_start_index,
            "hand_landmark_rows": landmark_count,
            "selected_first_frame_sha256": start_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(raw_temp / "clip_transform.json", clip_transform)

        # All validation has passed: archive old results, then install replacements.
        raw_history = raw_dir / "history" / stamp
        for name in (
            "rgb.mp4", "rgb_timestamps.csv", "rgb_conversion.json", "hand_tracking.jsonl",
            "hand_landmarks.csv", "manifest.json",
        ):
            replace_path(raw_dir / name, raw_temp / name, raw_history)
        os.replace(raw_temp / "clip_transform.json", raw_dir / "clip_transform.json")

        replace_path(raw_frames_dir, raw_frame_temp, raw_frames_dir.parent / "history" / stamp)
        replace_path(ten_frames_dir, ten_frame_temp, ten_frames_dir.parent / "history" / stamp)
        for source_dir, replacement in split_temps.items():
            replace_path(source_dir, replacement, source_dir.parent / "history" / stamp)
        replace_path(encoded_video, encoded_temp, encoded_video.parent / "history" / stamp)

        # This result bundle contains no masks; archive it so the picker shows TODO.
        empty_mask_dir = temp_root / "empty_mask_rgb"
        empty_mask_dir.mkdir()
        replace_path(mask_dir, empty_mask_dir, mask_dir.parent / "history" / stamp)

    print(json.dumps({
        "sequence": sequence,
        "source_30fps_start_index": start_30fps_index,
        "source_5fps_start_index": args.start_5fps_index,
        "rgb_frames": expected_30fps,
        "five_fps_frames": expected_5fps,
        "history_stamp": stamp,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
