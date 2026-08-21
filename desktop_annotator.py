#!/usr/bin/env python3
"""Native OpenCV launcher for the SAM2 realtime mask annotator.

This uses the included native OpenCV UI and self-contained SAM2 checkout.
There is no HTTP, JSON image transport, or browser-side mask compositing in
this path: the window receives frames directly from OpenCV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/workspace/shared/aria_recording/5fps_sampled_sequences")
DEFAULT_OUTPUT_ROOT = Path("/workspace/shared/aria_recording/5fps_sampled_masks")
NATIVE_UI_ROOT = PROJECT_ROOT / "native_ui"
SAM2_ROOT = PROJECT_ROOT / "vendor" / "sam2_realtime"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


@dataclass
class _PendingPredecessor:
    track_id: int
    frame_idx: int
    last_visible_frame: int


@dataclass
class _LineageRecorder:
    """Conservative automatic lineage from an explicit ID lifecycle.

    A raw tracking disappearance is not sufficient evidence for a structural
    event: it is also caused by occlusion or model loss. We therefore infer a
    relation only from the annotator's explicit ID lifecycle at one transition.
    """

    pending: List[_PendingPredecessor] = field(default_factory=list)
    recent_children: List[tuple[int, int]] = field(default_factory=list)
    relations: List[dict[str, Any]] = field(default_factory=list)
    transition_window: int = 12

    def _valid_pending(self, frame_idx: int) -> List[_PendingPredecessor]:
        return [p for p in self.pending if frame_idx - p.frame_idx <= self.transition_window]

    def end_predecessor(self, track_id: int, frame_idx: int) -> None:
        if any(p.track_id == int(track_id) for p in self.pending):
            return
        self.pending.append(
            _PendingPredecessor(int(track_id), int(frame_idx), max(0, int(frame_idx) - 1))
        )
        self._try_create_relation(int(frame_idx))

    def add_child(self, track_id: int, frame_idx: int) -> None:
        if any(child_id == int(track_id) for child_id, _ in self.recent_children):
            return
        self.recent_children.append((int(track_id), int(frame_idx)))
        self._try_create_relation(int(frame_idx))

    def _try_create_relation(self, frame_idx: int) -> None:
        candidates = self._valid_pending(int(frame_idx))
        if not candidates:
            return
        children = sorted(
            {
                child_id
                for child_id, child_frame in self.recent_children
                if any(
                    abs(int(child_frame) - int(parent.frame_idx)) <= self.transition_window
                    and int(child_id) != int(parent.track_id)
                    for parent in candidates
                )
            }
        )
        if len(candidates) == 1 and len(children) == 2:
            self._record_relation(
                relation_type="separation",
                parents=candidates,
                children=children,
                frame_idx=int(frame_idx),
            )
            return
        if len(candidates) != 2:
            return
        shared_children = [
            child_id
            for child_id in children
            if all(
                any(
                    existing_id == child_id
                    and abs(int(child_frame) - int(parent.frame_idx)) <= self.transition_window
                    for existing_id, child_frame in self.recent_children
                )
                for parent in candidates
            )
        ]
        if len(shared_children) == 1:
            self._record_relation(
                relation_type="joining",
                parents=candidates,
                children=shared_children,
                frame_idx=int(frame_idx),
            )

    def _record_relation(
        self,
        *,
        relation_type: str,
        parents: List[_PendingPredecessor],
        children: List[int],
        frame_idx: int,
    ) -> None:
        parent_ids = sorted(int(parent.track_id) for parent in parents)
        child_ids = sorted(int(child) for child in children)
        relation = {
            "relation_id": f"auto-{relation_type}-{len(self.relations) + 1:04d}",
            "type": str(relation_type),
            "predecessor_ids": parent_ids,
            "successor_ids": child_ids,
            "frame_idx": int(max([frame_idx] + [parent.frame_idx for parent in parents])),
            "predecessor_last_visible_frames": {
                str(parent.track_id): int(parent.last_visible_frame) for parent in parents
            },
            "source": "auto_id_lifecycle",
            "status": "auto",
        }
        self.relations.append(relation)
        for parent in parents:
            self.pending.remove(parent)
        self.recent_children = [
            (child_id, child_frame)
            for child_id, child_frame in self.recent_children
            if child_id not in set(child_ids)
        ]
        print(
            f"[Lineage] Auto {relation_type}: "
            f"predecessors {parent_ids} -> successors {child_ids} "
            f"at frame {relation['frame_idx'] + 1}"
        )

    def rename_id(self, old_id: int, new_id: int) -> None:
        for pending in self.pending:
            if pending.track_id == int(old_id):
                pending.track_id = int(new_id)
        self.recent_children = [
            (int(new_id) if child_id == int(old_id) else child_id, child_frame)
            for child_id, child_frame in self.recent_children
        ]
        for relation in self.relations:
            relation["predecessor_ids"] = [
                int(new_id) if item == int(old_id) else item for item in relation["predecessor_ids"]
            ]
            relation["successor_ids"] = [
                int(new_id) if item == int(old_id) else item for item in relation["successor_ids"]
            ]

    def mark_successor_deleted(self, track_id: int) -> None:
        for relation in self.relations:
            if int(track_id) not in relation.get("successor_ids", []):
                continue
            relation["status"] = "needs_review"
            removed = relation.setdefault("deleted_successor_ids", [])
            if int(track_id) not in removed:
                removed.append(int(track_id))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "relation_direction": "predecessor_to_successor",
            "auto_rule": {
                "separation": "one explicit object-ID deletion plus exactly two new object IDs within 12 frames",
                "joining": "two explicit object-ID deletions plus exactly one new object ID within 12 frames",
            },
            "relations": self.relations,
            "pending_predecessors": [
                {
                    "track_id": item.track_id,
                    "frame_idx": item.frame_idx,
                    "last_visible_frame": item.last_visible_frame,
                }
                for item in self.pending
            ],
        }


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = int((int(track_id) * 37) % 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _render_lineage_graph(
    output_path: Path,
    *,
    track_store: Any,
    relations: List[dict[str, Any]],
    pending: List[_PendingPredecessor],
    object_category_id: int,
    processed_frames: int,
) -> None:
    """Render saved object IDs and their structural-event graph."""
    tracks = getattr(track_store, "tracks", {})
    spans: dict[int, List[int]] = {}
    for track_id, track in tracks.items():
        if not isinstance(track, dict) or int(track.get("category_id", object_category_id)) != object_category_id:
            continue
        frame_map = track.get("frames", {})
        if not isinstance(frame_map, dict):
            continue
        indices = sorted(int(frame_idx) for frame_idx in frame_map.keys())
        if indices:
            spans[int(track_id)] = indices

    # Do not show a transient ID which never produced a saved mask. A relation
    # is shown only when every endpoint survived in the final annotation.
    track_ids = sorted(spans, key=lambda track_id: (spans[track_id][0], track_id))
    saved_ids = set(track_ids)
    visible_relations: List[dict[str, Any]] = []
    for relation in relations:
        parent_ids = [int(track_id) for track_id in relation.get("predecessor_ids", [])]
        child_ids = [int(track_id) for track_id in relation.get("successor_ids", [])]
        if parent_ids and child_ids and set(parent_ids + child_ids).issubset(saved_ids):
            visible_relations.append(relation)
    visible_pending = [item for item in pending if int(item.track_id) in saved_ids]

    if not track_ids:
        canvas = np.full((280, 1200, 3), (24, 28, 34), dtype=np.uint8)
        cv2.putText(canvas, "Structural lineage graph", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No saved object masks in the final annotation.", (30, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (195, 210, 225), 1, cv2.LINE_AA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f"{output_path.stem}.tmp.png")
        if not cv2.imwrite(str(temporary_path), canvas):
            raise RuntimeError(f"Failed to write lineage graph: {temporary_path}")
        temporary_path.replace(output_path)
        return

    # Place each saved ID in a left-to-right lineage layer. This deliberately
    # omits the dense per-frame timeline: the final PNG is a relationship graph
    # meant for quick structural review.
    relation_edges: List[tuple[int, int, str, str]] = []
    incoming_ids: set[int] = set()
    joining_children: set[int] = set()
    review_ids: set[int] = set()
    layer_for_id = {track_id: 0 for track_id in track_ids}
    for relation in sorted(visible_relations, key=lambda item: int(item.get("frame_idx", 0))):
        parent_ids = [int(track_id) for track_id in relation.get("predecessor_ids", [])]
        child_ids = [int(track_id) for track_id in relation.get("successor_ids", [])]
        relation_type = str(relation.get("type", "relation"))
        status = str(relation.get("status", "auto"))
        child_layer = max(layer_for_id[parent_id] for parent_id in parent_ids) + 1
        for child_id in child_ids:
            layer_for_id[child_id] = max(layer_for_id[child_id], child_layer)
            incoming_ids.add(child_id)
            if relation_type == "joining":
                joining_children.add(child_id)
            if status != "auto":
                review_ids.add(child_id)
            for parent_id in parent_ids:
                relation_edges.append((parent_id, child_id, relation_type, status))

    columns: dict[int, List[int]] = {}
    for track_id in track_ids:
        columns.setdefault(layer_for_id[track_id], []).append(track_id)
    for node_ids in columns.values():
        node_ids.sort(key=lambda track_id: (spans[track_id][0], track_id))

    max_layer = max(columns)
    max_rows = max(len(node_ids) for node_ids in columns.values())
    left, top, column_width, row_height, radius = 145, 125, 250, 112, 30
    width = max(900, left + (max_layer + 1) * column_width + 125)
    height = max(360, top + max_rows * row_height + 105)
    canvas = np.full((height, width, 3), (250, 250, 250), dtype=np.uint8)
    positions: dict[int, tuple[int, int]] = {}
    for layer, node_ids in columns.items():
        for row, track_id in enumerate(node_ids):
            positions[track_id] = (left + layer * column_width, top + row * row_height)

    text_color = (35, 35, 35)
    blue = (235, 120, 35)
    orange = (20, 130, 250)
    review = (45, 45, 220)
    cv2.putText(canvas, "Structural lineage graph", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)
    review_count = sum(1 for relation in visible_relations if relation.get("status") != "auto")
    cv2.putText(canvas, f"saved IDs: {len(track_ids)} | events: {len(visible_relations)} | needs review: {review_count}", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1, cv2.LINE_AA)

    def draw_curve(start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        sx, sy = start
        ex, ey = end
        control_distance = max(40, (ex - sx) // 2)
        points = []
        for step in range(25):
            t = step / 24.0
            x = (1 - t) ** 3 * sx + 3 * (1 - t) ** 2 * t * (sx + control_distance) + 3 * (1 - t) * t ** 2 * (ex - control_distance) + t ** 3 * ex
            y = (1 - t) ** 3 * sy + 3 * (1 - t) ** 2 * t * sy + 3 * (1 - t) * t ** 2 * ey + t ** 3 * ey
            points.append((int(round(x)), int(round(y))))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.arrowedLine(canvas, points[-3], points[-1], color, 2, cv2.LINE_AA, tipLength=0.45)

    for parent_id, child_id, relation_type, status in relation_edges:
        parent_x, parent_y = positions[parent_id]
        child_x, child_y = positions[child_id]
        edge_color = orange if relation_type == "joining" else blue
        draw_curve((parent_x + radius, parent_y), (child_x - radius, child_y), edge_color)

    for track_id in track_ids:
        x, y = positions[track_id]
        node_color = orange if track_id in joining_children else blue
        cv2.circle(canvas, (x, y), radius, node_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, review if track_id in review_ids else (35, 35, 35), 2, cv2.LINE_AA)
        label = str(track_id)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, label, (x - text_size[0] // 2, y + text_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if track_id not in incoming_ids:
            cv2.arrowedLine(canvas, (28, y), (x - radius, y), blue, 2, cv2.LINE_AA, tipLength=0.18)

    cv2.putText(canvas, "blue: separation / source ID | orange: joining result | red border: needs review", (30, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.png")
    if not cv2.imwrite(str(temporary_path), canvas):
        raise RuntimeError(f"Failed to write lineage graph: {temporary_path}")
    temporary_path.replace(output_path)


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")


def _load_gui_module():
    _require(NATIVE_UI_ROOT / "interactive_sam2_gui_ytvis.py", "native GUI")
    _require(SAM2_ROOT / "sam2", "bundled SAM2 realtime source")
    # The native UI is a normal Python module. Its module-global SAM2_ROOT is
    # deliberately replaced below, so models are loaded from this project.
    for item in (str(NATIVE_UI_ROOT), str(SAM2_ROOT.parent), str(SAM2_ROOT)):
        if item not in sys.path:
            sys.path.insert(0, item)
    import interactive_sam2_gui_ytvis as gui  # type: ignore

    gui.SAM2_ROOT = str(SAM2_ROOT)
    # The bundled realtime predictor is imported as ``sam2_realtime.sam2``.
    for item in reversed((str(SAM2_ROOT.parent), str(SAM2_ROOT))):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    return gui


def _make_lineage_gui_class(gui):
    """Attach project-specific lineage persistence to the native UI."""

    class LineageInteractiveSam2Gui(gui.InteractiveSam2Gui):
        def __init__(self, args) -> None:
            self.lineage = _LineageRecorder()
            super().__init__(args)

        def _delete_object_by_id(self, obj_id: int) -> None:
            track_id = int(obj_id)
            is_object = self._kind_for_track(track_id) == "object"
            # ``d`` is the explicit annotation boundary used to distinguish a
            # true structural disappearance from ordinary tracker occlusion.
            if is_object:
                self.lineage.end_predecessor(track_id, int(self.frame_idx))
            super()._delete_object_by_id(track_id)
            if is_object:
                self.lineage.mark_successor_deleted(track_id)

        def _create_new_active_object(self, kind: str = "object") -> int:
            created_id = int(super()._create_new_active_object(kind=kind))
            if str(kind) != "hand":
                self.lineage.add_child(created_id, int(self.frame_idx))
            return created_id

        def _set_active_object_by_id(self, target_id, *, create_if_missing=True, kind=None) -> bool:
            track_id = int(target_id)
            known_before = track_id in self._used_track_ids()
            ok = bool(
                super()._set_active_object_by_id(
                    track_id,
                    create_if_missing=create_if_missing,
                    kind=kind,
                )
            )
            if ok and not known_before and create_if_missing and self._kind_for_track(track_id) == "object":
                self.lineage.add_child(track_id, int(self.frame_idx))
            return ok

        def _rename_active_object(self, new_id: int) -> None:
            old_id = self.active_obj_id
            super()._rename_active_object(int(new_id))
            if old_id is not None and self.active_obj_id == int(new_id):
                self.lineage.rename_id(int(old_id), int(new_id))

        def _print_lineage(self) -> None:
            if not self.lineage.relations:
                print("[Lineage] No confirmed automatic relations.")
                return
            for relation in self.lineage.relations:
                print(
                    f"[Lineage] {relation['relation_id']}: "
                    f"{relation['predecessor_ids']} -> {relation['successor_ids']} "
                    f"({relation['type']}, {relation['status']})"
                )

        def _on_future_annotations_invalidated(self, start_frame: int) -> None:
            """Keep existing lineage evidence visible, but flag rewound events."""
            cutoff = int(start_frame)
            self.lineage.pending = [item for item in self.lineage.pending if item.frame_idx < cutoff]
            self.lineage.recent_children = [
                (track_id, frame_idx)
                for track_id, frame_idx in self.lineage.recent_children
                if frame_idx < cutoff
            ]
            for relation in self.lineage.relations:
                if int(relation.get("frame_idx", -1)) >= cutoff:
                    relation["status"] = "needs_review"
                    relation["invalidated_by_rewind"] = True
            print(f"[Lineage] Future relations from frame {cutoff + 1} marked needs_review.")

        def _handle_key(self, key: int) -> bool:
            if int(key & 0xFF) == ord("l"):
                self._log_key(int(key))
                self._print_lineage()
                return False
            return bool(super()._handle_key(key))

        def _save_outputs(self, *, force: bool = False, render_video: bool = False) -> bool:
            if not super()._save_outputs(force=force, render_video=render_video):
                return False
            output_dir = Path(self.output_dir)
            relation_path = output_dir / "lineage_relations.json"
            temporary = relation_path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.lineage.payload(), handle, ensure_ascii=False, indent=2)
            temporary.replace(relation_path)

            graph_path = output_dir / "lineage_graph.png"
            graph_saved = False
            try:
                _render_lineage_graph(
                    graph_path,
                    track_store=self.track_store,
                    relations=self.lineage.relations,
                    pending=self.lineage.pending,
                    object_category_id=int(self.args.category_id),
                    processed_frames=int(getattr(self, "max_frame_seen", self.frame_idx + 1)),
                )
                graph_saved = True
            except Exception as exc:
                print(f"[Lineage] Graph rendering skipped: {exc}")

            session_path = Path(getattr(self, "session_meta_out_path", output_dir / str(self.args.session_meta_out)))
            try:
                with session_path.open("r", encoding="utf-8") as handle:
                    session = json.load(handle)
                session["lineage_relations"] = str(relation_path.resolve())
                session["auto_relation_count"] = len(self.lineage.relations)
                if graph_saved:
                    session["lineage_graph"] = str(graph_path.resolve())
                temporary_session = session_path.with_suffix(".json.tmp")
                with temporary_session.open("w", encoding="utf-8") as handle:
                    json.dump(session, handle, ensure_ascii=False, indent=2)
                temporary_session.replace(session_path)
            except Exception as exc:
                print(f"[Lineage] Session metadata update skipped: {exc}")

            print(f"[Lineage] Saved relations: {relation_path} ({len(self.lineage.relations)} confirmed)")
            if graph_saved:
                print(f"[Lineage] Saved graph: {graph_path}")
            return True

    return LineageInteractiveSam2Gui


def _outputs_for(
    input_path: Path,
    output_root: Path,
    relative_input: Path | None = None,
) -> tuple[Path, str, str]:
    key = relative_input
    if key is None:
        key = Path(input_path.stem if input_path.is_file() else input_path.name)
    key = key.with_suffix("") if key.suffix else key

    # Aria sequence names end in ``_sep`` or ``_join``. Keep the event type
    # at the top of the annotation tree for event-level review/export.
    # A direct --input uses its source path because its output key is ``rgb``.
    event_type = _event_type_from_parts((relative_input or input_path).parts)
    if event_type:
        out_dir = output_root.joinpath(event_type, *key.parts)
    else:
        out_dir = output_root.joinpath(*key.parts)
    return out_dir, "annotations_ytvis.json", "interactive_session_meta.json"


def _event_type_from_parts(parts: Iterable[str]) -> str | None:
    """Return the Aria event suffix nearest to the sequence leaf, if any."""
    for part in reversed(tuple(parts)):
        normalized_name = Path(part).stem.lower()
        if normalized_name.endswith("_sep"):
            return "sep"
        if normalized_name.endswith("_join"):
            return "join"
    return None


def _make_args(
    gui,
    input_path: Path,
    options: argparse.Namespace,
    relative_input: Path | None = None,
):
    output_dir, ytvis_out, session_meta_out = _outputs_for(
        input_path,
        Path(options.output_dir),
        relative_input,
    )
    return gui.AppArgs(
        input=str(input_path.resolve()),
        output_dir=str(output_dir),
        ytvis_out=ytvis_out,
        session_meta_out=session_meta_out,
        display_name=(relative_input.as_posix() if relative_input is not None else input_path.name),
        video_id=int(options.video_id),
        category_id=int(options.category_id),
        hand_category_id=int(options.hand_category_id),
        max_frames=options.max_frames,
        start_id=int(options.start_id),
        hand_start_id=int(options.hand_start_id),
        brush_radius=int(options.brush_radius),
        window_width=int(options.window_width),
        autoplay=bool(options.autoplay),
        play_fps=options.play_fps,
        state_window=int(options.state_window),
        gc_every=int(options.gc_every),
        device=options.device,
        offload_video_to_cpu=bool(options.offload_video_to_cpu),
        offload_state_to_cpu=bool(options.offload_state_to_cpu),
        sam2_checkpoint=options.checkpoint,
        sam2_model_cfg=options.config,
    )


def _discover_dataset_inputs(gui, base_dir: Path) -> List[tuple[Path, Path]]:
    """Find only image-frame sequence directories named ``rgb`` recursively."""
    found: List[tuple[Path, Path]] = []
    for root_text, dir_names, file_names in os.walk(base_dir):
        root = Path(root_text)
        dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
        visible_files = sorted(name for name in file_names if not name.startswith("."))
        image_files = [name for name in visible_files if gui._is_image_file(name)]
        if root.name.lower() == "rgb" and image_files:
            relative_path = root.relative_to(base_dir)
            found.append((root, relative_path))
            dir_names[:] = []

    return sorted(found, key=lambda item: str(item[1]).lower())


def _pick_native_file() -> str | None:
    """Open an OS file picker when the Python desktop has a display."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select a video or image for SAM2 annotation",
            filetypes=[
                ("Media", "*.mp4 *.avi *.mov *.mkv *.webm *.mpeg *.mpg *.m4v *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return selected or None
    except Exception as exc:
        print(f"[Launcher] Native file picker unavailable: {exc}")
        return None


def _run_picker(gui, args_list: Iterable, output_root: Path) -> None:
    items: List = list(args_list)
    if not items:
        print("[Launcher] No supported video/image/frame-directory found.")
        return
    selected = 0
    while True:
        picker = gui.SequencePickerGui(items, start_index=selected)
        choice = picker.run()
        if choice is None:
            return
        selected = int(choice)
        item = items[selected]
        print(f"[Launcher] Open: {item.input}")
        gui.LineageInteractiveSam2Gui(item).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native OpenCV SAM2 realtime video mask annotator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, help="One video, image, or frame directory")
    group.add_argument(
        "--dataset",
        type=Path,
        help="Folder searched recursively for rgb frame directories; opens a TODO/DONE picker",
    )
    parser.add_argument("--pick", action="store_true", help="Open the operating system file picker")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Annotation output root (default: /workspace/shared/aria_recording/5fps_sampled_masks)",
    )
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-id", type=int, default=1)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--hand-category-id", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=100)
    parser.add_argument("--hand-start-id", type=int, default=0)
    parser.add_argument("--brush-radius", type=int, default=8)
    parser.add_argument("--window-width", type=int, default=1440)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--play-fps", type=float)
    parser.add_argument("--state-window", type=int, default=0, help="0 keeps full streaming state")
    parser.add_argument("--gc-every", type=int, default=0)
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if options.pick:
        picked = _pick_native_file()
        if picked:
            options.input = Path(picked)
    if options.input is None and options.dataset is None:
        options.dataset = DEFAULT_DATASET_ROOT
        print(f"[Launcher] Default dataset: {options.dataset}")

    gui = _load_gui_module()
    gui.LineageInteractiveSam2Gui = _make_lineage_gui_class(gui)
    if options.dataset is not None:
        base = options.dataset.expanduser().resolve()
        if not base.is_dir():
            raise SystemExit(f"Dataset folder not found: {base}")
        args_list = [
            _make_args(gui, input_path, options, relative_input)
            for input_path, relative_input in _discover_dataset_inputs(gui, base)
        ]
        _run_picker(gui, args_list, Path(options.output_dir))
        return

    input_path = options.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    gui.LineageInteractiveSam2Gui(_make_args(gui, input_path, options)).run()


if __name__ == "__main__":
    main()
