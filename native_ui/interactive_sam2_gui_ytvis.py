#!/usr/bin/env python3
import gc
import json
import os
import sys
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2_realtime")

for p in [PROJECT_ROOT, SAM2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import bbox_xywh_from_mask, coco_rle_to_mask, ensure_dir, get_image_paths, mask_to_coco_rle


def _resolve_under(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _write_json_atomic(path: str, payload: Any) -> None:
    """Write json atomically to avoid partial/corrupted files on interruption."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _format_key_event(key_raw: int) -> str:
    """Return a readable OpenCV key event label for terminal logs."""
    raw = int(key_raw)
    key = int(raw & 0xFF)
    special = {
        27: "Esc",
        10: "Enter",
        13: "Enter",
        32: "Space",
        81: "Left",
        82: "Up",
        83: "Right",
        84: "Down",
    }
    if key in special:
        label = special[key]
    elif 32 <= key <= 126:
        character = chr(key)
        label = f"Shift+{character.lower()}" if character.isalpha() and character.isupper() else repr(character)
    else:
        label = f"code {key}"
    return f"{label} (raw={raw})" if raw != key else label


class StreamFrameSource:
    def __init__(self, input_path: str, max_frames: Optional[int]) -> None:
        self.input_path = os.path.abspath(input_path)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.fps = 10.0
        self.total_frames: Optional[int] = None
        self.read_count = 0
        if os.path.isdir(self.input_path):
            self._mode = "dir"
        elif os.path.isfile(self.input_path) and _is_image_file(self.input_path):
            self._mode = "image"
        else:
            self._mode = "video"
        self._paths: List[str] = []
        self._cap: Optional[cv2.VideoCapture] = None
        self._path_idx = 0

        if self._mode in {"dir", "image"}:
            if self._mode == "dir":
                paths = get_image_paths(self.input_path)
            else:
                paths = [self.input_path]
            if self.max_frames is not None:
                paths = paths[: max(0, self.max_frames)]
            self._paths = paths
            self.total_frames = len(self._paths)
            return

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.input_path}")
        cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if cap_fps > 0.0:
            self.fps = cap_fps
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0:
            self.total_frames = min(n, self.max_frames) if self.max_frames is not None else n
        self._cap = cap

    def read_next(self) -> Optional[Tuple[np.ndarray, str]]:
        if self.max_frames is not None and self.read_count >= self.max_frames:
            return None

        if self._mode in {"dir", "image"}:
            while self._path_idx < len(self._paths):
                path = self._paths[self._path_idx]
                self._path_idx += 1
                frame = cv2.imread(path)
                if frame is None:
                    continue
                self.read_count += 1
                return frame, os.path.basename(path)
            return None

        if self._cap is None:
            return None
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return None
            name = f"{self.read_count:06d}.jpg"
            self.read_count += 1
            return frame, name

    def read_at(self, frame_idx: int) -> Optional[Tuple[np.ndarray, str]]:
        """Read an absolute input frame and position subsequent reads after it."""
        index = int(frame_idx)
        if index < 0:
            return None
        if self.total_frames is not None and index >= int(self.total_frames):
            return None
        if self._mode in {"dir", "image"}:
            if index >= len(self._paths):
                return None
            path = self._paths[index]
            frame = cv2.imread(path)
            if frame is None:
                return None
            self._path_idx = index + 1
            self.read_count = index + 1
            return frame, os.path.basename(path)
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self.read_count = index + 1
        return frame, f"{index:06d}.jpg"

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _id_color(obj_id: int) -> Tuple[int, int, int]:
    hue = int((obj_id * 37) % 180)
    sat = 200
    val = 255
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_center(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


@dataclass
class PromptState:
    pos_points: List[Tuple[int, int]] = field(default_factory=list)
    neg_points: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AppArgs:
    input: str
    output_dir: str
    ytvis_out: str
    session_meta_out: str
    display_name: Optional[str] = None
    video_id: int = 1
    category_id: int = 1
    hand_category_id: int = 0
    max_frames: Optional[int] = None
    start_id: int = 100
    hand_start_id: int = 0
    brush_radius: int = 8
    window_width: int = 1080
    autoplay: bool = False
    play_fps: Optional[float] = None
    state_window: int = 0
    gc_every: int = 0
    device: Optional[str] = None
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".mpeg",
        ".mpg",
        ".m4v",
    }


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


def _discover_inputs(base_dir: str) -> List[Tuple[str, str]]:
    """Discover annotatable inputs under base_dir.

    Returns:
        list of (input_path, obj_name)
    """
    dir_items: List[Tuple[str, str]] = []
    image_items: List[Tuple[str, str]] = []
    video_items: List[Tuple[str, str]] = []
    if not os.path.isdir(base_dir):
        return []
    for name in sorted(os.listdir(base_dir)):
        if not name or name.startswith("."):
            continue
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            # Frame directory candidate: include when it has readable images.
            try:
                paths = get_image_paths(path)
            except Exception:
                paths = []
            if len(paths) > 0:
                dir_items.append((path, os.path.basename(os.path.normpath(path))))
            continue
        if os.path.isfile(path) and _is_image_file(path):
            obj_name = os.path.splitext(os.path.basename(path))[0]
            image_items.append((path, obj_name))
            continue
        if os.path.isfile(path) and _is_video_file(path):
            obj_name = os.path.splitext(os.path.basename(path))[0]
            video_items.append((path, obj_name))

    # Prefer frame directories, then image files, then videos over same-stem
    # entries to avoid output filename collisions.
    selected: List[Tuple[str, str]] = []
    used_keys: set[str] = set()

    for path, obj_name in dir_items:
        key = str(obj_name).strip().lower()
        if not key or key in used_keys:
            continue
        selected.append((path, obj_name))
        used_keys.add(key)

    for path, obj_name in image_items:
        key = str(obj_name).strip().lower()
        if not key or key in used_keys:
            continue
        selected.append((path, obj_name))
        used_keys.add(key)

    for path, obj_name in video_items:
        key = str(obj_name).strip().lower()
        if not key:
            continue
        if key in used_keys:
            print(f"[GUI] Skip video input due to duplicate stem with directory: {path}")
            continue
        selected.append((path, obj_name))
        used_keys.add(key)

    return selected


class SequencePickerGui:
    """Simple launcher GUI for selecting which sequence to annotate."""

    def __init__(self, args_list: List[AppArgs], *, start_index: int = 0) -> None:
        self.args_list = list(args_list or [])
        self.window_name = "Annotation Sequence Picker"
        self.selected = int(max(0, min(len(self.args_list) - 1, int(start_index)))) if self.args_list else 0
        self.scroll = 0
        self.window_w = 1480
        self.window_h = 840
        self.line_h = 28

    @staticmethod
    def _out_path(args: AppArgs) -> str:
        return os.path.join(str(args.output_dir), str(args.ytvis_out))

    def _is_done(self, idx: int) -> bool:
        if not (0 <= int(idx) < len(self.args_list)):
            return False
        return os.path.exists(self._out_path(self.args_list[int(idx)]))

    def _jump_pending(self, direction: int) -> None:
        if not self.args_list:
            return
        direction = 1 if int(direction) >= 0 else -1
        n = len(self.args_list)
        cur = int(self.selected)
        for step in range(1, n + 1):
            nxt = (cur + direction * step) % n
            if not self._is_done(nxt):
                self.selected = int(nxt)
                return

    def _render(self) -> np.ndarray:
        canvas = np.full((self.window_h, self.window_w, 3), 24, dtype=np.uint8)
        if not self.args_list:
            cv2.putText(
                canvas,
                "No inputs found.",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            return canvas

        done_count = sum(1 for i in range(len(self.args_list)) if self._is_done(i))
        header = f"Sequences: {len(self.args_list)}   Done: {done_count}   Pending: {len(self.args_list) - done_count}"
        cv2.putText(canvas, header, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        help_line = "Enter: open selected | Up/Down: move | n/p: next/prev pending | r: refresh | esc: quit"
        cv2.putText(canvas, help_line, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)

        top_y = 86
        visible = max(1, (self.window_h - top_y - 18) // self.line_h)
        if self.selected < self.scroll:
            self.scroll = int(self.selected)
        if self.selected >= self.scroll + visible:
            self.scroll = int(self.selected - visible + 1)
        self.scroll = int(max(0, min(self.scroll, max(0, len(self.args_list) - visible))))

        for row, idx in enumerate(range(self.scroll, min(len(self.args_list), self.scroll + visible))):
            y = int(top_y + row * self.line_h)
            args = self.args_list[idx]
            input_path = str(args.input)
            display_name = str(getattr(args, "display_name", None) or os.path.basename(input_path))
            out_path = self._out_path(args)
            done = os.path.exists(out_path)
            is_sel = (idx == self.selected)
            if is_sel:
                cv2.rectangle(canvas, (8, y - 18), (self.window_w - 8, y + 8), (56, 88, 130), -1)
            status = "DONE" if done else "TODO"
            color = (120, 230, 140) if done else (235, 235, 235)
            line = f"{idx + 1:03d}. [{status}] {display_name}"
            cv2.putText(canvas, line[:200], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        return canvas

    def run(self) -> Optional[int]:
        keep_ratio_flag = int(getattr(cv2, "WINDOW_KEEPRATIO", 0))
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | keep_ratio_flag)
        try:
            cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
        except Exception:
            pass
        try:
            while True:
                vis = self._render()
                cv2.imshow(self.window_name, vis)
                key = cv2.waitKey(30) & 0xFF
                if key == 255:
                    continue
                print(f"[Picker][Key] {_format_key_event(key)}")
                if key in (27,):
                    return None
                if key in (13, 10):
                    return int(self.selected) if self.args_list else None
                if key in (ord("r"),):
                    continue
                if key in (ord("n"),):
                    self._jump_pending(+1)
                    continue
                if key in (ord("p"),):
                    self._jump_pending(-1)
                    continue
                # Up / Down arrows in OpenCV
                if key in (82, ord("k")):  # up
                    self.selected = int(max(0, self.selected - 1))
                    continue
                if key in (84, ord("j")):  # down
                    self.selected = int(min(len(self.args_list) - 1, self.selected + 1))
                    continue
        finally:
            cv2.destroyWindow(self.window_name)


class YTVISTrackStore:
    def __init__(self, num_frames: int, category_id: int) -> None:
        self.num_frames = int(max(0, num_frames))
        self.category_id = int(category_id)
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def _expand_to(self, n: int) -> None:
        n = int(max(0, n))
        if n <= self.num_frames:
            return
        self.num_frames = n

    def _ensure_track(self, track_id: int, *, category_id: Optional[int] = None) -> Dict[str, Any]:
        tid = int(track_id)
        tr = self.tracks.get(tid)
        if tr is None:
            tr = {
                "frames": {},
                "score": 1.0,
                "category_id": int(self.category_id if category_id is None else category_id),
            }
            self.tracks[tid] = tr
        elif category_id is not None:
            tr["category_id"] = int(category_id)
        return tr

    def set_track_category(self, track_id: int, category_id: int) -> None:
        tr = self._ensure_track(int(track_id), category_id=int(category_id))
        tr["category_id"] = int(category_id)

    def update_frame(
        self,
        frame_idx: int,
        masks: Dict[int, np.ndarray],
        track_category_map: Optional[Dict[int, int]] = None,
    ) -> None:
        frame_idx = int(frame_idx)
        self._expand_to(frame_idx + 1)
        # Overwrite semantics for this frame: clear existing sparse entries first.
        for tr in self.tracks.values():
            tr_frames = tr.get("frames", {})
            if isinstance(tr_frames, dict) and frame_idx in tr_frames:
                del tr_frames[frame_idx]
        for track_id, mask in masks.items():
            if mask is None:
                continue
            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                continue
            bbox = bbox_xywh_from_mask(mask_bool)
            if bbox is None:
                continue
            cat_id = None
            if track_category_map is not None:
                cat_id = track_category_map.get(int(track_id))
            tr = self._ensure_track(int(track_id), category_id=cat_id)
            tr_frames = tr.setdefault("frames", {})
            tr_frames[frame_idx] = {
                "segmentation": mask_to_coco_rle(mask_bool),
                "bbox": [int(v) for v in bbox],
                "area": int(np.count_nonzero(mask_bool)),
            }

    def delete_track(self, track_id: int) -> None:
        self.tracks.pop(int(track_id), None)

    def trim_track_from_frame(self, track_id: int, frame_idx: int) -> None:
        """Keep history before frame_idx and remove entries from frame_idx onward."""
        tid = int(track_id)
        start = int(frame_idx)
        tr = self.tracks.get(tid)
        if tr is None:
            return
        tr_frames = tr.get("frames", {})
        if not isinstance(tr_frames, dict):
            return
        keys_to_drop = [k for k in list(tr_frames.keys()) if int(k) >= start]
        for k in keys_to_drop:
            tr_frames.pop(k, None)
        if not tr_frames:
            self.tracks.pop(tid, None)

    def masks_at_frame(self, frame_idx: int) -> Dict[int, np.ndarray]:
        """Reconstruct visible masks for one stored frame without SAM2 inference."""
        result: Dict[int, np.ndarray] = {}
        for track_id, track in self.tracks.items():
            if not isinstance(track, dict):
                continue
            frame_entry = track.get("frames", {}).get(int(frame_idx))
            if not isinstance(frame_entry, dict):
                continue
            mask = coco_rle_to_mask(frame_entry.get("segmentation", {}))
            if mask.size > 0 and np.any(mask):
                result[int(track_id)] = mask
        return result

    def rename_track(self, old_id: int, new_id: int) -> Tuple[bool, str]:
        old_id = int(old_id)
        new_id = int(new_id)
        if old_id == new_id:
            return True, "same id"
        if old_id not in self.tracks:
            return False, f"old id {old_id} not found in track store"
        if new_id in self.tracks:
            return False, f"new id {new_id} already exists in track store"
        self.tracks[new_id] = self.tracks.pop(old_id)
        return True, "renamed"

    def export_predictions(self, video_id: int) -> List[Dict[str, Any]]:
        preds: List[Dict[str, Any]] = []
        for tid in sorted(self.tracks.keys()):
            tr = self.tracks[tid]
            tr_frames = tr.get("frames", {})
            if not isinstance(tr_frames, dict) or not tr_frames:
                continue
            segs = [None] * self.num_frames
            bboxes = [[0, 0, 0, 0] for _ in range(self.num_frames)]
            areas = [0] * self.num_frames
            for fi, item in tr_frames.items():
                idx = int(fi)
                if idx < 0 or idx >= self.num_frames:
                    continue
                if not isinstance(item, dict):
                    continue
                segs[idx] = item.get("segmentation")
                bboxes[idx] = item.get("bbox", [0, 0, 0, 0])
                areas[idx] = int(item.get("area", 0))
            preds.append(
                {
                    "video_id": int(video_id),
                    "category_id": int(tr["category_id"]),
                    "segmentations": segs,
                    "bboxes": bboxes,
                    "areas": areas,
                    "score": float(tr.get("score", 1.0)),
                    "track_id": int(tid),
                }
            )
        return preds


class InteractiveSam2Gui:
    def __init__(self, args: AppArgs) -> None:
        self.args = args
        self.frame_source = StreamFrameSource(args.input, args.max_frames)
        first = self.frame_source.read_next()
        if first is None:
            raise RuntimeError(f"No frames found from input: {args.input}")
        self.current_frame, self.current_frame_name = first

        self.frame_h, self.frame_w = self.current_frame.shape[:2]
        self.total_frames = self.frame_source.total_frames
        self.frame_idx = 0
        # ``frame_idx`` is the immutable dataset timeline index.  After going
        # back to edit an earlier frame we create a fresh streaming predictor,
        # whose local index starts at zero again.
        self.predictor_frame_idx = 0
        self.max_frame_seen = 1
        self.history_mode = False
        self.window_name = "SAM2 Interactive GUI"

        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"torch is required: {exc}") from exc
        self._torch = torch

        if args.device:
            if args.device.startswith("cuda") and not self._torch.cuda.is_available():
                print("[Warn] CUDA requested but unavailable. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = args.device
        else:
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.offload_video_to_cpu = bool(args.offload_video_to_cpu)
        self.offload_state_to_cpu = bool(args.offload_state_to_cpu)

        self.sam2_cfg = str(args.sam2_model_cfg)
        self.sam2_ckpt = _resolve_under(SAM2_ROOT, str(args.sam2_checkpoint))
        from sam2_realtime.sam2.build_sam import build_sam2_realtime_predictor

        self.predictor = build_sam2_realtime_predictor(
            self.sam2_cfg,
            self.sam2_ckpt,
            device=self.device,
        )
        with self._inference_context():
            self.inference_state = self.predictor.init_state(
                self.current_frame,
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )

        self.object_ids: List[int] = []
        self.current_masks: Dict[int, np.ndarray] = {}
        self.active_obj_id: Optional[int] = None
        self.next_obj_id = int(max(1, args.start_id))
        self.next_hand_id = int(max(0, getattr(args, "hand_start_id", 0)))
        self.active_id_by_kind: Dict[str, Optional[int]] = {"object": None, "hand": None}
        self.track_kind_by_id: Dict[int, str] = {}

        # All interaction sizes are specified in *display* units rather than
        # source-frame pixels.  The source frame can be 720p or 4K, while the
        # OpenCV window remains comfortably usable at ``window_width``.
        self.window_width = int(max(320, args.window_width))
        self.source_px_per_display_px = float(self.frame_w) / float(self.window_width)

        self.prompt_states: Dict[int, PromptState] = {}
        # `e` selects an existing ID, then turns the video canvas into a
        # point-prompt editor for that exact ID.  Keep this separate from the
        # free click/box workflow and the brush-mask workflow.
        self.prompt_edit_mode = False
        self.prompt_edit_obj_id: Optional[int] = None
        self.edit_mode = False
        self.edit_mask: Optional[np.ndarray] = None
        self.brush_radius = self._display_distance(float(max(1, args.brush_radius)))
        self.paint_mode = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.dragging_box = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.box_drag_min_size = self._display_distance(7.0)

        # Interactive labeling is step-driven: advance only when user presses a key.
        self.playing = False
        use_fps = float(args.play_fps) if args.play_fps and args.play_fps > 0 else float(self.frame_source.fps)
        self.play_interval = 1.0 / max(use_fps, 1e-6)
        self.last_tick = time.time()
        self.state_window = int(max(0, args.state_window))
        self.gc_every = int(max(0, args.gc_every))
        self.track_store = YTVISTrackStore(num_frames=0, category_id=int(args.category_id))
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )

        self.output_dir = os.path.abspath(args.output_dir)
        ensure_dir(self.output_dir)
        self.ytvis_out_path = os.path.join(self.output_dir, args.ytvis_out)
        self.video_name = os.path.splitext(os.path.basename(os.path.abspath(args.input)))[0]
        self.transparent_crop_dir = os.path.join(self.output_dir, "transparent_crops", self.video_name)
        self.outline_only_dir = os.path.join(self.output_dir, "outline_only", self.video_name)
        self.outline_button_rect: Optional[Tuple[int, int, int, int]] = None
        # Keep contour thick enough for clear visibility when saving.
        self.outline_thickness = int(max(self._display_distance(3.0), round(min(self.frame_h, self.frame_w) / 520.0)))
        self.existing_track_count = self._count_existing_tracks(self.ytvis_out_path)
        if self.existing_track_count > 0:
            print(
                f"[GUI] Existing annotation file detected "
                f"({self.existing_track_count} tracks): {self.ytvis_out_path}"
            )

    @staticmethod
    def _count_existing_tracks(path: str) -> int:
        if not os.path.isfile(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return 0
        if not isinstance(payload, list):
            return 0
        return int(sum(1 for x in payload if isinstance(x, dict)))

    def _inference_context(self):
        """Use SAM2's recommended BF16 CUDA inference path when available."""
        contexts = ExitStack()
        inference_mode = getattr(self._torch, "inference_mode", None)
        if callable(inference_mode):
            contexts.enter_context(inference_mode())
        else:
            no_grad = getattr(self._torch, "no_grad", None)
            if callable(no_grad):
                contexts.enter_context(no_grad())
            else:
                contexts.enter_context(nullcontext())

        autocast = getattr(self._torch, "autocast", None)
        if str(self.device).startswith("cuda") and callable(autocast):
            contexts.enter_context(autocast(device_type="cuda", dtype=self._torch.bfloat16))
        return contexts

    def _resize_display_window(self) -> None:
        target_w = int(max(320, self.window_width))
        scale = float(target_w) / float(max(1, self.frame_w))
        target_h = int(max(240, round(self.frame_h * scale)))
        try:
            cv2.resizeWindow(self.window_name, target_w, target_h)
        except Exception:
            pass

    def _display_distance(self, logical_pixels: float, *, minimum: int = 1) -> int:
        """Convert a visual distance in the OpenCV window to source pixels."""
        return int(max(minimum, round(float(logical_pixels) * self.source_px_per_display_px)))

    def _display_font_scale(self, logical_scale: float) -> float:
        """Keep OpenCV text the same physical size across input resolutions."""
        return max(0.18, float(logical_scale) * self.source_px_per_display_px)

    def run(self) -> None:
        keep_ratio_flag = int(getattr(cv2, "WINDOW_KEEPRATIO", 0))
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | keep_ratio_flag)
        self._resize_display_window()
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            done = False
            while not done:
                if self.playing:
                    now = time.time()
                    if now - self.last_tick >= self.play_interval:
                        self.last_tick = now
                        if not self._step_forward():
                            self.playing = False

                vis = self._render()
                cv2.imshow(self.window_name, vis)
                key_raw = cv2.waitKeyEx(1)
                if int(key_raw) != -1:
                    done = self._handle_key(int(key_raw))
        finally:
            cv2.destroyAllWindows()
            self.frame_source.close()
            self._save_outputs()

    def _decode_masks(self, obj_ids: List[int], mask_logits: Any) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        if mask_logits is None:
            return out
        for i, obj_id in enumerate(obj_ids):
            if i >= len(mask_logits):
                break
            mask = (mask_logits[i] > 0.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).squeeze()
            if mask.shape[:2] != (self.frame_h, self.frame_w):
                mask = cv2.resize(mask, (self.frame_w, self.frame_h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask.astype(bool)
            if np.any(mask_bool):
                out[int(obj_id)] = mask_bool
        return out

    def _refresh_from_tracker(self) -> None:
        state_ids = self.inference_state.get("obj_ids", []) if isinstance(self.inference_state, dict) else []
        if not state_ids:
            self.object_ids = []
            self.current_masks = {}
            self._trim_predictor_state()
            return
        with self._inference_context():
            ret = self.predictor.get_mask(self.inference_state, self.predictor_frame_idx)
        if not isinstance(ret, (tuple, list)) or len(ret) < 3:
            self._trim_predictor_state()
            return
        if len(ret) >= 4:
            _, obj_ids, mask_logits, new_state = ret
            self.inference_state = new_state
        else:
            _, obj_ids, mask_logits = ret
        self.object_ids = [int(x) for x in obj_ids]
        self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        self._sync_next_id()
        self._ensure_active_id()
        self._trim_predictor_state()

    def _trim_predictor_state(self) -> None:
        if self.state_window <= 0:
            return
        clear_fn = getattr(self.predictor, "clear_old_frames", None)
        if not callable(clear_fn):
            return
        if self.predictor_frame_idx <= self.state_window:
            return
        min_valid = int(max(0, self.predictor_frame_idx - self.state_window))
        try:
            clear_fn(self.inference_state, min_valid)
        except Exception:
            pass

    def _maybe_gc(self) -> None:
        if self.gc_every <= 0:
            return
        if self.frame_idx <= 0:
            return
        if (self.frame_idx % self.gc_every) != 0:
            return
        gc.collect()
        if str(self.device).startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def _sync_next_id(self) -> None:
        max_obj_id = int(max(0, self.next_obj_id - 1))
        max_hand_id = int(max(-1, self.next_hand_id - 1))
        for tid in self._used_track_ids():
            kind = self._kind_for_track(int(tid))
            if kind == "hand":
                max_hand_id = max(max_hand_id, int(tid))
            else:
                max_obj_id = max(max_obj_id, int(tid))
        self.next_obj_id = max(int(self.next_obj_id), int(max(1, max_obj_id + 1)))
        self.next_hand_id = max(int(self.next_hand_id), int(max(0, max_hand_id + 1)))

    def _used_track_ids(self) -> set[int]:
        ids: set[int] = set()
        ids.update(int(x) for x in self.object_ids)
        ids.update(int(x) for x in self.current_masks.keys())
        ids.update(int(x) for x in self.track_store.tracks.keys())
        ids.update(int(x) for x in self.track_kind_by_id.keys())
        return ids

    def _category_for_kind(self, kind: str) -> int:
        return int(self.args.hand_category_id) if str(kind) == "hand" else int(self.args.category_id)

    def _kind_from_category(self, category_id: int) -> str:
        if int(category_id) == int(self.args.hand_category_id):
            return "hand"
        return "object"

    def _kind_for_track(self, track_id: int) -> str:
        tid = int(track_id)
        known = self.track_kind_by_id.get(tid)
        if known in {"hand", "object"}:
            return str(known)
        tr = self.track_store.tracks.get(tid)
        if isinstance(tr, dict):
            try:
                cat = int(tr.get("category_id", self.args.category_id))
                kind = self._kind_from_category(cat)
                self.track_kind_by_id[tid] = kind
                return kind
            except Exception:
                pass
        # Fallback heuristic: IDs below object start are treated as hand IDs.
        # This matches the default convention hand>=0, object>=100.
        if tid < int(max(1, self.args.start_id)):
            kind = "hand"
        else:
            kind = "object"
        self.track_kind_by_id[tid] = kind
        return kind

    def _register_track_kind(self, track_id: int, kind: str) -> str:
        k = "hand" if str(kind) == "hand" else "object"
        tid = int(track_id)
        self.track_kind_by_id[tid] = k
        self.track_store.set_track_category(tid, self._category_for_kind(k))
        return k

    def _current_track_category_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for tid in self.current_masks.keys():
            kind = self._kind_for_track(int(tid))
            mapping[int(tid)] = int(self._category_for_kind(kind))
        return mapping

    def _allocate_new_id(self, kind: str) -> int:
        kind = "hand" if str(kind) == "hand" else "object"
        used = self._used_track_ids()
        if kind == "hand":
            cand = int(max(0, self.next_hand_id))
            while cand in used:
                cand += 1
            self.next_hand_id = int(cand + 1)
            return int(cand)
        cand = int(max(1, self.next_obj_id))
        while cand in used:
            cand += 1
        self.next_obj_id = int(cand + 1)
        return int(cand)

    def _prompt_counts(self, obj_id: int) -> Tuple[int, int]:
        ps = self.prompt_states.get(int(obj_id))
        if ps is None:
            return 0, 0
        return int(len(ps.pos_points)), int(len(ps.neg_points))

    def _mask_area(self, obj_id: int) -> int:
        m = self.current_masks.get(int(obj_id))
        if m is None:
            return 0
        return int(np.count_nonzero(m))

    def _print_status(self) -> None:
        ids = sorted(set(int(x) for x in self.object_ids))
        print("=" * 72)
        print(
            f"[GUI] frame={self.frame_idx + 1}/{self.total_frames if self.total_frames is not None else '?'} "
            f"active={self.active_obj_id} next_obj_id={self.next_obj_id} next_hand_id={self.next_hand_id}"
        )
        print(
            f"[GUI] active(object)={self.active_id_by_kind.get('object')} "
            f"active(hand)={self.active_id_by_kind.get('hand')}"
        )
        if not ids:
            print("[GUI] objects: none")
            return
        print(f"[GUI] objects({len(ids)}):")
        for oid in ids:
            p_cnt, n_cnt = self._prompt_counts(int(oid))
            area = self._mask_area(int(oid))
            mark = "*" if self.active_obj_id is not None and int(oid) == int(self.active_obj_id) else " "
            kind = self._kind_for_track(int(oid))
            print(
                f"  {mark} id={int(oid):4d} kind={kind:6s} "
                f"area={int(area):6d} prompts=+{int(p_cnt)}/-{int(n_cnt)}"
            )

    def _set_active_object_by_id(
        self,
        target_id: int,
        *,
        create_if_missing: bool = True,
        kind: Optional[str] = None,
    ) -> bool:
        tid = int(target_id)
        if tid < 0:
            print("[GUI] ID must be >= 0")
            return False
        requested_kind = str(kind) if kind in {"hand", "object"} else None
        if requested_kind == "object" and tid < 1:
            print("[GUI] Object ID must be >= 1")
            return False
        known_ids = set(int(x) for x in self.object_ids) | set(int(x) for x in self.track_store.tracks.keys())
        if tid in known_ids:
            self.active_obj_id = int(tid)
            existing_kind = self._kind_for_track(int(tid))
            self.active_id_by_kind[existing_kind] = int(tid)
            self._ensure_active_id()
            print(f"[GUI] Active ID set to existing track: {tid} ({existing_kind})")
            return True
        if not create_if_missing:
            print(f"[GUI] ID {tid} not found")
            return False
        self.active_obj_id = int(tid)
        self.prompt_states.setdefault(int(tid), PromptState())
        use_kind = (
            str(kind)
            if kind in {"hand", "object"}
            else ("hand" if int(tid) < int(max(1, self.args.start_id)) else "object")
        )
        use_kind = self._register_track_kind(int(tid), use_kind)
        self.active_id_by_kind[use_kind] = int(tid)
        if use_kind == "hand":
            self.next_hand_id = max(int(self.next_hand_id), int(tid) + 1)
        else:
            self.next_obj_id = max(int(self.next_obj_id), int(tid) + 1)
        print(f"[GUI] Active ID set to NEW track: {tid} ({use_kind})")
        return True

    def _prompt_select_object_id(self) -> None:
        self._print_status()
        try:
            raw = input("[GUI] Enter track ID to activate (empty=cancel): ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw:
            return
        try:
            tid = int(raw)
        except Exception:
            print(f"[GUI] invalid id: {raw}")
            return
        self._set_active_object_by_id(tid, create_if_missing=True)

    def _prompt_edit_active_prompts(self) -> None:
        if self.active_obj_id is None:
            self._create_new_active_object()
        if self.active_obj_id is None:
            return
        oid = int(self.active_obj_id)
        self.prompt_states.setdefault(oid, PromptState())
        print(f"[GUI] Prompt editor for ID {oid}")
        print("[GUI] cmds: '+ x y', '- x y', 'del+ N', 'del- N', 'list', 'clear', 'apply', 'status', 'q'")
        while True:
            p_cnt, n_cnt = self._prompt_counts(oid)
            try:
                raw = input(f"[GUI][id={oid}] +{p_cnt}/-{n_cnt} > ").strip()
            except Exception as exc:
                print(f"[GUI] input failed: {exc}")
                return
            if not raw:
                continue
            cmd = raw.lower()
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd in {"status", "s"}:
                self._print_status()
                continue
            if cmd == "clear":
                self._clear_active_prompts()
                continue
            if cmd == "apply":
                self._apply_prompt_state(oid)
                continue
            if cmd == "list":
                ps = self.prompt_states.setdefault(oid, PromptState())
                print(f"[GUI] + prompts: {list(enumerate(ps.pos_points, start=1))}")
                print(f"[GUI] - prompts: {list(enumerate(ps.neg_points, start=1))}")
                continue
            if cmd == "pop+":
                ps = self.prompt_states.setdefault(oid, PromptState())
                if ps.pos_points:
                    ps.pos_points.pop()
                self._apply_prompt_state(oid)
                continue
            if cmd == "pop-":
                ps = self.prompt_states.setdefault(oid, PromptState())
                if ps.neg_points:
                    ps.neg_points.pop()
                self._apply_prompt_state(oid)
                continue
            toks = raw.split()
            if len(toks) == 2 and toks[0].lower() in {"del+", "del-"}:
                try:
                    prompt_index = int(toks[1]) - 1
                    if prompt_index < 0:
                        raise ValueError()
                except Exception:
                    print("[GUI] use a 1-based prompt index, e.g. del+ 1")
                    continue
                ps = self.prompt_states.setdefault(oid, PromptState())
                prompts = ps.pos_points if toks[0].lower() == "del+" else ps.neg_points
                if prompt_index >= len(prompts):
                    print(f"[GUI] prompt index out of range: {prompt_index + 1}")
                    continue
                prompts.pop(prompt_index)
                self._apply_prompt_state(oid)
                continue
            if len(toks) == 3 and toks[0] in {"+", "-"}:
                try:
                    x = int(float(toks[1]))
                    y = int(float(toks[2]))
                except Exception:
                    print("[GUI] invalid coordinates")
                    continue
                x = int(np.clip(x, 0, self.frame_w - 1))
                y = int(np.clip(y, 0, self.frame_h - 1))
                ps = self.prompt_states.setdefault(oid, PromptState())
                if toks[0] == "+":
                    ps.pos_points.append((x, y))
                else:
                    ps.neg_points.append((x, y))
                self._apply_prompt_state(oid)
                continue
            print("[GUI] unknown cmd. use '+ x y', '- x y', del+ N, del- N, list, clear, apply, status, q")

    def _prompt_edit_track_prompts(self) -> None:
        """Select an existing ID, then edit that mask with mouse prompts."""
        if self.prompt_edit_mode:
            self.prompt_edit_mode = False
            self.prompt_edit_obj_id = None
            print("[GUI] Mouse prompt edit closed.")
            return
        self._print_status()
        try:
            raw = input("[GUI] Mouse-edit mask for track ID (empty=cancel): ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw:
            return
        try:
            track_id = int(raw)
        except Exception:
            print(f"[GUI] invalid id: {raw}")
            return
        if not self._set_active_object_by_id(track_id, create_if_missing=False):
            return
        if int(track_id) not in self.object_ids:
            print(f"[GUI] ID {track_id} has no visible mask on this frame.")
            return
        self.edit_mode = False
        self.edit_mask = None
        self.paint_mode = 0
        self.dragging_box = False
        self.drag_start = None
        self.drag_current = None
        self.prompt_edit_mode = True
        self.prompt_edit_obj_id = int(track_id)
        print(
            f"[GUI] Mouse prompt edit: ID {track_id}. "
            "Left-click=positive, right-click=negative, e=finish."
        )

    def _prompt_new_object_id(self) -> None:
        self._print_status()
        try:
            raw = input("[GUI] New object ID (empty=use next_obj_id): ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw:
            self._create_new_active_object()
            return
        try:
            nid = int(raw)
        except Exception:
            print(f"[GUI] invalid id: {raw}")
            return
        if nid < 1:
            print("[GUI] object id must be >= 1")
            return
        if self._set_active_object_by_id(nid, create_if_missing=True, kind="object"):
            self.prompt_states[int(nid)] = PromptState()

    def _ensure_active_id(self) -> None:
        if self.active_obj_id in self.object_ids:
            kind = self._kind_for_track(int(self.active_obj_id))
            self.active_id_by_kind[kind] = int(self.active_obj_id)
            return
        if self.object_ids:
            self.active_obj_id = int(sorted(self.object_ids)[0])
            kind = self._kind_for_track(int(self.active_obj_id))
            self.active_id_by_kind[kind] = int(self.active_obj_id)
        else:
            self.active_obj_id = None

    def _invalidate_future_annotations(self, start_frame: int) -> None:
        """Discard masks that will be recomputed by forward-only tracking."""
        start = int(max(0, start_frame))
        for track_id in list(self.track_store.tracks.keys()):
            self.track_store.trim_track_from_frame(int(track_id), start)
        callback = getattr(self, "_on_future_annotations_invalidated", None)
        if callable(callback):
            callback(start)

    def _prepare_history_for_forward_tracking(self) -> None:
        """Seed a new streaming state from masks visible on the selected frame."""
        if not self.history_mode:
            return
        seed_masks = self.track_store.masks_at_frame(self.frame_idx)
        with self._inference_context():
            self.inference_state = self.predictor.init_state(
                self.current_frame,
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )
            self.predictor_frame_idx = 0
            for track_id, mask in sorted(seed_masks.items()):
                self.predictor.add_new_mask(
                    self.inference_state,
                    frame_idx=self.predictor_frame_idx,
                    obj_id=int(track_id),
                    mask=mask.astype(bool),
                )
        self.current_masks = seed_masks
        self.object_ids = sorted(seed_masks)
        self.prompt_states.clear()
        self.history_mode = False
        self._invalidate_future_annotations(self.frame_idx + 1)
        self._sync_next_id()
        self._ensure_active_id()
        print(
            f"[GUI] Resumed forward tracking from frame {self.frame_idx + 1}; "
            f"future masks will be recomputed ({len(self.object_ids)} visible tracks)."
        )

    def _step_backward(self) -> bool:
        target = int(self.frame_idx - 1)
        if target < 0:
            print("[GUI] Already at the first frame.")
            return False
        item = self.frame_source.read_at(target)
        if item is None:
            print(f"[GUI] Cannot load previous frame {target + 1}.")
            return False
        self.frame_idx = target
        self.current_frame, self.current_frame_name = item
        self.current_masks = self.track_store.masks_at_frame(self.frame_idx)
        self.object_ids = sorted(self.current_masks)
        self.prompt_states.clear()
        self.prompt_edit_mode = False
        self.prompt_edit_obj_id = None
        self.edit_mode = False
        self.edit_mask = None
        self.playing = False
        self.history_mode = True
        self._sync_next_id()
        # Historical frames are review/edit checkpoints.  Do not silently
        # select the first visible track: a mouse click must never choose a
        # target or start changing a mask until the annotator selects an ID.
        self.active_obj_id = None
        self.active_id_by_kind = {"object": None, "hand": None}
        print(
            f"[GUI] Moved to previous frame {self.frame_idx + 1}. "
            "Select an ID (g or [ ]) before adding prompts, then use Space to propagate forward."
        )
        return True

    def _step_forward(self) -> bool:
        self.prompt_edit_mode = False
        self.prompt_edit_obj_id = None
        self._prepare_history_for_forward_tracking()
        step_start = time.perf_counter()
        next_item = self.frame_source.read_next()
        if next_item is None:
            return False
        frame_load_elapsed = time.perf_counter() - step_start
        self.frame_idx += 1
        self.predictor_frame_idx += 1
        self.max_frame_seen = max(int(self.max_frame_seen), int(self.frame_idx + 1))
        self.current_frame, self.current_frame_name = next_item
        self.prompt_states.clear()
        self.edit_mode = False
        self.edit_mask = None
        with self._inference_context():
            self.inference_state = self.predictor.add_frame(
                self.inference_state,
                self.current_frame,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
        backbone_elapsed = time.perf_counter() - step_start - frame_load_elapsed
        tracking_start = time.perf_counter()
        self._refresh_from_tracker()
        tracking_elapsed = time.perf_counter() - tracking_start
        self._maybe_gc()
        total_elapsed = time.perf_counter() - step_start
        print(
            f"[GUI][Perf] frame {self.frame_idx + 1}: "
            f"load={frame_load_elapsed * 1000:.0f}ms "
            f"backbone={backbone_elapsed * 1000:.0f}ms "
            f"track+decode={tracking_elapsed * 1000:.0f}ms "
            f"total={total_elapsed * 1000:.0f}ms "
            f"tracks={len(self.object_ids)}"
        )
        return True

    def _create_new_active_object(self, kind: str = "object") -> int:
        kind = "hand" if str(kind) == "hand" else "object"
        new_id = int(self._allocate_new_id(kind))
        self.active_obj_id = int(new_id)
        self.active_id_by_kind[kind] = int(new_id)
        self.prompt_states[self.active_obj_id] = PromptState()
        self._register_track_kind(int(new_id), kind)
        print(f"[GUI] Active ID set to new {kind}: {self.active_obj_id}")
        return int(new_id)

    def _run_click_update(self, obj_id: int, kind: Optional[str] = None) -> None:
        self._prepare_history_for_forward_tracking()
        obj_id = int(obj_id)
        use_kind = self._register_track_kind(obj_id, kind or self._kind_for_track(obj_id))
        self.active_id_by_kind[use_kind] = int(obj_id)
        self.active_obj_id = int(obj_id)
        ps = self.prompt_states.get(obj_id)
        if ps is None:
            return
        all_points = list(ps.pos_points) + list(ps.neg_points)
        if not all_points:
            return
        labels = [1] * len(ps.pos_points) + [0] * len(ps.neg_points)
        points_np = np.array(all_points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)

        with self._inference_context():
            ret = self.predictor.add_new_points_or_box(
                self.inference_state,
                frame_idx=self.predictor_frame_idx,
                obj_id=int(obj_id),
                points=points_np,
                labels=labels_np,
                clear_old_points=True,
            )
        if not isinstance(ret, (tuple, list)) or len(ret) < 3:
            return
        if len(ret) >= 4:
            _, obj_ids, mask_logits, new_state = ret
            self.inference_state = new_state
        else:
            _, obj_ids, mask_logits = ret
        self.object_ids = [int(x) for x in obj_ids]
        self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        self._trim_predictor_state()
        self._sync_next_id()
        self._ensure_active_id()

    def _apply_prompt_state(self, obj_id: int) -> None:
        """Recompute one ID's current-frame mask after an editor mutation."""
        obj_id = int(obj_id)
        self.active_obj_id = obj_id
        pos_count, neg_count = self._prompt_counts(obj_id)
        if pos_count + neg_count > 0:
            self._run_click_update(obj_id)
        else:
            self._clear_active_prompts()

    def _run_box_update(self, obj_id: int, box_xyxy: List[int]) -> None:
        self._prepare_history_for_forward_tracking()
        obj_id = int(obj_id)
        use_kind = self._register_track_kind(obj_id, self._kind_for_track(obj_id))
        self.active_id_by_kind[use_kind] = int(obj_id)
        self.active_obj_id = int(obj_id)
        if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
            return
        try:
            x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        except Exception:
            return
        x1 = int(np.clip(x1, 0, self.frame_w - 1))
        x2 = int(np.clip(x2, 0, self.frame_w - 1))
        y1 = int(np.clip(y1, 0, self.frame_h - 1))
        y2 = int(np.clip(y2, 0, self.frame_h - 1))
        if x2 <= x1 or y2 <= y1:
            return

        box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
        with self._inference_context():
            ret = self.predictor.add_new_points_or_box(
                self.inference_state,
                frame_idx=self.predictor_frame_idx,
                obj_id=int(obj_id),
                box=box_np,
                clear_old_points=True,
            )
        if not isinstance(ret, (tuple, list)) or len(ret) < 3:
            return
        if len(ret) >= 4:
            _, obj_ids, mask_logits, new_state = ret
            self.inference_state = new_state
        else:
            _, obj_ids, mask_logits = ret
        self.object_ids = [int(x) for x in obj_ids]
        self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        # Box prompt is a replacement-style interaction; clear stored click prompts for this ID.
        self.prompt_states[int(obj_id)] = PromptState()
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        self._trim_predictor_state()
        self._sync_next_id()
        self._ensure_active_id()

    def _clear_active_prompts(self) -> None:
        self._prepare_history_for_forward_tracking()
        if self.active_obj_id is None:
            return
        state_ids = set(self.inference_state.get("obj_ids", [])) if isinstance(self.inference_state, dict) else set()
        if int(self.active_obj_id) not in state_ids:
            self.prompt_states[self.active_obj_id] = PromptState()
            return
        self.prompt_states[self.active_obj_id] = PromptState()
        try:
            with self._inference_context():
                ret = self.predictor.clear_all_prompts_in_frame(
                    self.inference_state,
                    frame_idx=self.predictor_frame_idx,
                    obj_id=int(self.active_obj_id),
                    need_output=True,
                )
            if isinstance(ret, (tuple, list)) and len(ret) >= 3:
                _, obj_ids, mask_logits = ret
                self.object_ids = [int(x) for x in obj_ids]
                self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        except Exception:
            pass
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        self._trim_predictor_state()

    def _delete_object_by_id(self, obj_id: int) -> None:
        self._prepare_history_for_forward_tracking()
        obj_id = int(obj_id)
        obj_kind = self._kind_for_track(int(obj_id))
        try:
            with self._inference_context():
                obj_ids, _ = self.predictor.remove_object(
                    self.inference_state,
                    obj_id=obj_id,
                    strict=False,
                    need_output=False,
                )
            self.object_ids = [int(x) for x in obj_ids]
        except Exception:
            self.object_ids = [x for x in self.object_ids if int(x) != obj_id]
        self.current_masks.pop(obj_id, None)
        self.prompt_states.pop(obj_id, None)
        if self.prompt_edit_obj_id == obj_id:
            self.prompt_edit_mode = False
            self.prompt_edit_obj_id = None
        self.track_kind_by_id.pop(int(obj_id), None)
        if self.active_id_by_kind.get(obj_kind) == int(obj_id):
            self.active_id_by_kind[obj_kind] = None
        # Keep masks before deletion frame, remove from current frame onward.
        self.track_store.trim_track_from_frame(obj_id, self.frame_idx)
        self._refresh_from_tracker()
        print(f"[GUI] Deleted track ID: {obj_id}")

    def _delete_active_object(self) -> None:
        if self.active_obj_id is None:
            return
        self._delete_object_by_id(int(self.active_obj_id))

    def _prompt_delete_object_id(self) -> None:
        self._print_status()
        try:
            raw = input("[GUI] Delete object ID(s), comma-separated (empty=cancel): ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw:
            return

        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        if not tokens:
            return

        delete_ids: List[int] = []
        invalid_tokens: List[str] = []
        seen: set[int] = set()
        for tok in tokens:
            try:
                did = int(tok)
                if did < 0:
                    raise ValueError("id must be >= 0")
            except Exception:
                invalid_tokens.append(tok)
                continue
            if did in seen:
                continue
            seen.add(did)
            delete_ids.append(did)

        if invalid_tokens:
            print(f"[GUI] invalid ids skipped: {', '.join(invalid_tokens)}")
        if not delete_ids:
            return

        for did in delete_ids:
            self._delete_object_by_id(int(did))

    def _rename_active_object(self, new_id: int) -> None:
        self._prepare_history_for_forward_tracking()
        if self.active_obj_id is None:
            return
        old_id = int(self.active_obj_id)
        old_kind = self._kind_for_track(int(old_id))
        new_id = int(new_id)
        if new_id < 0:
            print("[GUI] New ID must be >= 0")
            return
        if old_kind == "object" and new_id < 1:
            print("[GUI] Object ID must be >= 1")
            return
        if old_id == new_id:
            return
        if new_id in self.object_ids:
            print(f"[GUI] New ID {new_id} already exists")
            return
        if new_id in self.track_store.tracks:
            print(f"[GUI] New ID {new_id} already exists in saved tracks")
            return
        old_mask = self.current_masks.get(old_id)
        if old_mask is None or not np.any(old_mask):
            print("[GUI] Cannot rename: active object has no visible mask on current frame")
            return

        try:
            with self._inference_context():
                self.predictor.remove_object(
                    self.inference_state,
                    obj_id=old_id,
                    strict=False,
                    need_output=False,
                )
        except Exception:
            pass

        with self._inference_context():
            ret = self.predictor.add_new_mask(
                self.inference_state,
                frame_idx=self.predictor_frame_idx,
                obj_id=new_id,
                mask=old_mask,
            )
        if isinstance(ret, (tuple, list)) and len(ret) >= 3:
            _, obj_ids, mask_logits = ret
            self.object_ids = [int(x) for x in obj_ids]
            self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        else:
            self._refresh_from_tracker()

        ok, msg = self.track_store.rename_track(old_id, new_id)
        if not ok:
            print(f"[GUI] Track store rename warning: {msg}")
        if old_id in self.prompt_states:
            self.prompt_states[new_id] = self.prompt_states.pop(old_id)
        if self.prompt_edit_obj_id == old_id:
            self.prompt_edit_obj_id = int(new_id)
        self.track_kind_by_id.pop(int(old_id), None)
        self._register_track_kind(int(new_id), old_kind)
        self.active_id_by_kind[old_kind] = int(new_id)
        self.active_obj_id = new_id
        self._sync_next_id()
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        print(f"[GUI] Renamed ID {old_id} -> {new_id}")

    def _prompt_rename_ids(self) -> None:
        self._print_status()
        try:
            raw_old = input("[GUI] Rename old ID (empty=cancel): ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw_old:
            return
        try:
            old_id = int(raw_old)
        except Exception:
            print(f"[GUI] invalid old id: {raw_old}")
            return
        if not self._set_active_object_by_id(old_id, create_if_missing=False):
            return
        try:
            raw_new = input(f"[GUI] New ID for {old_id}: ").strip()
        except Exception as exc:
            print(f"[GUI] input failed: {exc}")
            return
        if not raw_new:
            return
        try:
            new_id = int(raw_new)
        except Exception:
            print(f"[GUI] invalid new id: {raw_new}")
            return
        self._rename_active_object(int(new_id))

    def _start_edit_mode(self) -> None:
        if self.active_obj_id is None:
            print("[GUI] Select an active ID first")
            return
        base_mask = self.current_masks.get(int(self.active_obj_id))
        if base_mask is None:
            print("[GUI] Active ID has no mask on this frame")
            return
        self.edit_mask = base_mask.copy().astype(bool)
        self.edit_mode = True
        self.paint_mode = 0

    def _cancel_edit_mode(self) -> None:
        self.edit_mode = False
        self.edit_mask = None
        self.paint_mode = 0

    def _apply_edit_mask(self) -> None:
        self._prepare_history_for_forward_tracking()
        if not self.edit_mode or self.edit_mask is None or self.active_obj_id is None:
            return
        obj_id = int(self.active_obj_id)
        obj_kind = self._register_track_kind(int(obj_id), self._kind_for_track(int(obj_id)))
        self.active_id_by_kind[obj_kind] = int(obj_id)
        with self._inference_context():
            ret = self.predictor.add_new_mask(
                self.inference_state,
                frame_idx=self.predictor_frame_idx,
                obj_id=obj_id,
                mask=self.edit_mask.astype(bool),
            )
        if isinstance(ret, (tuple, list)) and len(ret) >= 3:
            _, obj_ids, mask_logits = ret
            self.object_ids = [int(x) for x in obj_ids]
            self.current_masks = self._decode_masks(self.object_ids, mask_logits)
        else:
            self._refresh_from_tracker()
        self.prompt_states[obj_id] = PromptState()
        self.track_store.update_frame(
            self.frame_idx,
            self.current_masks,
            track_category_map=self._current_track_category_map(),
        )
        self._cancel_edit_mode()
        print(f"[GUI] Applied edited mask for ID {obj_id}")
        self._trim_predictor_state()

    def _paint_edit_mask(self, x: int, y: int, fill_value: bool) -> None:
        if self.edit_mask is None:
            return
        rr = int(max(1, self.brush_radius))
        cv2.circle(self.edit_mask, (int(x), int(y)), rr, int(fill_value), -1)

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Optional[Tuple[int, int, int, int]]) -> bool:
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return int(x1) <= int(x) <= int(x2) and int(y1) <= int(y) <= int(y2)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _userdata: Any) -> None:
        self.mouse_x = int(x)
        self.mouse_y = int(y)

        if event == cv2.EVENT_LBUTTONDOWN and self._point_in_rect(x, y, self.outline_button_rect):
            self._save_outline_only_image()
            return

        if self.edit_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.paint_mode = 1
                self._paint_edit_mask(x, y, True)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.paint_mode = -1
                self._paint_edit_mask(x, y, False)
            elif event == cv2.EVENT_MOUSEMOVE and self.paint_mode != 0:
                if self.paint_mode > 0:
                    self._paint_edit_mask(x, y, True)
                else:
                    self._paint_edit_mask(x, y, False)
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self.paint_mode = 0
            return

        if self.prompt_edit_mode:
            target_id = self.prompt_edit_obj_id
            if target_id is None or int(target_id) not in self.object_ids:
                print("[GUI] Mouse prompt edit closed: no active ID.")
                self.prompt_edit_mode = False
                self.prompt_edit_obj_id = None
                return
            obj_id = int(target_id)
            kind = self._kind_for_track(obj_id)
            ps = self.prompt_states.setdefault(obj_id, PromptState())
            if event == cv2.EVENT_LBUTTONDOWN:
                ps.pos_points.append((int(x), int(y)))
                self._run_click_update(obj_id, kind=kind)
            elif event == cv2.EVENT_RBUTTONDOWN:
                ps.neg_points.append((int(x), int(y)))
                self._run_click_update(obj_id, kind=kind)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.active_obj_id is None:
                print("[GUI] Prompt ignored: choose or create an ID first (n/h, [ ], or g).")
                return
            self.dragging_box = True
            self.drag_start = (int(x), int(y))
            self.drag_current = (int(x), int(y))
            return
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_box:
            self.drag_current = (int(x), int(y))
            return
        elif event == cv2.EVENT_LBUTTONUP and self.dragging_box:
            sx, sy = self.drag_start if self.drag_start is not None else (int(x), int(y))
            ex, ey = int(x), int(y)
            self.dragging_box = False
            self.drag_start = None
            self.drag_current = None

            if self.active_obj_id is None:
                print("[GUI] Prompt ignored: no active ID.")
                return
            oid = int(self.active_obj_id)
            prompt_kind = self._kind_for_track(oid)
            if max(abs(ex - sx), abs(ey - sy)) >= int(self.box_drag_min_size):
                x1, x2 = min(sx, ex), max(sx, ex)
                y1, y2 = min(sy, ey), max(sy, ey)
                self._run_box_update(oid, [int(x1), int(y1), int(x2), int(y2)])
            else:
                ps = self.prompt_states.setdefault(oid, PromptState())
                ps.pos_points.append((int(ex), int(ey)))
                self._run_click_update(oid, kind=prompt_kind)
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.active_obj_id is None:
                print("[GUI] Negative prompt ignored: choose or create an ID first (n/h, [ ], or g).")
                return
            oid = int(self.active_obj_id)
            prompt_kind = self._kind_for_track(oid)
            ps = self.prompt_states.setdefault(oid, PromptState())
            ps.neg_points.append((int(x), int(y)))
            self._run_click_update(oid, kind=prompt_kind)

    def _handle_key(self, key: int) -> bool:
        key_raw = int(key)
        self._log_key(key_raw)
        key = int(key_raw & 0xFF)
        key_has_modifier = bool(key_raw != key)

        # quit
        if key in (27, ord("q")):
            return True

        # frame step
        if key == ord(" "):
            self._step_forward()
            return False
        if key == ord("."):
            self._step_forward()
            return False
        if key == ord(","):
            self._step_backward()
            return False

        # ID select
        if key == ord("]"):
            ids = sorted(self.object_ids)
            if ids:
                if self.active_obj_id not in ids:
                    self.active_obj_id = ids[0]
                else:
                    k = ids.index(self.active_obj_id)
                    self.active_obj_id = ids[(k + 1) % len(ids)]
            return False
        if key == ord("["):
            ids = sorted(self.object_ids)
            if ids:
                if self.active_obj_id not in ids:
                    self.active_obj_id = ids[0]
                else:
                    k = ids.index(self.active_obj_id)
                    self.active_obj_id = ids[(k - 1) % len(ids)]
            return False

        # new / delete / rename / clear clicks
        if key == ord("n"):
            if key_has_modifier:
                self._create_new_active_object(kind="hand")
            else:
                self._create_new_active_object()
            return False
        if key == ord("N"):
            self._create_new_active_object(kind="hand")
            return False
        if key == ord("h"):
            self._create_new_active_object(kind="hand")
            return False
        if key == ord("d"):
            self._prompt_delete_object_id()
            return False
        if key == ord("x"):
            self._clear_active_prompts()
            return False
        if key == ord("c"):
            self._prompt_rename_ids()
            return False
        if key == ord("r"):
            if self.active_obj_id is not None:
                try:
                    raw = input(f"new id for {self.active_obj_id}: ").strip()
                    if raw:
                        self._rename_active_object(int(raw))
                except Exception as exc:
                    print(f"[GUI] rename failed: {exc}")
            return False
        if key == ord("g"):
            self._prompt_select_object_id()
            return False
        if key == ord("v"):
            self._print_status()
            return False
        if key == ord("p"):
            self.prompt_edit_mode = False
            self.prompt_edit_obj_id = None
            self._prompt_edit_active_prompts()
            return False

        # Prompt editor / brush edit mode
        if key == ord("e"):
            self._prompt_edit_track_prompts()
            return False
        if key == ord("b"):
            self.prompt_edit_mode = False
            self.prompt_edit_obj_id = None
            if self.edit_mode:
                self._cancel_edit_mode()
            else:
                self._start_edit_mode()
            return False
        if key == ord("a"):
            self._apply_edit_mask()
            return False
        if key == ord("z"):
            if self.edit_mode:
                self._start_edit_mode()
            return False
        if key in (ord("+"), ord("=")):
            self.brush_radius = min(self._display_distance(200.0), self.brush_radius + self._display_distance(2.0))
            return False
        if key in (ord("-"), ord("_")):
            self.brush_radius = max(self._display_distance(1.0), self.brush_radius - self._display_distance(2.0))
            return False

        # quick save
        if key == ord("s"):
            self._save_outputs(force=True)
            return False
        if key in (ord("t"), ord("T")):
            self._save_active_transparent_crop()
            return False
        if key in (ord("o"), ord("O")):
            self._save_outline_only_image()
            return False

        return False

    @staticmethod
    def _log_key(key_raw: int) -> None:
        print(f"[GUI][Key] {_format_key_event(key_raw)}")

    def _render_outline_only(self) -> np.ndarray:
        out = self.current_frame.copy()
        ids = sorted(set(self.object_ids) | set(self.current_masks.keys()))
        thick = int(max(3, self.outline_thickness))
        for oid in ids:
            mask = self.current_masks.get(oid)
            if mask is None:
                continue
            mask_u8 = mask.astype(np.uint8)
            if not np.any(mask_u8):
                continue
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            color = _id_color(int(oid))
            cv2.drawContours(out, contours, -1, color, thick, lineType=cv2.LINE_AA)
        return out

    def _draw_outline_save_button(self, out: np.ndarray) -> None:
        _, w = out.shape[:2]
        pad = self._display_distance(12.0)
        btn_h = self._display_distance(44.0)
        btn_w = self._display_distance(270.0)
        x2 = int(max(pad + 1, w - pad))
        x1 = int(max(pad, x2 - btn_w))
        y1 = int(pad)
        y2 = int(y1 + btn_h)
        self.outline_button_rect = (x1, y1, x2, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), (28, 68, 28), -1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 255, 220), self._display_distance(2.0))
        label = "Save Outline (o)"
        label_origin = (x1 + self._display_distance(12.0), y1 + self._display_distance(29.0))
        label_scale = self._display_font_scale(0.72)
        cv2.putText(
            out,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (235, 255, 235),
            self._display_distance(2.0),
            cv2.LINE_AA,
        )

    def _render(self) -> np.ndarray:
        base = self.current_frame.copy()
        out = base.copy()

        ids = sorted(set(self.object_ids) | set(self.current_masks.keys()))
        for oid in ids:
            mask = self.current_masks.get(oid)
            if mask is None:
                continue
            color = _id_color(int(oid))
            alpha = 0.35 if oid != self.active_obj_id else 0.5
            overlay = out.copy()
            overlay[mask] = color
            out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, self._display_distance(2.0))
            cx, cy = _mask_center(mask)
            kind = self._kind_for_track(int(oid))
            txt = f"id:{oid}({kind[0]})"
            if oid == self.active_obj_id:
                txt += " *"
            cv2.putText(
                out,
                txt,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._display_font_scale(0.76),
                color,
                self._display_distance(2.0),
                cv2.LINE_AA,
            )

        if self.edit_mode and self.edit_mask is not None and self.active_obj_id is not None:
            color = (0, 255, 255)
            overlay = out.copy()
            overlay[self.edit_mask.astype(bool)] = color
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0)
            cv2.circle(out, (self.mouse_x, self.mouse_y), self.brush_radius, color, self._display_distance(1.0))
        elif self.dragging_box and self.drag_start is not None and self.drag_current is not None:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            x1, x2 = min(int(sx), int(cx)), max(int(sx), int(cx))
            y1, y2 = min(int(sy), int(cy)), max(int(sy), int(cy))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), self._display_distance(2.0))
            cv2.putText(
                out,
                "BOX",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._display_font_scale(0.7),
                (0, 255, 255),
                self._display_distance(2.0),
                cv2.LINE_AA,
            )

        # draw click prompts
        for oid, ps in self.prompt_states.items():
            for (x, y) in ps.pos_points:
                cv2.circle(out, (int(x), int(y)), self._display_distance(6.0), (0, 255, 0), -1)
            for (x, y) in ps.neg_points:
                radius = self._display_distance(6.0)
                cv2.circle(out, (int(x), int(y)), radius, (0, 0, 255), -1)
                cv2.line(out, (int(x) - radius, int(y) - radius), (int(x) + radius, int(y) + radius), (0, 0, 255), self._display_distance(1.0))
                cv2.line(out, (int(x) - radius, int(y) + radius), (int(x) + radius, int(y) - radius), (0, 0, 255), self._display_distance(1.0))

        if self.total_frames is None:
            frame_text = f"{self.frame_idx + 1}/?"
        else:
            frame_text = f"{self.frame_idx + 1}/{self.total_frames}"
        if self.edit_mode:
            interaction_mode = "BRUSH"
            mouse_help = "Brush: left add | right erase | b cancel | a apply"
        elif self.prompt_edit_mode:
            interaction_mode = "PROMPT"
            mouse_help = "Prompt ID: left positive | right negative | e finish"
        else:
            interaction_mode = "CLICK"
            mouse_help = "Mouse: active ID only — left positive/box | right negative"
        lines = [
            f"Frame {frame_text} | {interaction_mode} | active ID: {self.active_obj_id if self.active_obj_id is not None else '-'}",
            (
                f"Object: {self.active_id_by_kind.get('object')} | Hand: {self.active_id_by_kind.get('hand')} | "
                f"next object: {self.next_obj_id} | next hand: {self.next_hand_id}"
            ),
            mouse_help,
            "Keys: Space/. next | , previous | n object | h hand | [ ] ID | e mouse prompts | d delete | s save | q exit",
            "More: p active prompts | b brush edit | a apply brush | g select | c rename | t crop | o outline",
        ]
        y0 = self._display_distance(30.0)
        line_height = self._display_distance(30.0)
        panel_height = y0 + len(lines) * line_height + self._display_distance(10.0)
        hud_overlay = out.copy()
        cv2.rectangle(hud_overlay, (0, 0), (out.shape[1], panel_height), (12, 16, 20), -1)
        out = cv2.addWeighted(hud_overlay, 0.76, out, 0.24, 0.0)
        for i, line in enumerate(lines):
            y = y0 + i * line_height
            origin = (self._display_distance(12.0), y)
            font_scale = self._display_font_scale(0.64)
            cv2.putText(out, line, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), self._display_distance(2.0), cv2.LINE_AA)

        self._draw_outline_save_button(out)
        return out

    def _save_active_transparent_crop(self) -> None:
        if self.active_obj_id is None:
            print("[GUI] Transparent save skipped: no active ID.")
            return
        oid = int(self.active_obj_id)
        mask = self.current_masks.get(oid)
        if mask is None:
            print(f"[GUI] Transparent save skipped: ID {oid} has no mask on this frame.")
            return

        mask_bool = mask.astype(bool)
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            print(f"[GUI] Transparent save skipped: ID {oid} mask is empty.")
            return

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        crop_bgr = self.current_frame[y1:y2, x1:x2]
        crop_alpha = (mask_bool[y1:y2, x1:x2].astype(np.uint8) * 255)

        if crop_bgr.size == 0:
            print(f"[GUI] Transparent save skipped: invalid crop for ID {oid}.")
            return

        rgba = np.dstack((crop_bgr, crop_alpha))
        ensure_dir(self.transparent_crop_dir)
        out_name = f"{self.video_name}_f{self.frame_idx:06d}_id{oid}.png"
        out_path = os.path.join(self.transparent_crop_dir, out_name)
        ok = bool(cv2.imwrite(out_path, rgba))
        if not ok:
            print(f"[GUI] Failed to save transparent crop: {out_path}")
            return
        print(f"[GUI] Saved transparent crop: {out_path}")

    def _save_outline_only_image(self) -> None:
        if not self.current_masks:
            print("[GUI] Outline save skipped: no masks on this frame.")
            return
        vis = self._render_outline_only()
        ensure_dir(self.outline_only_dir)
        out_name = f"{self.video_name}_f{self.frame_idx:06d}_outline.png"
        out_path = os.path.join(self.outline_only_dir, out_name)
        ok = bool(cv2.imwrite(out_path, vis))
        if not ok:
            print(f"[GUI] Failed to save outline image: {out_path}")
            return
        print(f"[GUI] Saved outline image: {out_path} (thickness={self.outline_thickness}px)")

    def _save_outputs(self, *, force: bool = False) -> None:
        preds = self.track_store.export_predictions(video_id=int(self.args.video_id))
        if (not force) and len(preds) == 0 and self.existing_track_count > 0:
            print(
                "[GUI][Warn] Current in-memory tracks are empty; "
                f"keeping existing file unchanged: {self.ytvis_out_path}"
            )
            return

        _write_json_atomic(self.ytvis_out_path, preds)
        self.existing_track_count = int(len(preds))
        num_hand_tracks = int(
            sum(1 for p in preds if int(p.get("category_id", -999999)) == int(self.args.hand_category_id))
        )
        num_object_tracks = int(len(preds) - num_hand_tracks)

        session_name = str(getattr(self.args, "session_meta_out", "interactive_session_meta.json"))
        session_path = os.path.join(self.output_dir, session_name)
        payload = {
            "video_name": self.video_name,
            "input": os.path.abspath(self.args.input),
            "num_frames_processed": int(self.max_frame_seen),
            "num_frames_total_hint": int(self.total_frames) if self.total_frames is not None else None,
            "video_id": int(self.args.video_id),
            "category_id": int(self.args.category_id),
            "object_category_id": int(self.args.category_id),
            "hand_category_id": int(self.args.hand_category_id),
            "num_tracks": int(len(preds)),
            "num_object_tracks": int(num_object_tracks),
            "num_hand_tracks": int(num_hand_tracks),
            "ytvis_output": os.path.abspath(self.ytvis_out_path),
            "last_frame_name": self.current_frame_name,
        }
        _write_json_atomic(session_path, payload)
        print(f"[GUI] Saved YTVIS predictions: {self.ytvis_out_path} ({len(preds)} tracks)")
        print(f"[GUI] Saved session metadata: {session_path}")


def main() -> None:
    raise SystemExit("Use the project launcher: python desktop_annotator.py --pick")


if __name__ == "__main__":
    main()
