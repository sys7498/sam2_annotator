#!/usr/bin/env bash
# Extract every source video frame to JPEG while preserving the Aria directory layout.
set -euo pipefail

ARIA_RECORDING_ROOT="${ARIA_RECORDING_ROOT:-/workspace/shared/aria_recording}"
SOURCE_ROOT="$ARIA_RECORDING_ROOT/raw"
OVERWRITE=0
DRY_RUN=0
EXTRACT_ALL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/extract_aria_recording_native_fps.sh <participant_folder> [--overwrite] [--dry-run]
  bash scripts/extract_aria_recording_native_fps.sh --all [--overwrite] [--dry-run]

Examples:
  bash scripts/extract_aria_recording_native_fps.sh jdh
  bash scripts/extract_aria_recording_native_fps.sh --all
  bash scripts/extract_aria_recording_native_fps.sh --all --overwrite

The source is /workspace/shared/aria_recording/raw/<participant_folder> by default.
Each video is decoded at its native FPS, with one JPEG per source video frame. Output
is /workspace/shared/aria_recording/<fps>fps_raw/<participant>/<intermediate path>/rgb/.
For example, a 30fps rgb.mp4 becomes 30fps_raw/jdh/example/rgb/000000.jpg.
Set ARIA_RECORDING_ROOT to use a different recording root.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PARTICIPANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --all) EXTRACT_ALL=1 ;;
    --help|-h) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$PARTICIPANT" ]]; then
        echo "Only one participant folder can be specified." >&2
        exit 2
      fi
      PARTICIPANT="$1"
      ;;
  esac
  shift
done

if [[ "$EXTRACT_ALL" -eq 1 && -n "$PARTICIPANT" ]]; then
  echo "Use either one participant folder or --all, not both." >&2
  exit 2
fi
if [[ "$EXTRACT_ALL" -ne 1 && ( -z "$PARTICIPANT" || "$PARTICIPANT" == */* || "$PARTICIPANT" == "." || "$PARTICIPANT" == ".." ) ]]; then
  echo "Participant must be a single folder name under raw/ (for example: jdh)." >&2
  exit 2
fi
for command in ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required. Install it with: sudo apt-get install -y ffmpeg" >&2
    exit 1
  fi
done

fps_label() {
  local source_path="$1"
  local rate label
  rate="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$source_path")"
  if [[ -z "$rate" || "$rate" == "0/0" ]]; then
    echo "Could not determine video FPS: $source_path" >&2
    return 1
  fi
  label="$(awk -v rate="$rate" 'BEGIN { split(rate, p, "/"); if (p[2] == 0) exit 1; fps = p[1] / p[2]; if (fps == int(fps)) printf "%d", fps; else { printf "%.3f", fps; sub(/0+$/, ""); sub(/\.$/, "") } }')"
  if [[ -z "$label" ]]; then
    echo "Could not normalize video FPS '$rate': $source_path" >&2
    return 1
  fi
  printf '%s\n' "$label"
}

if [[ "$EXTRACT_ALL" -eq 1 ]]; then
  mapfile -t PARTICIPANTS < <(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
else
  PARTICIPANTS=("$PARTICIPANT")
fi
if [[ "${#PARTICIPANTS[@]}" -eq 0 ]]; then
  echo "No participant folders found under: $SOURCE_ROOT" >&2
  exit 1
fi

processed_total=0
skipped_total=0
for participant_name in "${PARTICIPANTS[@]}"; do
  source_dir="$SOURCE_ROOT/$participant_name"
  if [[ ! -d "$source_dir" ]]; then
    echo "Source participant folder not found: $source_dir" >&2
    exit 1
  fi
  echo "== Participant: $participant_name =="
  found=0
  while IFS= read -r -d '' source_path; do
    found=1
    relative_path="${source_path#"$source_dir"/}"
    fps="$(fps_label "$source_path")"
    destination_root="$ARIA_RECORDING_ROOT/${fps}fps_raw"
    output_dir="$destination_root/$participant_name/${relative_path%.*}"
    if [[ -d "$output_dir" && "$OVERWRITE" -ne 1 ]]; then
      if find "$output_dir" -maxdepth 1 -type f -iname '*.jpg' -print -quit | grep -q .; then
        echo "[skip] $participant_name/$relative_path (${fps}fps frames already exist)"
        ((skipped_total += 1))
        continue
      fi
      echo "[repair] $participant_name/$relative_path (empty prior output directory)"
    fi
    echo "[native ${fps}fps JPEG] $participant_name/$relative_path -> ${fps}fps_raw/"
    if [[ "$DRY_RUN" -ne 1 ]]; then
      mkdir -p "$(dirname "$output_dir")"
      temporary_dir="${output_dir}.partial.$$"
      rm -rf -- "$temporary_dir"
      mkdir -p "$temporary_dir"
      # -vsync 0 writes one image for each decoded source frame without rate conversion.
      ffmpeg -nostdin -hide_banner -loglevel warning -y -i "$source_path" \
        -map 0:v:0 -an -vsync 0 -q:v 2 -start_number 0 "$temporary_dir/%06d.jpg"
      if ! find "$temporary_dir" -maxdepth 1 -type f -iname '*.jpg' -print -quit | grep -q .; then
        echo "No JPEG frames were generated for: $source_path" >&2
        exit 1
      fi
      if [[ -d "$output_dir" ]]; then
        if [[ "$OVERWRITE" -ne 1 ]]; then
          echo "Output directory appeared while extracting: $output_dir" >&2
          exit 1
        fi
        rm -rf -- "$output_dir"
      fi
      mv "$temporary_dir" "$output_dir"
    fi
    ((processed_total += 1))
  done < <(find "$source_dir" -type f ! -name '.*' ! -name '._*' -iname '*.mp4' -print0 | sort -z)
  if [[ "$found" -eq 0 ]]; then
    echo "[warn] No MP4 files found under: $source_dir" >&2
  fi
done

if [[ "$processed_total" -eq 0 && "$skipped_total" -eq 0 ]]; then
  exit 1
fi
echo "Done. processed=$processed_total skipped=$skipped_total output=$ARIA_RECORDING_ROOT/<fps>fps_raw"
