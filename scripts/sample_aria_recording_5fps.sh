#!/usr/bin/env bash
# Sample one participant's Aria raw MP4s to 5 fps while preserving subfolders.
set -euo pipefail

ARIA_RECORDING_ROOT="${ARIA_RECORDING_ROOT:-/workspace/shared/aria_recording}"
SOURCE_ROOT="$ARIA_RECORDING_ROOT/raw"
DESTINATION_ROOT="$ARIA_RECORDING_ROOT/5fps_sampled"
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sample_aria_recording_5fps.sh <participant_folder> [--overwrite] [--dry-run]

Examples:
  bash scripts/sample_aria_recording_5fps.sh jdh
  bash scripts/sample_aria_recording_5fps.sh jdh --overwrite

The default source is /workspace/shared/aria_recording/raw/<participant_folder>.
Output is written to /workspace/shared/aria_recording/5fps_sampled/<participant_folder>,
with every intermediate directory preserved. Set ARIA_RECORDING_ROOT to override this root.
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

if [[ -z "$PARTICIPANT" || "$PARTICIPANT" == */* || "$PARTICIPANT" == "." || "$PARTICIPANT" == ".." ]]; then
  echo "Participant must be a single folder name under raw/ (for example: jdh)." >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

SOURCE_DIR="$SOURCE_ROOT/$PARTICIPANT"
DESTINATION_DIR="$DESTINATION_ROOT/$PARTICIPANT"
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source participant folder not found: $SOURCE_DIR" >&2
  exit 1
fi

processed=0
skipped=0
while IFS= read -r -d '' source_path; do
  relative_path="${source_path#"$SOURCE_DIR"/}"
  output_path="$DESTINATION_DIR/${relative_path%.*}.mp4"
  if [[ -f "$output_path" && "$OVERWRITE" -ne 1 ]]; then
    echo "[skip] $relative_path (already exists)"
    ((skipped += 1))
    continue
  fi
  echo "[5 fps] $relative_path"
  if [[ "$DRY_RUN" -ne 1 ]]; then
    mkdir -p "$(dirname "$output_path")"
    ffmpeg -hide_banner -loglevel warning -y -i "$source_path" \
      -map 0:v:0 -an -vf "fps=5" \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart \
      "$output_path"
  fi
  ((processed += 1))
done < <(find "$SOURCE_DIR" -type f ! -name '.*' ! -name '._*' -iname '*.mp4' -print0 | sort -z)

if [[ "$processed" -eq 0 && "$skipped" -eq 0 ]]; then
  echo "No MP4 files found under: $SOURCE_DIR" >&2
  exit 1
fi
echo "Done. processed=$processed skipped=$skipped output=$DESTINATION_DIR"
