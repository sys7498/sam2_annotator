#!/usr/bin/env bash
# Sample Aria raw MP4s to 5 fps JPEG frame sequences while preserving layout.
set -euo pipefail

ARIA_RECORDING_ROOT="${ARIA_RECORDING_ROOT:-/workspace/shared/aria_recording}"
SOURCE_ROOT="$ARIA_RECORDING_ROOT/raw"
DESTINATION_ROOT="$ARIA_RECORDING_ROOT/5fps_sampled"
OVERWRITE=0
DRY_RUN=0
SAMPLE_ALL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sample_aria_recording_5fps.sh <participant_folder> [--overwrite] [--dry-run]
  bash scripts/sample_aria_recording_5fps.sh --all [--overwrite] [--dry-run]

Examples:
  bash scripts/sample_aria_recording_5fps.sh jdh
  bash scripts/sample_aria_recording_5fps.sh --all
  bash scripts/sample_aria_recording_5fps.sh jdh --overwrite

The default source is /workspace/shared/aria_recording/raw/<participant_folder>.
Output is written to /workspace/shared/aria_recording/5fps_sampled/<participant_folder>,
with every intermediate directory preserved. Each `rgb.mp4` becomes an `rgb/` folder
with `000000.jpg`, `000001.jpg`, ... at 5 fps. Set ARIA_RECORDING_ROOT to override this root.
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
    --all) SAMPLE_ALL=1 ;;
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

if [[ "$SAMPLE_ALL" -eq 1 && -n "$PARTICIPANT" ]]; then
  echo "Use either one participant folder or --all, not both." >&2
  exit 2
fi
if [[ "$SAMPLE_ALL" -ne 1 && ( -z "$PARTICIPANT" || "$PARTICIPANT" == */* || "$PARTICIPANT" == "." || "$PARTICIPANT" == ".." ) ]]; then
  echo "Participant must be a single folder name under raw/ (for example: jdh)." >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

if [[ "$SAMPLE_ALL" -eq 1 ]]; then
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
  destination_dir="$DESTINATION_ROOT/$participant_name"
  if [[ ! -d "$source_dir" ]]; then
    echo "Source participant folder not found: $source_dir" >&2
    exit 1
  fi
  echo "== Participant: $participant_name =="
  found=0
  while IFS= read -r -d '' source_path; do
    found=1
    relative_path="${source_path#"$source_dir"/}"
    output_dir="$destination_dir/${relative_path%.*}"
    if [[ -d "$output_dir" && "$OVERWRITE" -ne 1 ]]; then
      if find "$output_dir" -maxdepth 1 -type f -iname '*.jpg' -print -quit | grep -q .; then
        echo "[skip] $participant_name/$relative_path (JPEG frames already exist)"
        ((skipped_total += 1))
        continue
      fi
      echo "[repair] $participant_name/$relative_path (empty prior output directory)"
    fi
    echo "[5 fps JPEG] $participant_name/$relative_path"
    if [[ "$DRY_RUN" -ne 1 ]]; then
      mkdir -p "$(dirname "$output_dir")"
      temporary_dir="${output_dir}.partial.$$"
      rm -rf -- "$temporary_dir"
      mkdir -p "$temporary_dir"
      # ffmpeg otherwise consumes this loop's stdin (the NUL-delimited find
      # result) as interactive commands and stops after the first video.
      ffmpeg -nostdin -hide_banner -loglevel warning -y -i "$source_path" \
        -map 0:v:0 -an -vf "fps=5" \
        -q:v 2 -start_number 0 "$temporary_dir/%06d.jpg"
      if ! find "$temporary_dir" -maxdepth 1 -type f -iname '*.jpg' -print -quit | grep -q .; then
        echo "No JPEG frames were generated for: $source_path" >&2
        exit 1
      fi
      if [[ -d "$output_dir" ]]; then
        if [[ "$OVERWRITE" -ne 1 ]]; then
          echo "Output directory appeared while sampling: $output_dir" >&2
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
echo "Done. processed=$processed_total skipped=$skipped_total output=$DESTINATION_ROOT"
