#!/usr/bin/env bash
# Downsample extracted 30fps Aria JPEG sequences to 10fps while preserving layout.
set -euo pipefail

ARIA_RECORDING_ROOT="${ARIA_RECORDING_ROOT:-/workspace/shared/aria_recording}"
SOURCE_ROOT="$ARIA_RECORDING_ROOT/30fps_raw"
DESTINATION_ROOT="$ARIA_RECORDING_ROOT/10fps_sampled_sequences"
OVERWRITE=0
DRY_RUN=0
SAMPLE_ALL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sample_aria_frames_10fps.sh <participant_folder> [--overwrite] [--dry-run]
  bash scripts/sample_aria_frames_10fps.sh --all [--overwrite] [--dry-run]

Samples the extracted 30fps_raw JPEG frames at indices 0, 3, 6, ... and writes
renumbered JPEGs to 10fps_sampled_sequences/<participant>/<task>/rgb/. The
output uses hard links: annotation reads the same immutable JPEG pixels without
duplicating the source data on disk.
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
  echo "Participant must be a single folder name under 30fps_raw/." >&2
  exit 2
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
  while IFS= read -r -d '' source_rgb_dir; do
    found=1
    relative_path="${source_rgb_dir#"$source_dir"/}"
    output_dir="$destination_dir/$relative_path"
    source_count="$(find "$source_rgb_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print | wc -l)"
    expected_count=$(( (source_count + 2) / 3 ))
    output_count=0
    if [[ -d "$output_dir" ]]; then
      output_count="$(find "$output_dir" -maxdepth 1 -type f -iname '*.jpg' -print | wc -l)"
    fi
    if [[ "$output_count" -eq "$expected_count" && "$expected_count" -gt 0 && "$OVERWRITE" -ne 1 ]]; then
      echo "[skip] $participant_name/$relative_path ($output_count frames already exist)"
      ((skipped_total += 1))
      continue
    fi
    if [[ -d "$output_dir" && "$output_count" -gt 0 && "$OVERWRITE" -ne 1 ]]; then
      echo "Incomplete output exists; use --overwrite to replace it: $output_dir" >&2
      exit 1
    fi
    echo "[10 fps JPEG] $participant_name/$relative_path ($source_count -> $expected_count frames)"
    if [[ "$DRY_RUN" -ne 1 ]]; then
      temporary_dir="${output_dir}.partial.$$"
      rm -rf -- "$temporary_dir"
      mkdir -p "$temporary_dir"
      source_index=0
      output_index=0
      while IFS= read -r -d '' source_frame; do
        if (( source_index % 3 == 0 )); then
          printf -v output_name '%06d.jpg' "$output_index"
          ln -- "$source_frame" "$temporary_dir/$output_name"
          ((output_index += 1))
        fi
        ((source_index += 1))
      done < <(find "$source_rgb_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | sort -z)
      if [[ "$output_index" -ne "$expected_count" ]]; then
        echo "Unexpected output count for: $source_rgb_dir ($output_index != $expected_count)" >&2
        exit 1
      fi
      if [[ -d "$output_dir" ]]; then
        rm -rf -- "$output_dir"
      fi
      mkdir -p "$(dirname "$output_dir")"
      mv "$temporary_dir" "$output_dir"
    fi
    ((processed_total += 1))
  done < <(find "$source_dir" -type d -name rgb -print0 | sort -z)
  if [[ "$found" -eq 0 ]]; then
    echo "[warn] No rgb frame directories found under: $source_dir" >&2
  fi
done

echo "Done. processed=$processed_total skipped=$skipped_total output=$DESTINATION_ROOT"
