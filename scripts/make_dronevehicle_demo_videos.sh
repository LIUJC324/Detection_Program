#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RGB_DIR="${RGB_DIR:-$PROJECT_ROOT/datasets/raw/dronevehicle/val_unpack/val/valimg}"
THERMAL_DIR="${THERMAL_DIR:-$PROJECT_ROOT/datasets/raw/dronevehicle/val_unpack/val/valimgr}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/demo_video}"
FPS="${FPS:-8}"
FRAME_COUNT="${FRAME_COUNT:-150}"
START_NUMBER="${START_NUMBER:-1}"

mkdir -p "$OUT_DIR"

RGB_VIDEO="$OUT_DIR/dronevehicle_rgb_demo.mp4"
THERMAL_VIDEO="$OUT_DIR/dronevehicle_thermal_demo.mp4"
SIDE_BY_SIDE_VIDEO="$OUT_DIR/dronevehicle_rgb_thermal_side_by_side.mp4"
SIDE_BY_SIDE_FLV="$OUT_DIR/dronevehicle_rgb_thermal_side_by_side.flv"

if [[ ! -d "$RGB_DIR" ]]; then
  echo "RGB_DIR not found: $RGB_DIR" >&2
  exit 1
fi

if [[ ! -d "$THERMAL_DIR" ]]; then
  echo "THERMAL_DIR not found: $THERMAL_DIR" >&2
  exit 1
fi

ffmpeg -y -hide_banner -loglevel error \
  -framerate "$FPS" \
  -start_number "$START_NUMBER" \
  -i "$RGB_DIR/%05d.jpg" \
  -frames:v "$FRAME_COUNT" \
  -c:v mpeg4 \
  -pix_fmt yuv420p \
  "$RGB_VIDEO"

ffmpeg -y -hide_banner -loglevel error \
  -framerate "$FPS" \
  -start_number "$START_NUMBER" \
  -i "$THERMAL_DIR/%05d.jpg" \
  -frames:v "$FRAME_COUNT" \
  -c:v mpeg4 \
  -pix_fmt yuv420p \
  "$THERMAL_VIDEO"

ffmpeg -y -hide_banner -loglevel error \
  -i "$RGB_VIDEO" \
  -i "$THERMAL_VIDEO" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" \
  -c:v mpeg4 \
  -pix_fmt yuv420p \
  "$SIDE_BY_SIDE_VIDEO"

ffmpeg -y -hide_banner -loglevel error \
  -i "$SIDE_BY_SIDE_VIDEO" \
  -c:v flv \
  "$SIDE_BY_SIDE_FLV"

echo "Generated demo videos:"
echo "  $RGB_VIDEO"
echo "  $THERMAL_VIDEO"
echo "  $SIDE_BY_SIDE_VIDEO"
echo "  $SIDE_BY_SIDE_FLV"
