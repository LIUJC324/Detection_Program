#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/datasets/raw/dronevehicle"
MIRROR_BASE="https://hf-mirror.com/datasets/McCheng/DroneVehicle/resolve/main"
DOWNLOADER="${PROJECT_ROOT}/scripts/download_http_file.py"

mkdir -p "${RAW_DIR}"

download_split() {
  local split="$1"
  local url="${MIRROR_BASE}/${split}.zip"
  local out="${RAW_DIR}/${split}.zip"

  echo "[download] ${split}: ${url}"
  python3 "${DOWNLOADER}" "${url}" "${out}"
  echo "[done] ${split}: ${out}"
}

if [[ "$#" -eq 0 ]]; then
  set -- train val test
fi

for split in "$@"; do
  case "${split}" in
    train|val|test)
      download_split "${split}"
      ;;
    *)
      echo "Unsupported split: ${split}" >&2
      echo "Usage: $0 [train] [val] [test]" >&2
      exit 1
      ;;
  esac
done
