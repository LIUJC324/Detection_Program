#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
COMPONENT="${1:-both}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/lib/frp}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-10}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"

resolve_frp_version() {
  if [[ -n "${FRP_VERSION:-}" ]]; then
    printf '%s\n' "$FRP_VERSION"
    return
  fi

  curl -fsSL \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    https://api.github.com/repos/fatedier/frp/releases/latest \
    | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' \
    | head -n 1
}

resolve_arch() {
  case "$(uname -m)" in
    x86_64|amd64)
      printf 'amd64\n'
      ;;
    aarch64|arm64)
      printf 'arm64\n'
      ;;
    *)
      echo "unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
}

FRP_VERSION="$(resolve_frp_version)"
if [[ -z "$FRP_VERSION" ]]; then
  echo "failed to resolve FRP_VERSION from GitHub release API" >&2
  exit 1
fi

ARCH="$(resolve_arch)"
ARCHIVE="frp_${FRP_VERSION}_linux_${ARCH}.tar.gz"
DOWNLOAD_URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/${ARCHIVE}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$INSTALL_DIR" "$BIN_DIR"
curl -L --fail \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  -o "$TMP_DIR/$ARCHIVE" "$DOWNLOAD_URL"
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$TMP_DIR"
EXTRACTED_DIR="$TMP_DIR/frp_${FRP_VERSION}_linux_${ARCH}"

install_component() {
  local name="$1"
  install -m 0755 "$EXTRACTED_DIR/$name" "$INSTALL_DIR/$name"
  ln -sf "$INSTALL_DIR/$name" "$BIN_DIR/$name"
}

case "$COMPONENT" in
  client)
    install_component frpc
    ;;
  server)
    install_component frps
    ;;
  both)
    install_component frpc
    install_component frps
    ;;
  *)
    echo "unsupported component: $COMPONENT (expected client|server|both)" >&2
    exit 1
    ;;
esac

echo "installed frp ${FRP_VERSION} to $INSTALL_DIR"
echo "linked binaries in $BIN_DIR"
