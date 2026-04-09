#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST, for example 8.137.107.232}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/opt/rgbt_uav_detection}"
REMOTE_CONFIG_DIR="${REMOTE_CONFIG_DIR:-/etc/rgbt_uav_detection}"
REMOTE_SUDO="${REMOTE_SUDO:-}"

FRP_BIND_PORT="${FRP_BIND_PORT:-7000}"
REMOTE_MODEL_PORT="${REMOTE_MODEL_PORT:-18000}"
FRP_DASHBOARD_PORT="${FRP_DASHBOARD_PORT:-17500}"
FRP_DASHBOARD_USER="${FRP_DASHBOARD_USER:-admin}"
FRP_DASHBOARD_PASSWORD_FILE="${FRP_DASHBOARD_PASSWORD_FILE:-$HOME/.config/rgbt_uav_detection/frp_dashboard_password}"
FRP_AUTH_TOKEN_FILE="${FRP_AUTH_TOKEN_FILE:-$HOME/.config/rgbt_uav_detection/frp_auth_token}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-10}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"

generate_secret() {
  od -An -N 24 -tx1 /dev/urandom | tr -d ' \n'
}

if [[ -z "${FRP_AUTH_TOKEN:-}" ]]; then
  if [[ -f "$FRP_AUTH_TOKEN_FILE" ]]; then
    FRP_AUTH_TOKEN="$(tr -d '[:space:]' < "$FRP_AUTH_TOKEN_FILE")"
  else
    echo "set FRP_AUTH_TOKEN or create $FRP_AUTH_TOKEN_FILE first" >&2
    exit 1
  fi
fi

if [[ -z "${FRP_DASHBOARD_PASSWORD:-}" ]]; then
  if [[ -f "$FRP_DASHBOARD_PASSWORD_FILE" ]]; then
    FRP_DASHBOARD_PASSWORD="$(tr -d '[:space:]' < "$FRP_DASHBOARD_PASSWORD_FILE")"
  else
    mkdir -p "$(dirname "$FRP_DASHBOARD_PASSWORD_FILE")"
    FRP_DASHBOARD_PASSWORD="$(generate_secret)"
    printf '%s\n' "$FRP_DASHBOARD_PASSWORD" > "$FRP_DASHBOARD_PASSWORD_FILE"
    chmod 600 "$FRP_DASHBOARD_PASSWORD_FILE"
  fi
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

LATEST_VERSION="$(
  curl -fsSL \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    https://api.github.com/repos/fatedier/frp/releases/latest \
    | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' \
    | head -n 1
)"
FRP_VERSION="${FRP_VERSION:-$LATEST_VERSION}"

if [[ -z "$FRP_VERSION" ]]; then
  echo "failed to resolve FRP_VERSION" >&2
  exit 1
fi

read -r -d '' REMOTE_SCRIPT <<'EOF' || true
set -euo pipefail

REMOTE_BASE_DIR="__REMOTE_BASE_DIR__"
REMOTE_CONFIG_DIR="__REMOTE_CONFIG_DIR__"
FRP_BIND_PORT="__FRP_BIND_PORT__"
REMOTE_MODEL_PORT="__REMOTE_MODEL_PORT__"
FRP_DASHBOARD_PORT="__FRP_DASHBOARD_PORT__"
FRP_DASHBOARD_USER="__FRP_DASHBOARD_USER__"
FRP_DASHBOARD_PASSWORD="__FRP_DASHBOARD_PASSWORD__"
FRP_AUTH_TOKEN="__FRP_AUTH_TOKEN__"
FRP_VERSION="__FRP_VERSION__"
REMOTE_SUDO="__REMOTE_SUDO__"

if [[ "$(uname -m)" == "x86_64" || "$(uname -m)" == "amd64" ]]; then
  FRP_ARCH="amd64"
elif [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
  FRP_ARCH="arm64"
else
  echo "unsupported remote architecture: $(uname -m)" >&2
  exit 1
fi

run_root() {
  if [[ -n "$REMOTE_SUDO" ]]; then
    $REMOTE_SUDO "$@"
  else
    "$@"
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ARCHIVE="frp_${FRP_VERSION}_linux_${FRP_ARCH}.tar.gz"
URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/${ARCHIVE}"

curl -L --fail \
  --connect-timeout __CURL_CONNECT_TIMEOUT__ \
  --max-time __CURL_MAX_TIME__ \
  -o "$TMP_DIR/$ARCHIVE" "$URL"
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$TMP_DIR"
EXTRACTED_DIR="$TMP_DIR/frp_${FRP_VERSION}_linux_${FRP_ARCH}"

run_root mkdir -p "$REMOTE_BASE_DIR/bin" "$REMOTE_CONFIG_DIR"
run_root install -m 0755 "$EXTRACTED_DIR/frps" "$REMOTE_BASE_DIR/bin/frps"

cat > "$TMP_DIR/frps.toml" <<CFG
bindPort = ${FRP_BIND_PORT}

auth.method = "token"
auth.token = "${FRP_AUTH_TOKEN}"

transport.tls.force = true

allowPorts = [
  { start = ${REMOTE_MODEL_PORT}, end = ${REMOTE_MODEL_PORT} },
]

webServer.addr = "0.0.0.0"
webServer.port = ${FRP_DASHBOARD_PORT}
webServer.user = "${FRP_DASHBOARD_USER}"
webServer.password = "${FRP_DASHBOARD_PASSWORD}"
CFG

cat > "$TMP_DIR/rgbt-frps.service" <<UNIT
[Unit]
Description=RGBT UAV FRP server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${REMOTE_BASE_DIR}/bin/frps -c ${REMOTE_CONFIG_DIR}/frps.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

run_root install -m 0644 "$TMP_DIR/frps.toml" "$REMOTE_CONFIG_DIR/frps.toml"
run_root install -m 0644 "$TMP_DIR/rgbt-frps.service" /etc/systemd/system/rgbt-frps.service
run_root systemctl daemon-reload
run_root systemctl enable --now rgbt-frps.service
run_root systemctl --no-pager --full status rgbt-frps.service
EOF

REMOTE_SCRIPT="$(
  printf '%s' "$REMOTE_SCRIPT" \
    | sed \
      -e "s|__REMOTE_BASE_DIR__|$REMOTE_BASE_DIR|g" \
      -e "s|__REMOTE_CONFIG_DIR__|$REMOTE_CONFIG_DIR|g" \
      -e "s|__FRP_BIND_PORT__|$FRP_BIND_PORT|g" \
      -e "s|__REMOTE_MODEL_PORT__|$REMOTE_MODEL_PORT|g" \
      -e "s|__FRP_DASHBOARD_PORT__|$FRP_DASHBOARD_PORT|g" \
      -e "s|__FRP_DASHBOARD_USER__|$FRP_DASHBOARD_USER|g" \
      -e "s|__FRP_DASHBOARD_PASSWORD__|$FRP_DASHBOARD_PASSWORD|g" \
      -e "s|__FRP_AUTH_TOKEN__|$FRP_AUTH_TOKEN|g" \
      -e "s|__FRP_VERSION__|$FRP_VERSION|g" \
      -e "s|__REMOTE_SUDO__|$REMOTE_SUDO|g" \
      -e "s|__CURL_CONNECT_TIMEOUT__|$CURL_CONNECT_TIMEOUT|g" \
      -e "s|__CURL_MAX_TIME__|$CURL_MAX_TIME|g"
)"

ssh -o BatchMode=yes -p "$SSH_PORT" "$SSH_TARGET" "bash -s" <<<"$REMOTE_SCRIPT"
