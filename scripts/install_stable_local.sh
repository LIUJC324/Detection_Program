#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/.config/rgbt_uav_detection}"
SYSTEMD_DIR="${SYSTEMD_DIR:-$HOME/.config/systemd/user}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$PROJECT_DIR/configs/deploy_stable.yaml}"

FRP_SERVER_ADDR="${FRP_SERVER_ADDR:?set FRP_SERVER_ADDR, for example 8.137.107.232}"
FRP_SERVER_PORT="${FRP_SERVER_PORT:-7000}"
FRP_AUTH_TOKEN_FILE="${FRP_AUTH_TOKEN_FILE:-$CONFIG_DIR/frp_auth_token}"
LOCAL_MODEL_HOST="${LOCAL_MODEL_HOST:-127.0.0.1}"
LOCAL_MODEL_PORT="${LOCAL_MODEL_PORT:-18000}"
REMOTE_MODEL_PORT="${REMOTE_MODEL_PORT:-18000}"
ENABLE_NOW="${ENABLE_NOW:-0}"
DOWNLOAD_FRP="${DOWNLOAD_FRP:-1}"

generate_secret() {
  od -An -N 24 -tx1 /dev/urandom | tr -d ' \n'
}

mkdir -p "$CONFIG_DIR" "$SYSTEMD_DIR" "$BIN_DIR"

if [[ -z "${FRP_AUTH_TOKEN:-}" ]]; then
  if [[ -f "$FRP_AUTH_TOKEN_FILE" ]]; then
    FRP_AUTH_TOKEN="$(tr -d '[:space:]' < "$FRP_AUTH_TOKEN_FILE")"
  else
    FRP_AUTH_TOKEN="$(generate_secret)"
    printf '%s\n' "$FRP_AUTH_TOKEN" > "$FRP_AUTH_TOKEN_FILE"
    chmod 600 "$FRP_AUTH_TOKEN_FILE"
  fi
fi

render_template() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  sed \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__DEPLOY_CONFIG__|$DEPLOY_CONFIG|g" \
    -e "s|__FRP_SERVER_ADDR__|$FRP_SERVER_ADDR|g" \
    -e "s|__FRP_SERVER_PORT__|$FRP_SERVER_PORT|g" \
    -e "s|__FRP_AUTH_TOKEN__|$FRP_AUTH_TOKEN|g" \
    -e "s|__LOCAL_MODEL_HOST__|$LOCAL_MODEL_HOST|g" \
    -e "s|__LOCAL_MODEL_PORT__|$LOCAL_MODEL_PORT|g" \
    -e "s|__REMOTE_MODEL_PORT__|$REMOTE_MODEL_PORT|g" \
    -e "s|__FRPC_BIN__|$BIN_DIR/frpc|g" \
    -e "s|__FRPC_CONFIG__|$CONFIG_DIR/frpc.toml|g" \
    "$src" > "$dst"
}

if [[ "$DOWNLOAD_FRP" == "1" ]]; then
  "$PROJECT_DIR/scripts/download_frp.sh" client
fi

render_template \
  "$PROJECT_DIR/deploy/stable/frpc.toml.template" \
  "$CONFIG_DIR/frpc.toml"

render_template \
  "$PROJECT_DIR/deploy/stable/systemd/rgbt-model.user.service.template" \
  "$SYSTEMD_DIR/rgbt-model.service"

render_template \
  "$PROJECT_DIR/deploy/stable/systemd/rgbt-frpc.user.service.template" \
  "$SYSTEMD_DIR/rgbt-frpc.service"

echo "rendered:"
echo "  $CONFIG_DIR/frpc.toml"
echo "  $FRP_AUTH_TOKEN_FILE"
echo "  $SYSTEMD_DIR/rgbt-model.service"
echo "  $SYSTEMD_DIR/rgbt-frpc.service"

if [[ "$ENABLE_NOW" == "1" ]]; then
  systemctl --user daemon-reload
  systemctl --user enable --now rgbt-model.service
  systemctl --user enable --now rgbt-frpc.service
fi

echo
echo "next:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now rgbt-model.service"
echo "  systemctl --user enable --now rgbt-frpc.service"
echo
echo "if you want user services to survive logout:"
echo "  loginctl enable-linger $USER"
