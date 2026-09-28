#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$HOME/.local/share/applications"
APP="$HOME/.local/share/applications/nero-eval-workbench.desktop"
cat > "$APP" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=nero真机评测工作台
Comment=VLA 模型与任务导入、起手位、模拟、真机评测与记录
Exec=/bin/bash $PLATFORM_DIR/launch.sh
Path=$PLATFORM_DIR
Icon=$PLATFORM_DIR/nero_eval_workbench/icon.svg
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF
chmod +x "$APP"
DESKTOP_DIR="$(xdg-user-dir DESKTOP)"
mkdir -p "$DESKTOP_DIR"
cp "$APP" "$DESKTOP_DIR/NERO-Evaluation.desktop"
chmod +x "$DESKTOP_DIR/NERO-Evaluation.desktop"
gio set "$DESKTOP_DIR/NERO-Evaluation.desktop" metadata::trusted true 2>/dev/null || true
printf 'Desktop entry: %s\n' "$APP"
