#!/usr/bin/env bash
set -eo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec bash "$PLATFORM_DIR/launch.sh" "$@"
