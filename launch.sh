#!/usr/bin/env bash
set -eo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [ -e "$PLATFORM_DIR/.migration-in-progress" ]; then
    echo '工作台路径迁移进行中，请稍后重新启动。' >&2
    exit 2
fi
source "$PLATFORM_DIR/env.sh"
if [ "${1:-}" = "--check-source" ]; then
    python3 -c 'import os; from pathlib import Path; import nero_eval_workbench; print("CWD:", os.getcwd()); print("Platform source:", Path(nero_eval_workbench.__path__[0]).resolve()); print("Runtime:", Path(os.environ["NERO_PLATFORM_ROOT"]) / "runtime/openpi")'
    exit 0
fi
    python3 -c 'import os; import nero_eval_workbench.worker as w; from nero_eval_workbench.common import PACKAGE, ROOT, RUNS, EXPORT, DEFAULT_EXPORTS, PREFERENCES; from nero_eval_workbench.library import root; print("CWD:", os.getcwd()); print("GUI source:", PACKAGE); print("Worker source:", w.__file__); print("Runtime:", ROOT); print("Builtin models:", EXPORT); print("Model/task library:",root()); print("Records:", RUNS); print("Default exports:", DEFAULT_EXPORTS); print("Preferences:", PREFERENCES)'
    exit 0
fi
mkdir -p "$PLATFORM_DIR/logs"
LOG="$PLATFORM_DIR/logs/desktop_$(date +%Y%m%d_%H%M%S)_$$.log"
set +e
python3 -m nero_eval_workbench.desktop "$@" 2>&1 | tee "$LOG"
CODE=$?
if [ "$CODE" -ne 0 ] && [ -n "$DISPLAY" ] && command -v zenity >/dev/null; then
    zenity --error --title='nero真机评测工作台启动失败' --width=660 --text="$(tail -18 "$LOG")" 2>/dev/null || true
fi
exit "$CODE"
