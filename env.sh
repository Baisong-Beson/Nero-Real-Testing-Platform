#!/usr/bin/env bash
NERO_PLATFORM_ROOT="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export NERO_PLATFORM_ROOT
export OPENPI_ROOT="$NERO_PLATFORM_ROOT/runtime/openpi"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "$CONDA_BASE" ] || source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
MSG_SETUP="$NERO_PLATFORM_ROOT/runtime/ros/agx_arm_msgs/share/agx_arm_msgs/local_setup.bash"
[ -f "$MSG_SETUP" ] && source "$MSG_SETUP"
export PYTHONPATH="$NERO_PLATFORM_ROOT:$NERO_PLATFORM_ROOT/act_eval_workbench:$NERO_PLATFORM_ROOT/runtime/openpi/examples/nero_pi05_bridge:$NERO_PLATFORM_ROOT/runtime/openpi/src:$NERO_PLATFORM_ROOT/runtime/openpi/packages/openpi-client/src:$PYTHONPATH"
