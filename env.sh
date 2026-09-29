#!/usr/bin/env bash

# Resolve the platform directory from this file so desktop launchers work
# regardless of their current working directory.
NERO_PLATFORM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NERO_PLATFORM_ROOT

# The desktop entry is launched by a non-login shell. Prefer the local
# act-server environment when it is available, while allowing an override.
NERO_CONDA_ROOT="${NERO_CONDA_ROOT:-$HOME/miniconda3}"
NERO_CONDA_ENV="${NERO_CONDA_ENV:-act-server}"
if [ -f "$NERO_CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$NERO_CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$NERO_CONDA_ENV" >/dev/null 2>&1 || true
fi

export OPENPI_ROOT="$NERO_PLATFORM_ROOT/runtime/openpi"
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi
MSG_SETUP="$NERO_PLATFORM_ROOT/runtime/ros/agx_arm_msgs/share/agx_arm_msgs/local_setup.bash"
if [ -f "$MSG_SETUP" ]; then
    source "$MSG_SETUP"
fi

PYTHONPATH_ENTRIES="$NERO_PLATFORM_ROOT:$NERO_PLATFORM_ROOT/act_eval_workbench:$NERO_PLATFORM_ROOT/runtime/openpi/src:$NERO_PLATFORM_ROOT/runtime/openpi/packages/openpi-client/src"
BRIDGE_ROOT="$NERO_PLATFORM_ROOT/runtime/openpi/examples/nero_pi05_bridge"
if [ -d "$BRIDGE_ROOT" ]; then
    PYTHONPATH_ENTRIES="$PYTHONPATH_ENTRIES:$BRIDGE_ROOT"
fi
export PYTHONPATH="$PYTHONPATH_ENTRIES:${PYTHONPATH:-}"
