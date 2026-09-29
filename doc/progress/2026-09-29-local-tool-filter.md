# Local task tooling exclusion

## Raw evidence

- The public Git index no longer contains `runtime/openpi/examples/nero_pi05_bridge` or the local PI0.5, task, and hardware verification utilities.
- The local bridge directory remains present on the workstation and is now added to `PYTHONPATH` only when it exists.
- `bash ./launch.sh --check-source` and `python3 -m compileall -q act_eval_workbench nero_eval_workbench driver_patches runtime/openpi/src` passed after the change.

## Interpretation

The public repository now contains platform source and reusable checks without local task execution tools or PI0.5 bridge code. Workstation-only tooling remains outside the published Git index.
