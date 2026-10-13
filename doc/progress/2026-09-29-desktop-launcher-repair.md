# Desktop launcher repair

## Raw evidence

- `bash ./launch.sh --check-source` passed after the launcher and environment changes.
- The launcher selected the configured `act-server` Python environment and found its installed runtime dependencies.
- With the active X11 display configured, `bash ./launch.sh --self-test` reached the Tk interface. The run then stopped at the existing default-selection assertion in the UI self-test.

## Interpretation

The desktop shortcut failure was caused by the launcher exiting through stale debug code before the GUI command and by the desktop shell not loading the runtime environment. The shortcut now reaches the GUI startup path; the remaining self-test assertion is unrelated to shortcut startup.
