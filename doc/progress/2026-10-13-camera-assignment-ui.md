# Camera assignment and preview UI

## Evidence

The non-GUI suite passed 133 tests; one display-dependent test was then run separately and passed on X11. Independent synthetic wrist and main-camera streams passed. Live preview reassignment, source swapping, and source removal passed on isolated localhost ROS domain 174. The real Tk widgets were exercised at the minimum supported window size; preview switching and stale-frame clearing passed. Raw logs and screenshots are retained only in ignored local logs.

## Interpretation

The preview can select main, left, or right views. Device assignment is persisted separately from public source. Automatic replacement requires a unique unassigned device and a unique missing role of the same kind. Ambiguous cases remain pending for user selection. No robot motion was requested.

## Limits

Physical cameras were disconnected before final integration verification. Earlier live right-camera reception was measured after correcting its device binding. Reconnection with the final UI should be checked when hardware is available.

## Publication

Local checkpoint completed. No Open-Science publisher is configured.
