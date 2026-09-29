# Progress index

## Current status

- 2026-09-28: The source-only testing-platform snapshot was prepared.
- Python compilation and package checks passed for the public source tree.
- The source entry-point check passed after publication.
- The public snapshot excludes model weights, task datasets, task parameter caches, machine-specific calibration, runtime records, and private test artifacts.
- The public README describes the platform scope, setup entry points, safety boundary, and external-resource requirement.
- The public repository history contains one clean source-only commit.

Milestones:

- doc/progress/2026-09-28-public-source-snapshot.md — source-only boundary, source check, and publication checkpoint.

- 2026-09-28: Privacy scan completed with no tracked workstation paths; machine-specific documentation, connection settings, generated installs, and private data tools were removed.

- 2026-09-28: A fresh public clone passed the source entry-point check and Python compilation; the temporary clone was removed after verification.

- 2026-09-29: The desktop launcher was repaired by removing an unreachable debug block, resolving the platform root from the launcher location, and activating the available runtime environment for non-login desktop launches.
- 2026-09-29: The source check passed and the desktop self-test reached the Tk interface on the active display; the self-test then stopped at an existing default-selection assertion.

- 2026-09-29: Local task tooling was excluded from the public source snapshot. The PI0.5 bridge example package and local task/hardware verification utilities were removed from the Git index while remaining available in the local workspace.
- 2026-09-29: The local source check and Python compilation passed after the exclusion; the local bridge directory was retained for the desktop installation, and the public tracked file list contains no PI0.5 bridge files.
