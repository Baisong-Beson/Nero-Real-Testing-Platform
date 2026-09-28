# Public source snapshot - 2026-09-28

## Raw evidence

- The public repository contains reusable platform source, interfaces, tests, launch helpers, and documentation.
- The source entry-point check and Python compilation passed after the clean snapshot was prepared.
- Model weights, task datasets, task parameter caches, machine-specific calibration, runtime records, and private test artifacts are excluded.
- The README describes the platform without naming a machine, experiment, private path, or experiment result.
- The repository was recreated and published with one clean source-only commit.

## Interpretation

The repository is a source-only testing platform. Complete deployments supply private model and task resources separately.

- Privacy scan: no tracked workstation path, host name, user directory, model cache, task cache, or runtime artifact path remained after cleanup.

- Clean clone verification passed; the temporary download was deleted after the check.
