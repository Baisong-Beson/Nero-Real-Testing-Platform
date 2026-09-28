#!/usr/bin/env python3
"""Post-process VR experiment episode.json prompts to the dual_v1 drawer task.

Collector sessions such as cube/no_assist stored a layout label in ``prompt``.
The actual teleop task matches session_20260814. This writes that language
string into ``prompt`` and keeps the original under ``prompt_original``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DUAL_V1_PROMPT = (
    "pick up the marker with the right arm, open the drawer with the left arm, "
    "put the marker in the drawer, and close the drawer"
)


def stamp_file(path: Path, *, prompt: str, dry_run: bool) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    current = data.get("prompt")
    if isinstance(current, str) and current.strip() == prompt:
        return "unchanged"
    if "prompt_original" not in data and isinstance(current, str):
        data["prompt_original"] = current
    data["prompt"] = prompt
    data["prompt_source"] = "postprocess_dual_v1"
    if dry_run:
        return "would_stamp"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "stamped"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Session root or parent containing episode.json files",
    )
    parser.add_argument("--prompt", default=DUAL_V1_PROMPT)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also stamp episode.json files under a failed/ directory",
    )
    parser.add_argument(
        "--path-contains",
        default="",
        help="Only stamp episode.json whose path contains this substring (e.g. session_20260831)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    paths = sorted(root.rglob("episode.json"))
    if not args.include_failed:
        paths = [p for p in paths if "failed" not in p.parts]
    if args.path_contains:
        paths = [p for p in paths if args.path_contains in str(p)]
    if not paths:
        raise SystemExit(f"no episode.json under {root}")

    counts = {"stamped": 0, "would_stamp": 0, "unchanged": 0}
    for path in paths:
        status = stamp_file(path, prompt=args.prompt, dry_run=args.dry_run)
        counts[status] += 1
        print(f"{status}\t{path}")
    print(
        f"done root={root} files={len(paths)} "
        + " ".join(f"{k}={v}" for k, v in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
