# NERO Real Testing Platform

A source-only testing platform for NERO robot software. It provides reusable platform code, ROS 2 integration, model and task interfaces, validation utilities, safety checks, and developer entry points.

## Scope

The repository contains platform source code, interfaces, tests, launch helpers, and documentation needed to build a testing workflow. Model weights, task datasets, task parameter caches, machine-specific calibration, runtime logs, generated records, private test artifacts, and workstation-specific paths are excluded.

Complete deployments provide external robot, model, and task resources separately.

## Quick start

    source ./env.sh
    bash ./launch.sh --check-source

Run source checks with:

    source ./env.sh
    python3 -m compileall -q act_eval_workbench nero_eval_workbench driver_patches
    python3 -m unittest discover -s act_eval_workbench -t . -p 'test*.py' -v

Data-dependent checks require external resources that are not part of this repository.

## Safety

Review the safety checks before connecting hardware. Start with validation or simulation modes and keep physical execution disabled until external configuration has been reviewed.
