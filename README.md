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

## Camera views

The live monitor supports main, left wrist, and right wrist video previews. Use the view selector to change the displayed camera and **Camera settings** to scan devices and assign roles. Only fresh frames are displayed.

Device bindings are saved locally. Automatic replacement updates a serial number only when a missing role has exactly one unassigned replacement of the same device family. Ambiguous matches require manual selection. Camera changes are blocked while a task is running.

Preview roles do not rewrite a model's observation schema; configure model inputs separately for each deployment. Camera and CAN identifiers, driver installation paths, and logs are excluded from version control.

## Safety

Review the safety checks before connecting hardware. Start with validation or simulation modes and keep physical execution disabled until external configuration has been reviewed.
