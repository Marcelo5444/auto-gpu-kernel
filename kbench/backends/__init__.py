"""Backends all expose: run(cfg, payload) -> {label: list[WorkloadResult]}."""

from __future__ import annotations

BACKENDS = ("local", "modal", "fal", "slurm")


def get(name: str):
    if name == "local":
        from . import local

        return local
    if name == "modal":
        from . import modal as modal_backend

        return modal_backend
    if name == "fal":
        from . import fal as fal_backend

        return fal_backend
    if name == "slurm":
        from . import slurm as slurm_backend

        return slurm_backend
    raise SystemExit(f"unknown backend {name!r} (expected one of {', '.join(BACKENDS)})")
