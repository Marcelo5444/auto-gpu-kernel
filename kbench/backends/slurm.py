"""SLURM backend. The job runs in a pyxis container on a node you already hold.

Unlike Modal and fal there is no image to build: the .sqsh is whatever the cluster has,
so the ImageSpec's env is applied by hand at run time and its apt/pip/run steps are the
image author's problem, exactly as on the local backend.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from kbench import slurm
from kbench.job import Job
from kbench.results import WorkloadResult


def run(cfg, job: Job) -> dict[str, list[WorkloadResult]]:
    spec = slurm.spec(cfg.slurm, cfg.gpu_count)
    if spec is None:
        raise SystemExit("backend 'slurm' needs a [remote.slurm] table in config.toml")

    stage, pythonpath = slurm.stage_package(spec)
    payload = stage / "job"
    payload.mkdir(parents=True, exist_ok=True)
    # Image env is baked into the image on remote backends; apply it by hand here.
    job = replace(job, data_path=cfg.data_path, env={**cfg.image.env, **job.env})
    (payload / "job.json").write_text(json.dumps(asdict(job), indent=2))

    work = spec.container_path(payload)
    cmd = replace(spec, export=(*spec.export, f"PYTHONPATH={pythonpath}")).command(
        [spec.python, "-m", "kbench.slurm", f"{work}/job.json", f"{work}/out.json"],
        workdir=work,
    )
    (payload / "out.json").unlink(missing_ok=True)
    print(f"[kbench] srun into job {spec.jobid}")
    subprocess.run(cmd, check=False, timeout=cfg.timeout_s)

    out = payload / "out.json"
    if not out.exists():
        raise SystemExit(
            f"slurm step produced no results ({out}); the srun output above has the reason"
        )
    return {
        label: [WorkloadResult(**r) for r in results]
        for label, results in json.loads(out.read_text()).items()
    }
