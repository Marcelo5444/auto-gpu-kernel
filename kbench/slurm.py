"""Run on an already-allocated SLURM job, via `srun --overlap` into a pyxis container.

The driving machine is normally a login node: no GPU, no benchmark stack, and a home
directory the compute nodes cannot see. So nothing is shared by import — the kbench
package and the job payload are staged onto a filesystem both sides mount, and the
container re-enters them as `python -m kbench.slurm <job.json> <out.json>`.

The allocation is not created here. Hold nodes yourself (`sbatch --wrap="sleep ..."`)
and point `jobid` at the result; the container and page cache stay warm across runs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Enroot only runs its nvidia hook for NGC-derived images, so on any other image the
# container sees zero GPUs unless these are exported into it.
DEFAULT_EXPORT = ("NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=all")


@dataclass(frozen=True)
class SrunSpec:
    """How to re-enter one allocation. Mounts are `host:container[:flags]`, as pyxis takes them."""

    jobid: str
    image: str
    mounts: tuple[str, ...] = ()
    stage: str = ""
    python: str = "python3"
    gpus: int = 1
    cpus: int = 16
    export: tuple[str, ...] = DEFAULT_EXPORT
    args: tuple[str, ...] = ()

    def container_path(self, host_path: str | Path) -> str:
        """Translate a path on the driving machine to its path inside the container.

        Every path crossing the boundary goes through here, so a config that forgot a
        mount fails with the offending path instead of a FileNotFoundError in the job.
        """
        path = Path(host_path).resolve()
        for mount in self.mounts:
            source, _, rest = mount.partition(":")
            target = rest.split(":")[0]
            if not target:
                continue
            source = Path(source).resolve()
            if path == source:
                return target
            if path.is_relative_to(source):
                return f"{target}/{path.relative_to(source)}"
        raise SystemExit(
            f"{path} is not under any container mount ({', '.join(self.mounts) or 'none'});"
            " add it to slurm.mounts, or move the project onto a mounted filesystem"
        )

    def command(self, argv: list[str], workdir: str) -> list[str]:
        """Wrap a container-side argv in the srun invocation that runs it."""
        cmd = [
            "srun",
            f"--jobid={self.jobid}",
            "--overlap",
            "--ntasks=1",
            f"--gres=gpu:{self.gpus}",
            f"--cpus-per-task={self.cpus}",  # without this pyxis gets 1 CPU and dies
            # The driving cwd is usually invisible to the node; chdir somewhere that exists
            # so the step does not warn and silently land in /tmp.
            "--chdir=/tmp",
            "--export=" + ",".join(("ALL", *self.export)),
            f"--container-image={self.image}",
            "--container-writable",
            "--no-container-mount-home",
        ]
        if self.mounts:
            cmd.append("--container-mounts=" + ",".join(self.mounts))
        cmd.append(f"--container-workdir={workdir}")
        return cmd + list(self.args) + argv


def spec(raw: dict, gpus: int) -> SrunSpec | None:
    """Build a spec from a [*.slurm] table, or None when the runner is not configured.

    The job id changes with every allocation, so KBENCH_SLURM_JOBID overrides the file —
    holding a fresh node should not mean editing config.toml.
    """
    # Opting in is a [*.slurm] table or KBENCH_SLURM_JOBID. SLURM_JOB_ID alone must not
    # count: kbench run from inside any allocation would otherwise reroute itself.
    if not raw and not os.environ.get("KBENCH_SLURM_JOBID"):
        return None
    jobid = (
        os.environ.get("KBENCH_SLURM_JOBID")
        or str(raw.get("jobid", ""))
        or os.environ.get("SLURM_JOB_ID", "")
    )
    image = os.environ.get("KBENCH_SLURM_IMAGE") or raw.get("image", "")
    missing = [name for name, value in (("jobid", jobid), ("image", image)) if not value]
    if missing:
        raise SystemExit(
            f"slurm runner: missing {', '.join(missing)}"
            " (set it in the [*.slurm] config table, or KBENCH_SLURM_JOBID / KBENCH_SLURM_IMAGE)"
        )
    return SrunSpec(
        jobid=str(jobid),
        image=str(image),
        mounts=tuple(raw.get("mounts", ())),
        stage=os.environ.get("KBENCH_SLURM_STAGE") or raw.get("stage", ""),
        python=raw.get("python", "python3"),
        gpus=int(raw.get("gpus", gpus)),
        cpus=int(raw.get("cpus", 16)),
        export=tuple(raw.get("export", DEFAULT_EXPORT)),
        args=tuple(raw.get("args", ())),
    )


def stage_package(spec: SrunSpec) -> tuple[Path, str]:
    """Copy kbench to the shared staging dir. Returns (host dir, container PYTHONPATH)."""
    if not spec.stage:
        raise SystemExit("slurm runner: set stage to a directory both sides mount")
    stage = Path(spec.stage).resolve()
    package = stage / "kbench"
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(
        Path(__file__).resolve().parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return stage, spec.container_path(stage)


def main(argv: list[str]) -> int:
    """Container side of the FlashInfer backend."""
    from kbench.flashinfer import evaluate
    from kbench.job import Job

    job_path, out_path = argv
    results = evaluate(Job(**json.loads(Path(job_path).read_text())))
    Path(out_path).write_text(
        json.dumps({k: [asdict(r) for r in v] for k, v in results.items()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
