"""Agent-side Slurm batch runner: think on the driving machine, run on the cluster.

The optimization agent (kopt loop) edits code and reasons about results on the driving
machine, which has no GPU.  The GPU work happens in an ``sbatch`` job whose batch step
runs inside a pyxis/enroot container on a compute node.  This module owns the driver
side of that boundary:

* the framework checkout (this repository, ``kbench`` + ``kopt``) is never placed on
  the cluster.  The container does not import it -- the generated task harness runs
  standalone contract scripts against a code-under-test checkout passed as ``--repo``.
* the only thing shipped is the minimal runnable payload: the generated ``harness/``
  scripts and the candidate ``repo/`` subtree, into a uniquely named directory under a
  dedicated bundle root (your home, an NFS mount the compute nodes can see) -- NOT the
  shared project scratch, which already holds the long-lived project trees other jobs
  read.  That path is fresh every run and removed when the run finishes, so parallel runs
  never touch each other's code and nothing lingers on a shared tree.
* every scheduler interaction (submit, status, cancel) runs over ``ssh`` to the login
  node from the driving machine, and uses only the interactive scheduler view (squeue
  -i) plus job accounting (sacct).  No GPU step is launched on the login node itself.

This is the batch counterpart to :mod:`kbench.slurm`, which re-enters a node you already
hold.  Batch is what you want when the agent thinks on a laptop and the cluster is
remote: submit, wait for the run, read the artifact back.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def pj(*parts):
    """Join non-empty path segments with a single slash; tolerate a leading root."""
    segments = [str(p).strip().strip("/") for p in parts if str(p).strip().strip("/")]
    if not segments:
        return "/"
    joined = "/".join(segments)
    if str(parts[0] if parts else "").startswith("/"):
        return "/" + joined
    return joined


@dataclass(frozen=True)
class BatchSpec:
    """How to reach the cluster and request one containerised GPU step.

    ``image`` is a pyxis ``.sqsh``/``.squashfs`` path on the shared filesystem or an
    ``nvcr.io#...`` reference that enroot imports on the fly.  ``bundle_root`` is where a
    run's self-contained bundle is written; it must be visible to the compute nodes (a
    home/NFS path the container mounts), it is NOT the shared project scratch root -- the
    per-run bundle never lives alongside the long-lived project trees other jobs read, and
    it is removed when the run finishes.  Only the container's own compile caches (Triton,
    cuTile, inductor) belong in node-local /tmp.
    """

    ssh_host: str
    bundle_root: str
    image: str
    partition: str
    account: str = ""
    gpus: int = 1
    cpus: int = 16
    time_limit: str = "02:00:00"
    exclusive: bool = True
    mount_target: str = "/work"
    python: str = "python3"
    # Exported env is applied inside the container and never overwrites a task value;
    # the forward-compat escape hatch is kept present because the container CUDA often
    # leads the node driver on these H100 nodes.
    export: tuple = ("ALL", "NVIDIA_DISABLE_REQUIRE=1")
    non_ngc: bool = False
    container_extra: tuple = ()
    extra_sbatch: tuple = ()
    keep_bundle: bool = False
    runid_prefix: str = "kopt"
    poll_interval: int = 10

    def non_ngc_export(self):
        # enroot fires its nvidia hook only for NGC images; a plain image needs these or
        # the container sees zero GPUs.  Added as exported env, still inside the container.
        if self.non_ngc:
            return ("NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=all")
        return ()

    def container_mounts(self, bundle_host):
        return "%s:%s" % (bundle_host, self.mount_target)


def _truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def run_id(prefix="kopt"):
    """A fresh per-run id from the wall clock, pid, and the monotonic timer."""
    return "%s-%s-%s-%s" % (prefix, time.strftime("%Y%m%d-%H%M%S"), os.getpid(), int(time.monotonic() * 1000))


def spec(raw, gpus):
    """Build a batch spec from a ``[task.slurm]`` table, or None when batch is off.

    Batch is enabled only when the table says ``mode = "batch"`` (or ``batch = true``).
    Being *inside* an allocation (SLURM_JOB_ID present) must not switch it on, otherwise
    kbench launched from within a compute shell would reroute its own work.
    """
    if not raw:
        return None
    wants = str(raw.get("mode", "")).strip().lower() == "batch" or _truthy(raw.get("batch"))
    if not wants:
        return None
    ssh_host = os.environ.get("KBENCH_SLURM_SSH") or str(raw.get("ssh_host", ""))
    bundle_root = (
        os.environ.get("KBENCH_SLURM_BUNDLE_ROOT")
        or os.environ.get("KBENCH_SLURM_SCRATCH")  # back-compat alias
        or str(raw.get("bundle_root", raw.get("scratch_root", "")))
    )
    image = os.environ.get("KBENCH_SLURM_IMAGE") or str(raw.get("image", ""))
    partition = os.environ.get("KBENCH_SLURM_PARTITION") or str(raw.get("partition", ""))
    missing = [
        name
        for name, value in (
            ("ssh_host", ssh_host),
            ("bundle_root", bundle_root),
            ("image", image),
            ("partition", partition),
        )
        if not value
    ]
    if missing:
        raise SystemExit("slurm batch: missing %s (set [task.slurm] keys or KBENCH_SLURM_* env)" % ", ".join(missing))
    export = tuple(raw.get("export", ("ALL", "NVIDIA_DISABLE_REQUIRE=1")))
    return BatchSpec(
        ssh_host=ssh_host,
        bundle_root=str(bundle_root).rstrip("/"),
        image=image,
        partition=partition,
        account=str(raw.get("account", "")),
        gpus=int(raw.get("gpus", gpus)),
        cpus=int(raw.get("cpus", 16)),
        time_limit=str(raw.get("time_limit", "02:00:00")),
        exclusive=_truthy(raw.get("exclusive", True)),
        mount_target=str(raw.get("mount_target", "/work")),
        python=str(raw.get("python", "python3")),
        export=export,
        non_ngc=_truthy(raw.get("non_ngc", False)),
        container_extra=tuple(str(a) for a in raw.get("container_extra", ())),
        extra_sbatch=tuple(str(a) for a in raw.get("extra_sbatch", ())),
        keep_bundle=_truthy(raw.get("keep_bundle", False)),
        runid_prefix=str(raw.get("runid_prefix", "kopt")),
        poll_interval=int(raw.get("poll_interval", 10)),
    )


_SSH_BASE = (
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=30",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
)


def _remote(spec_, script, timeout=None):
    """Run ``script`` with ``bash -lc`` on the login node; submission + status only."""
    cmd = [*_SSH_BASE, spec_.ssh_host, "bash -lc %s" % shlex.quote(script)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def bundle_paths(spec_, rid):
    """(host path under bundle_root, container path) for one run's bundle."""
    leaf = "%s-%s" % (spec_.runid_prefix, rid)
    host = pj(spec_.bundle_root, leaf)
    return host, spec_.mount_target


class Session:
    """One run's lifecycle against the cluster.

    Opens a unique bundle directory, ships the generated harness once and each distinct
    candidate (keyed by its own fingerprint) as ``repo/``, submits one batch job per
    contract script, waits for completion by polling the interactive scheduler plus job
    accounting, and reads the produced JSON back over ssh.  The framework checkout itself
    is never shipped.
    """

    def __init__(self, spec_, cfg_root, rid=None):
        self.spec = spec_
        self.cfg_root = Path(cfg_root).resolve()
        self.rid = rid or run_id(spec_.runid_prefix)
        self.host_bundle, self.container_bundle = bundle_paths(spec_, self.rid)
        self._candidate_fp = {}
        self._bundle_open = False
        self._log = []
        # run_mode opens one Session per mode so validate + benchmark share a bundle;
        # phase names the candidate lane (main / a / b) and started anchors the wall time.
        self.phase = "main"
        self.started = time.monotonic()

    def log(self, message):
        self._log.append(message)
        print("[slurm-batch] %s" % message)

    # -- local staging ---------------------------------------------------
    def _local_parent(self):
        base = os.environ.get(
            "KBENCH_LOCAL_BUNDLE",
            os.path.join(os.path.expanduser("~"), ".cache", "kbench-slurm-bundle"),
        )
        return Path(base) / ("%s-%s" % (self.spec.runid_prefix, self.rid))

    def _rsync(self, source, dest):
        dest.mkdir(parents=True, exist_ok=True)
        # rsync with --delete so stale files from a previous attempt are removed; we do
        # not rely on a checksum cache, so a re-ship is always fresh.
        cmd = ["rsync", "-a", "--delete", "%s/" % str(source).rstrip("/"), "%s/" % str(dest).rstrip("/")]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _ensure_bundle(self):
        if self._bundle_open:
            return
        out_dir = pj(self.host_bundle, "out")
        result = _remote(self.spec, "mkdir -p %s" % shlex.quote(out_dir))
        if result.returncode != 0:
            raise SystemExit("could not open run bundle on %s: %s" % (self.spec.ssh_host, result.stderr.strip()))
        self._bundle_open = True
        self.log("opened run bundle %s (container %s)" % (self.host_bundle, self.container_bundle))

    # -- shipping --------------------------------------------------------
    def _tar_pipe(self, parent, relpaths):
        # tar | ssh tar: the archive is produced locally and streamed straight into the
        # remote bundle over ssh.  We check the producer (tar) via PIPESTATUS, not just
        # the exit code of the last command in the pipe.
        members = " ".join(shlex.quote(p) for p in relpaths)
        return (
            "set -o pipefail; mkdir -p %s && tar -c --xz -f - -C %s %s | "
            "ssh %s 'tar -x --xz -f - -C %s' && exit ${PIPESTATUS[0]}"
            % (
                shlex.quote(self.host_bundle),
                shlex.quote(str(parent)),
                members,
                shlex.quote(self.spec.ssh_host),
                shlex.quote(self.host_bundle),
            )
        )

    def ship_harness(self, harness_dir):
        self._ensure_bundle()
        harness_dir = Path(harness_dir).resolve()
        staging = self._local_parent()
        self._rsync(harness_dir, staging / "harness")
        # Keep harness/ as a sibling of repo/ in the bundle; the contract scripts locate
        # their fixtures via KBENCH_HARNESS (set to dirname(__file__)) and the candidate
        # via --repo, so this layout matches what kbench already exports locally.
        result = subprocess.run(["bash", "-c", self._tar_pipe(staging, ["harness"])], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit("failed to ship harness: %s" % (result.stderr or result.stdout).strip())
        self.log("shipped harness -> %s/harness" % self.container_bundle)

    def ship_candidate(self, work, phase):
        """Ship ``work`` (the candidate checkout) as ``repo/`` keyed by its fingerprint.

        ``phase`` separates the baseline tree, the A worktree, and the B current tree so
        a paired run does not overwrite one candidate with the other.
        """
        self._ensure_bundle()
        from kbench import task as taskmod

        rev = taskmod.workdir_rev(Path(work))
        fp = "%s@%s" % (phase, rev)
        if self._candidate_fp.get(phase) == fp:
            self.log("reuse candidate phase=%s (fingerprint %s)" % (phase, fp))
            return
        staging = self._local_parent()
        self._rsync(Path(work).resolve(), staging / "repo")
        result = subprocess.run(["bash", "-c", self._tar_pipe(staging, ["repo"])], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit("failed to ship candidate: %s" % (result.stderr or result.stdout).strip())
        self._candidate_fp[phase] = fp
        self.log("shipped candidate phase=%s rev=%s -> %s/repo" % (phase, rev, self.container_bundle))

    # -- submission script ----------------------------------------------
    def submit_script(self, script_name, mode, phase, script_container_rel, repo_container_rel, out_container_rel):
        """Render the batch script that runs one contract script inside the container.

        Paths are container-relative under the mounted bundle.  The deterministic seed is
        the FIRST torch invocation; any environment the optimiser needs is exported inside
        the container and never overwrites a task value.
        """
        s = self.spec
        out_leaf = pj(self.host_bundle, "out", phase, mode)
        # The candidate tree is mounted as <mount>/repo and the container runs with it as
        # the working directory, so Python's cwd entry ('' on the import path) makes the
        # code-under-test importable -- the same thing the local backend gets from running
        # the contract scripts with cwd=work.  No env-var path hacking, and the long-lived
        # project trees on shared scratch are never touched.
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=%s-%s-%s-%s" % (s.runid_prefix, phase, mode, Path(script_name).stem),
            "#SBATCH --partition=%s" % s.partition,
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:%d" % s.gpus,
            "#SBATCH --cpus-per-task=%d" % s.cpus,
            "#SBATCH --time=%s" % s.time_limit,
            "#SBATCH --output=%s" % pj(out_leaf, "%s.log" % Path(script_name).stem),
            "#SBATCH --error=%s" % pj(out_leaf, "%s.err" % Path(script_name).stem),
            "#SBATCH --export=%s" % ",".join((*s.export, *s.non_ngc_export())),
        ]
        if s.account:
            lines.append("#SBATCH --account=%s" % s.account)
        if s.exclusive:
            lines.append("#SBATCH --exclusive")
        lines += list(s.extra_sbatch)
        # pyxis container directives: mount only the run bundle (never the framework,
        # never node /tmp); home is not mounted so host ~/.local cannot shadow container pkgs.
        # The working directory is the candidate so its modules import from cwd, as on local.
        lines += [
            "#SBATCH --container-image=%s" % s.image,
            "#SBATCH --container-mounts=%s" % s.container_mounts(self.host_bundle),
            "#SBATCH --container-workdir=%s" % repo_container_rel,
            "#SBATCH --container-no-mount-home",
        ]
        lines += list(s.container_extra)
        seed_py = (
            "import torch; torch.manual_seed(1337); torch.cuda.manual_seed_all(); "
            "torch.cuda.reset_accum(); "
            "print('KBENCH_SEED_OK 1337 dev', torch.cuda.current_device())"
        )
        seed_cmd = "%s -c %s" % (s.python, shlex.quote(seed_py))
        run_cmd = "%s %s --repo %s --mode %s --output %s" % (
            s.python,
            shlex.quote(script_container_rel),
            shlex.quote(repo_container_rel),
            mode,
            shlex.quote(out_container_rel),
        )
        body = [
            "echo [kbench] batch run %s %s phase %s" % (script_name, mode, phase),
            seed_cmd,
            run_cmd,
        ]
        return "\n".join(lines + ["", *body, ""]) + "\n"

    # -- status (interactive scheduler + accounting only) ----------------
    def job_state(self, job_id):
        result = _remote(self.spec, "squeue -i -o %%T -j %s | tail -1" % shlex.quote(str(job_id)))
        return result.stdout.strip().lower()

    def queue_reason(self, job_id):
        # While the job has not started, always surface the queue reason from the
        # interactive scheduler view (squeue -i), never the daemon.
        result = _remote(self.spec, "squeue -i -o %%R -j %s | tail -1" % shlex.quote(str(job_id)))
        return result.stdout.strip()

    def job_completion(self, job_id):
        # Once the job is on the cluster, read its final state from job accounting.
        result = _remote(self.spec, "sacct -j %s -X -o state,exitcode -p | tail -1" % shlex.quote(str(job_id)))
        return result.stdout.strip()

    # -- run one script -------------------------------------------------
    def run_script(self, script_path, work, harness_dir, mode, phase, out_local, timeout_s):
        """Ship inputs, submit one batch job, poll to completion, retrieve the JSON.

        Returns (exit_code, wall_seconds).  The container-side JSON is read back over ssh
        because the driving machine does not mount the cluster scratch.
        """
        started = time.monotonic()
        self.ship_harness(harness_dir)
        self.ship_candidate(work, phase)

        script_name = Path(script_path).name
        script_container_rel = pj(self.container_bundle, "harness", script_name)
        repo_container_rel = pj(self.container_bundle, "repo")
        out_name = "%s.json" % Path(script_path).stem
        out_container_rel = pj(self.container_bundle, "out", phase, mode, out_name)
        out_host_rel = pj(self.host_bundle, "out", phase, mode, out_name)

        content = self.submit_script(
            script_name, mode, phase, script_container_rel, repo_container_rel, out_container_rel
        )
        submit_dir = pj(self.host_bundle, "driver")
        submit_file = pj(submit_dir, "%s-%s-%s.sbatch" % (phase, mode, Path(script_path).stem))
        write_remote = "mkdir -p %s && cat > %s <<'KBENCH_EOF'\n%sKBENCH_EOF" % (
            shlex.quote(submit_dir), shlex.quote(submit_file), content
        )
        wr = _remote(self.spec, write_remote)
        if wr.returncode != 0:
            self.log("failed to write batch script: %s" % wr.stderr.strip())
            return (99, time.monotonic() - started)

        sr = _remote(self.spec, "sbatch --parsable %s" % shlex.quote(submit_file))
        try:
            job_id = int(sr.stdout.strip().split()[0])
        except (ValueError, IndexError):
            self.log("sbatch parse failed: %r" % sr.stdout)
            return (99, time.monotonic() - started)
        self.log("submitted job %d" % job_id)

        deadline = time.monotonic() + timeout_s
        final = None
        while time.monotonic() < deadline:
            final = self.job_completion(str(job_id))
            if final:
                break
            state = self.job_state(str(job_id))
            if state in ("cancelled", "timeout", "failed", "node_fail", "out_of_memory"):
                final = state
                break
            # Not yet in the accounting records: still pending in the queue; show why.
            self.log("job %d state=%s reason=%s" % (job_id, state or "pending", self.queue_reason(str(job_id))))
            time.sleep(self.spec.poll_interval)
        else:
            self.log("job %d exceeded %ds; cancelling" % (job_id, timeout_s))
            _remote(self.spec, "scancel %s" % shlex.quote(str(job_id)))
            return (124, time.monotonic() - started)

        exit_code = _parse_exit(final)
        read = _remote(self.spec, "cat %s" % shlex.quote(out_host_rel))
        Path(out_local).parent.mkdir(parents=True, exist_ok=True)
        if read.returncode == 0 and read.stdout.strip():
            Path(out_local).write_text(read.stdout)
        else:
            # Missing artifact: record the reason, never fabricate a result.
            self.log("no artifact for %s (final=%s): %s" % (out_name, final, read.stderr.strip()))
        return (exit_code, time.monotonic() - started)

    def close(self):
        # Remove the per-run bundle by default so the home quota is protected and stale
        # run code never lingers.  The scored artifact was already read back into the
        # local out.json before close, and the job's own log lives in the Slurm record; the
        # bundle is only kept on request (keep_bundle=true) for inspection.  The clean-up
        # runs over ssh after the job, never from inside it.
        if self.spec.keep_bundle:
            self.log("kept run bundle %s" % self.host_bundle)
            return
        if self._bundle_open:
            _remote(self.spec, "rm -rf %s" % shlex.quote(self.host_bundle))
            self.log("removed run bundle %s" % self.host_bundle)
            self._bundle_open = False


def _parse_exit(state):
    if not state:
        return 1
    # ``sacct -o state,exitcode -p`` prints e.g. ``COMPLETED|0`` or ``FAILED|1``.
    for token in reversed(state.replace("|", " ").split()):
        head = token.split(":")[0]
        if head.isdigit():
            return int(head)
    if any(tok in state.upper() for tok in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(0)
