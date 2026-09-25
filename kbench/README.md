# kbench

Kbench separates benchmark adapters (what to run) from execution backends (where to
run it). FlashInfer is the built-in adapter and arbitrary repositories use an adapter
backed by the agent-generated `validate.py` and `benchmark.py` scripts.

```mermaid
classDiagram
    direction LR

    class BenchmarkAdapter {
        <<abstract>>
        +run(candidates, request)
        +bench(request)
        +ab(baseline, request)
    }
    class FlashInferAdapter
    class GeneratedTaskAdapter
    class Job
    class ValidateScript {
        <<agent-generated>>
    }
    class BenchmarkScript {
        <<agent-generated>>
    }

    FlashInferAdapter --|> BenchmarkAdapter
    GeneratedTaskAdapter --|> BenchmarkAdapter
    FlashInferAdapter --> Job : sends through local / Modal / FAL / SLURM
    GeneratedTaskAdapter --> ValidateScript
    GeneratedTaskAdapter --> BenchmarkScript
```

## The SLURM runner

`kbench.slurm` re-enters an allocation you already hold with `srun --overlap`, and is
used by both adapters: the `slurm` backend runs a `Job` there, and a `[task.slurm]` table
makes the generated `validate.py` / `benchmark.py` run there too. It never allocates:
hold the node yourself and set `KBENCH_SLURM_JOBID`, so one warm container serves many
runs.

The driving machine is normally a login node with no GPU and a home directory the compute
nodes cannot see, so the kbench package and the job payload are staged onto a shared
filesystem rather than shared by import, and every path crossing into the container is
translated through the configured mount map. A project that is not under a mount fails on
the driving machine, naming the path, instead of as a `FileNotFoundError` inside the job.

Measurements record `slurm / <gpu> xN` as their provenance, so they are never compared
against numbers taken locally. See [configs/slurm.toml](../configs/slurm.toml).

## The SLURM batch runner (think on the agent, run on the cluster)

`kbench.slurm_batch` is the batch counterpart to the `srun --overlap` runner above. It is
meant for the case where the optimizing agent runs on a machine without a GPU (a laptop, a
login node) and the cluster is remote: the agent edits the candidate and reasons about the
result locally, while every GPU step is an `sbatch` job whose batch step enters a
pyxis/enroot container on a compute node, and the agent just submits, waits, and reads the
artifact back over `ssh`.

Enable it for a generated task by adding a `[task.slurm] mode = "batch" ...` table (see
[configs/adasplash_get_output_h100_slurm.toml](../configs/adasplash_get_output_h100_slurm.toml)).
The same `validate.py` / `benchmark.py` contract runs unchanged; only the transport differs.
The two runners are mutually exclusive: a `[task.slurm]` table selects one mode.

The batch runner keeps the framework off the cluster.  `kbench` and `kopt` are never
copied to the cluster -- the container does not import them.  The only things shipped
per run are the generated `harness/` scripts and the candidate (code-under-test) tree, into
a directory named by a freshly generated run id under a dedicated `bundle_root` (your home,
an NFS mount the compute nodes can see) -- never the shared project scratch, which already
holds the long-lived project trees other jobs read.  Because the path is
new every run and removed when the run finishes, parallel runs never touch each other's code
and a run never edits or steps on a long-lived shared tree.  Status is read from the
interactive scheduler (`squeue -i`) and job accounting (`sacct`); no GPU command is ever run
on the login node, which is used only for submission and status.  A deterministic seed is the
first torch call, the forward-compat `NVIDIA_DISABLE_REQUIRE=1` is exported by default, and
the candidate is mounted as the only container path with `--container-no-mount-home` so a
host `~/.local` cannot shadow container packages.

Provenance is recorded as `slurm-batch / <gpu> xN`, so batch numbers never compare against
local or overlap-mode measurements.  Each run's bundle (script, `*.log`, the result
JSON) is removed by default after the result is read back; set `keep_bundle = true` to keep
it under `bundle_root` for inspection.  See `kbench/slurm_batch.py`.
