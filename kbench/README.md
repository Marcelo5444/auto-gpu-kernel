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
