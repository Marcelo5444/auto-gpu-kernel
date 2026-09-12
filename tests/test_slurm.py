from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from kbench import backends, slurm
from kbench.config import load
from kbench.job import Job
from kbench.slurm import SrunSpec
from kbench.task import _script_cmd, provenance

MOUNTS = ("/mnt/lustre01/users/me:/lustre_home", "/mnt/numa1/users/me:/scratch:rw")

TASK_TOML = """\
[task]
name = "demo"
objective = "go faster"
measure = "run bench.py"
validate = "run pytest"

[task.repo]
path = "../repo"

[task.hardware]
gpus = 4
gpu = "GB200"
"""


def spec(**overrides) -> SrunSpec:
    return SrunSpec(
        jobid="14255",
        image="/mnt/lustre01/users/me/images/runner.sqsh",
        mounts=MOUNTS,
        stage="/mnt/lustre01/users/me/kbench-stage",
        python="/lustre_home/envs/runner/bin/python3",
        gpus=4,
        **overrides,
    )


class SpecTest(unittest.TestCase):
    def test_runner_is_off_until_it_is_configured(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(slurm.spec({}, gpus=1))

    def test_being_inside_an_allocation_does_not_enable_the_runner(self) -> None:
        # kbench run from inside any srun step would otherwise reroute itself.
        with patch.dict("os.environ", {"SLURM_JOB_ID": "999"}, clear=True):
            self.assertIsNone(slurm.spec({}, gpus=1))

    def test_env_overrides_the_configured_job_id(self) -> None:
        raw = {"jobid": "1", "image": "img.sqsh"}
        with patch.dict("os.environ", {"KBENCH_SLURM_JOBID": "14255"}, clear=True):
            self.assertEqual(slurm.spec(raw, gpus=1).jobid, "14255")

    def test_a_table_without_an_image_is_rejected(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as caught:
                slurm.spec({"jobid": "14255"}, gpus=1)
        self.assertIn("image", str(caught.exception))

    def test_gpu_count_falls_back_to_the_remote_setting(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            built = slurm.spec({"jobid": "1", "image": "i.sqsh"}, gpus=4)
        self.assertEqual(built.gpus, 4)


class ContainerPathTest(unittest.TestCase):
    def test_paths_are_translated_through_the_mount_map(self) -> None:
        self.assertEqual(
            spec().container_path("/mnt/lustre01/users/me/proj/harness/bench.py"),
            "/lustre_home/proj/harness/bench.py",
        )

    def test_a_mount_root_maps_to_the_target_itself(self) -> None:
        self.assertEqual(spec().container_path("/mnt/numa1/users/me"), "/scratch")

    def test_mount_flags_are_not_mistaken_for_the_target(self) -> None:
        self.assertEqual(spec().container_path("/mnt/numa1/users/me/cache"), "/scratch/cache")

    def test_an_unmounted_path_names_itself_in_the_error(self) -> None:
        # The common failure is a project on node-local /home; say so on the driving
        # machine rather than as a FileNotFoundError inside the job.
        with self.assertRaises(SystemExit) as caught:
            spec().container_path("/home/me/auto-gpu-kernel")
        self.assertIn("/home/me/auto-gpu-kernel", str(caught.exception))


class CommandTest(unittest.TestCase):
    def test_command_carries_the_flags_the_cluster_requires(self) -> None:
        cmd = spec().command(["python3", "-c", "pass"], workdir="/lustre_home/proj")
        self.assertEqual(cmd[0], "srun")
        self.assertIn("--jobid=14255", cmd)
        self.assertIn("--overlap", cmd)
        self.assertIn("--gres=gpu:4", cmd)
        self.assertIn("--cpus-per-task=16", cmd)
        self.assertIn("--container-workdir=/lustre_home/proj", cmd)
        self.assertIn("--container-mounts=" + ",".join(MOUNTS), cmd)
        self.assertEqual(cmd[-3:], ["python3", "-c", "pass"])

    def test_the_nvidia_variables_are_exported_into_the_container(self) -> None:
        # Without them a non-NGC image sees zero GPUs.
        export = next(a for a in spec().command([], workdir="/w") if a.startswith("--export="))
        self.assertIn("ALL", export)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", export)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=all", export)

    def test_extra_args_land_before_the_payload(self) -> None:
        cmd = spec(args=("--container-env=FOO",)).command(["true"], workdir="/w")
        self.assertEqual(cmd[-2:], ["--container-env=FOO", "true"])


class TaskRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        self.root.mkdir()
        (self.root / "config.toml").write_text(TASK_TOML)
        self.addCleanup(self.tmp.cleanup)

    def _cmd(self, cfg) -> list[str]:
        return _script_cmd(
            cfg,
            self.root / "harness/bench.py",
            self.root,
            "quick",
            self.root / "out.json",
        )

    def test_scripts_run_locally_when_no_runner_is_configured(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = replace(load(self.root), slurm={})
            self.assertNotEqual(self._cmd(cfg)[0], "srun")
            self.assertEqual(provenance(cfg), "local / GB200 x4")

    def test_scripts_are_re_entered_into_the_container_with_mapped_paths(self) -> None:
        table = {
            "jobid": "14255",
            "image": "runner.sqsh",
            "mounts": [f"{Path(self.tmp.name).resolve()}:/lustre_home"],
            "python": "/lustre_home/envs/runner/bin/python3",
        }
        with patch.dict("os.environ", {}, clear=True):
            cfg = replace(load(self.root), slurm=table)
            cmd = self._cmd(cfg)
            self.assertEqual(provenance(cfg), "slurm / GB200 x4")
        self.assertEqual(cmd[0], "srun")
        self.assertIn("/lustre_home/envs/runner/bin/python3", cmd)
        self.assertIn("/lustre_home/proj/harness/bench.py", cmd)
        self.assertIn("--repo", cmd)
        self.assertIn("/lustre_home/proj", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("/lustre_home/proj/out.json", cmd)
        # The driving machine's own paths must not leak into the container argv.
        self.assertNotIn(str(self.root / "harness/bench.py"), cmd)


class BackendTest(unittest.TestCase):
    def test_slurm_is_a_registered_backend(self) -> None:
        self.assertIn("slurm", backends.BACKENDS)
        self.assertTrue(hasattr(backends.get("slurm"), "run"))

    def test_the_job_survives_the_json_round_trip(self) -> None:
        # The container rebuilds the job from this file; keep it primitive-only.
        job = Job(
            definition="demo",
            language="triton",
            entry_point="solution.py::kernel",
            sources={"main": {"solution.py": "x = 1"}},
            data_path="/data",
            env={"A": "B"},
        )
        self.assertEqual(Job(**json.loads(json.dumps(asdict(job)))), job)


if __name__ == "__main__":
    unittest.main()
