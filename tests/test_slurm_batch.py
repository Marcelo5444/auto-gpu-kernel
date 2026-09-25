"""Offline tests for the agent-side Slurm batch runner.

Nothing here reaches the cluster: the ssh transport and the tar|ssh pipe are replaced
with local stand-ins so the real submit/ship/poll/retrieve logic runs against a local
directory standing in for the shared scratch root.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kbench import slurm_batch
from kbench.slurm_batch import BatchSpec, Session


def make_spec(**overrides):
    base = dict(
        ssh_host="marcelos@dlcluster",
        scratch_root="/home/scratch.marcelos_wwfo",
        image="cuda133-pytorch-amd64.sqsh",
        partition="dgxh100",
        gpus=1,
    )
    base.update(overrides)
    return BatchSpec(**base)


class GateTest(unittest.TestCase):
    def test_absent_table_disables(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(slurm_batch.spec({}, 1))

    def test_no_batch_marker_disables_even_inside_an_allocation(self):
        # kbench launched from within a compute shell must not reroute its own work.
        with patch.dict("os.environ", {"SLURM_JOB_ID": "777"}, clear=True):
            spec_in = {"ssh_host": "h", "scratch_root": "/s", "image": "i", "partition": "p"}
            self.assertIsNone(slurm_batch.spec(spec_in, 1))

    def test_mode_batch_enables(self):
        raw = {"mode": "batch", "ssh_host": "h", "scratch_root": "/s", "image": "i", "partition": "p"}
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNotNone(slurm_batch.spec(raw, 1))

    def test_batch_true_enables(self):
        raw = {"batch": True, "ssh_host": "h", "scratch_root": "/s", "image": "i", "partition": "p"}
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNotNone(slurm_batch.spec(raw, 1))

    def test_missing_field_names_itself(self):
        raw = {"mode": "batch", "ssh_host": "h", "scratch_root": "/s", "image": "i"}
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as caught:
                slurm_batch.spec(raw, 1)
        self.assertIn("partition", str(caught.exception))

    def test_env_overrides_file(self):
        raw = {"mode": "batch", "ssh_host": "h", "scratch_root": "/s", "image": "i", "partition": "p"}
        with patch.dict("os.environ", {"KBENCH_SLURM_PARTITION": "preprod"}, clear=True):
            self.assertEqual(slurm_batch.spec(raw, 1).partition, "preprod")


class BundleUniquenessTest(unittest.TestCase):
    def test_two_runs_never_share_a_bundle(self):
        spec_ = make_spec()
        one = Session(spec_, os.getcwd(), rid="RUN-A")
        two = Session(spec_, os.getcwd(), rid="RUN-B")
        self.assertNotEqual(one.host_bundle, two.host_bundle)
        self.assertTrue(one.host_bundle.startswith("/home/scratch.marcelos_wwfo/kopt-RUN-A"))
        self.assertTrue(one.host_bundle.endswith(one.rid))
        self.assertNotIn("auto-gpu-kernel", one.host_bundle)

    def test_container_mount_is_the_bundle(self):
        spec_ = make_spec(mount_target="/work")
        sess = Session(spec_, os.getcwd(), rid="R1")
        self.assertEqual(spec_.container_mounts(sess.host_bundle), sess.host_bundle + ":/work")


class SubmitScriptTest(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(account="wwfo-emea_b200", exclusive=True)
        self.sess = Session(self.spec, os.getcwd(), rid="SS-1")

    def script(self):
        return self.sess.submit_script(
            "benchmark.py", "quick", "b",
            "/work/harness/benchmark.py", "/work/repo", "/work/out/b/quick/benchmark.json",
        )

    def test_carries_the_partition_and_gpu(self):
        s = self.script()
        self.assertIn("#SBATCH --partition=dgxh100", s)
        self.assertIn("#SBATCH --gres=gpu:1", s)
        self.assertIn("#SBATCH --account=wwfo-emea_b200", s)
        self.assertIn("#SBATCH --exclusive", s)

    def test_container_mounts_only_the_bundle(self):
        s = self.script()
        self.assertIn("#SBATCH --container-mounts=%s:/work" % self.sess.host_bundle, s)
        self.assertIn("#SBATCH --container-no-mount-home", s)
        # The framework checkout must never appear in the submission.
        self.assertNotIn("auto-gpu-kernel", s)

    def test_forward_compat_export_present(self):
        s = self.script()
        export = next(line for line in s.splitlines() if "--export=" in line)
        self.assertIn("NVIDIA_DISABLE_REQUIRE=1", export)

    def test_seed_is_first_torch_call(self):
        body = self.script().splitlines()
        seed_index = next(i for i, line in enumerate(body) if "manual_seed(1337)" in line)
        run_index = next(i for i, line in enumerate(body) if "--repo" in line)
        self.assertLess(seed_index, run_index)

    def test_output_lands_under_the_bundle_not_tmp(self):
        s = self.script()
        out = next(line for line in s.splitlines() if "--output=" in line)
        self.assertIn(self.sess.host_bundle, out)
        self.assertNotIn("/tmp/", out)


class ExitParseTest(unittest.TestCase):
    def test_completed(self):
        self.assertEqual(slurm_batch._parse_exit("COMPLETED|0:0"), 0)

    def test_failed_code(self):
        self.assertEqual(slurm_batch._parse_exit("FAILED|1:0"), 1)

    def test_running_is_zero(self):
        self.assertEqual(slurm_batch._parse_exit("RUNNING"), 0)

    def test_none_is_failure(self):
        self.assertEqual(slurm_batch._parse_exit(None), 1)


class RunScriptOfflineTest(unittest.TestCase):
    """Drive the real run_script path with the cluster simulated on the local box."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scratch = Path(self.tmp.name) / "scratch"
        self.scratch.mkdir(parents=True)
        os.environ["KBENCH_LOCAL_BUNDLE"] = str(Path(self.tmp.name) / "localbundle")

        self.work = Path(self.tmp.name) / "candidate"
        (self.work / "harness").mkdir(parents=True)
        (self.work / "fwd_output.py").write_text("get_output = 1\n")
        (self.work / "harness" / "benchmark.py").write_text("print('bench')\n")
        (self.work / "harness" / "validate.py").write_text("print('val')\n")

        self.submitted = []

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("KBENCH_LOCAL_BUNDLE", None)

    def fake_remote(self, spec_, script, timeout=None):
        # Emulate the slurm CLIs and cat against the local scratch stand-in.
        class Result:
            def __init__(self, out, code=0, err=""):
                self.stdout, self.returncode, self.stderr = out, code, err

        if "sbatch --parsable" in script:
            self.submitted.append(script)
            return Result("4242\n")
        if "sacct" in script:
            return Result("COMPLETED|0:0\n")
        if "squeue" in script:
            return Result("RUNNING\n")
        if script.startswith("cat "):
            target = script.split(" ", 1)[1].strip("'")
            path = Path(target)
            return Result(path.read_text() if path.exists() else "", 0 if path.exists() else 1)
        if script.startswith("mkdir"):
            subprocess.run(["bash", "-c", script], check=True)
            return Result("")
        # The write of the batch script (mkdir + cat > file <<EOF).
        subprocess.run(["bash", "-c", script.replace("<<'KBENCH_EOF'", "<<'KBENCH_EOF'")], check=False)
        return Result("")

    def run_with_fake(self, spec_, phase):
        sess = Session(spec_, self.work, rid="E2E-1")
        # Seed the artifact the container would have produced, under the emulated scratch.
        out_dir = sess.host_bundle.replace(str(self.scratch), str(self.scratch))
        artifact = Path(self.scratch) / Path(sess.host_bundle).relative_to(self.scratch) / "out" / phase / "quick" / "benchmark.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"value": 0.5, "unit": "ms", "lower_is_better": True, "samples": [0.5]}))
        return sess, artifact

    def test_batch_run_ships_submits_waits_and_reads_back(self):
        spec_ = make_spec(scratch_root=str(self.scratch))
        sess, artifact = self.run_with_fake(spec_, "b")
        out_local = Path(self.tmp.name) / "out.json"

        with patch("kbench.slurm_batch._remote", side_effect=self.fake_remote), \
             patch("subprocess.run") as fake_proc, \
             patch("kbench.task.workdir_rev", return_value="abc123"), \
             patch("time.sleep"):
            # subprocess.run is used for rsync + the tar|ssh pipe; make them succeed.
            fake_proc.return_value = subprocess.CompletedProcess([], 0, "", "")
            exit_code, _ = sess.run_script(
                script_path=self.work / "harness" / "benchmark.py",
                work=self.work,
                harness_dir=self.work / "harness",
                mode="quick",
                phase="b",
                out_local=out_local,
                timeout_s=30,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self.submitted), 1)
        self.assertTrue(out_local.exists())
        data = json.loads(out_local.read_text())
        self.assertEqual(data["value"], 0.5)

    def test_no_cluster_path_leaks_the_framework_checkout(self):
        # The candidate is shipped as repo/, never as an import of kbench itself.
        spec_ = make_spec(scratch_root=str(self.scratch))
        sess, _ = self.run_with_fake(spec_, "a")
        submit = sess.submit_script(
            "validate.py", "quick", "a",
            "/work/harness/validate.py", "/work/repo", "/work/out/a/quick/validate.json",
        )
        # The framework must never be referenced as a shipped/importable path.  The
        # [kbench] echo label and the KBENCH_SEED_OK banner are intended driver text.
        for leak in ("import kbench", "kbench/slurm", "kbench/task", "/kopt/", "auto-gpu-kernel"):
            self.assertNotIn(leak, submit)


if __name__ == "__main__":
    unittest.main()
