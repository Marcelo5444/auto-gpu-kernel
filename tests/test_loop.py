"""Loop stopping rules, driven by a fake omp client that fakes each turn's side effects."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kopt.loop import LOG_PROMPT, Loop


class FakeClient:
    """Each script entry is what the agent 'does' in one turn:
    'log'   -> writes experiments/exp_N/result.md (a logged experiment)
    'bench' -> appends a line to .kopt/bench.jsonl (ran kbench, no record)
    'idle'  -> nothing
    """

    def __init__(self, project: Path, script: list[str]):
        self.project = project
        self.script = list(script)
        self.prompts: list[str] = []
        self.calls = 0

    def prompt_and_wait(self, prompt, timeout=None):
        self.prompts.append(prompt)
        action = self.script.pop(0) if self.script else "idle"
        if action == "log":
            n = len(list((self.project / "experiments").glob("exp_*"))) + 1
            d = self.project / "experiments" / f"exp_{n}"
            d.mkdir(parents=True)
            (d / "result.md").write_text("done\n")
        elif action == "bench":
            path = self.project / ".kopt" / "bench.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(json.dumps({"mode": "quick"}) + "\n")
        self.calls += 1
        return SimpleNamespace(assistant_text="ok")

    def get_session_stats(self):
        # Cost/tokens/tool_calls grow every turn so no turn looks "dead".
        return SimpleNamespace(
            cost=0.1 * self.calls,
            tokens=SimpleNamespace(total=100 * self.calls),
            tool_calls=3 * self.calls,
        )

    def stop(self):
        pass


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name)
        (self.project / "experiments").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_loop(self, script: list[str], **kw) -> tuple[Loop, FakeClient]:
        loop = Loop(project=self.project, **kw)
        client = FakeClient(self.project, script)

        def connect(self_, log):
            self_._client = client
            self_._base = (0.0, 0, 0)
            return client

        with patch.object(Loop, "_connect", connect):
            loop.run()
        return loop, client

    def test_logged_turns_count_as_experiments(self) -> None:
        loop, client = self.run_loop(["log", "log"], max_iterations=2)
        self.assertEqual([i.experiment for i in loop.history], ["exp_1", "exp_2"])
        self.assertEqual(client.prompts, [loop.prompt, loop.prompt])

    def test_unlogged_turn_gets_one_recovery_prompt(self) -> None:
        # Turn 1 benchmarks but never logs; turn 2 is asked only to write the record.
        loop, client = self.run_loop(["bench", "log", "log"], max_iterations=3)
        self.assertTrue(loop.history[0].unlogged)
        self.assertFalse(loop.history[0].stalled)
        self.assertEqual(client.prompts[1], LOG_PROMPT)
        self.assertTrue(loop.history[1].recovery)
        self.assertEqual(loop.history[1].experiment, "exp_1")
        self.assertEqual(client.prompts[2], loop.prompt)

    def test_three_idle_turns_stop_the_run(self) -> None:
        loop, _ = self.run_loop(["idle", "idle", "idle", "log"], max_iterations=10)
        self.assertEqual(len(loop.history), 3)
        self.assertTrue(all(i.stalled for i in loop.history))

    def test_benchmarking_without_logging_does_not_loop_forever(self) -> None:
        # bench, recovery turn also benches without logging, bench, ... must terminate.
        loop, client = self.run_loop(["bench"] * 10, max_iterations=10)
        self.assertLess(len(loop.history), 10)
        # Recovery prompts are never issued twice in a row.
        for a, b in zip(client.prompts, client.prompts[1:], strict=False):
            self.assertFalse(a == LOG_PROMPT and b == LOG_PROMPT)


if __name__ == "__main__":
    unittest.main()
