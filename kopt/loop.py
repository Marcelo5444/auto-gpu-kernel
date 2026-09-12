"""Drive omp through repeated optimization iterations.

One `omp` process, one prompt per iteration. The loop owns the stopping rules the agent
cannot be trusted to enforce on itself: spend, iteration count, and whether anything
actually happened.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omp_rpc import RpcClient

from kopt.record import Recorder, new_run_log

DEFAULT_PROMPT = "/skill:optimize"
LOG_PROMPT = "/skill:log-experiment"


@dataclass
class Iteration:
    idx: int
    experiment: str | None
    """Experiment that gained a `result.md` during this turn, if any."""
    benchmarks: int
    """kbench runs recorded in .kopt/bench.jsonl during this turn."""
    cost: float
    tokens: int
    tool_calls: int
    seconds: float
    assistant_text: str
    recovery: bool = False
    """This turn was the one-off log-experiment prompt after an unlogged turn."""

    @property
    def unlogged(self) -> bool:
        """Benchmarks ran but no experiment was logged — work without a record.
        Usually the turn was cut short (context, --max-time) before log-experiment."""
        return self.experiment is None and self.benchmarks > 0

    @property
    def stalled(self) -> bool:
        """Nothing measurable happened: no experiment logged and no benchmark run."""
        return self.experiment is None and self.benchmarks == 0


@dataclass
class Loop:
    project: Path
    prompt: str = DEFAULT_PROMPT
    max_iterations: int = 10
    budget: float | None = None
    """Hard spend cap in USD, checked between iterations."""
    max_time: str | None = None
    """Per-iteration wall clock passed to omp, e.g. "20m"."""
    timeout: float = 3600.0
    """Seconds to wait for one turn. omp-rpc defaults to 30s, which no real
    optimization iteration finishes inside."""
    model: str | None = None
    thinking: str | None = None
    """omp thinking level: off|minimal|low|medium|high|xhigh|max."""
    fresh: bool = False
    """Start a new omp session each iteration instead of persisting one."""
    history: list[Iteration] = field(default_factory=list)
    _client: Any = field(default=None, init=False, repr=False)
    _base: tuple[float, int, int] = field(default=(0.0, 0, 0), init=False, repr=False)
    """Cumulative session (cost, tokens, tool_calls) at the start of the current turn."""

    # --- progress tracking ----------------------------------------------
    def _logged(self) -> set[str]:
        """Experiments that have a `result.md`.

        Tracking `result.md` rather than the `exp_N/` directory matters: when the
        research agent reserves `exp_N/` with a `plan.md`, the next iteration fills in
        that *existing* folder (see the folder-reservation rule in the optimize skill).
        Watching for new directories would score that legitimate turn as a stall.
        `log-experiment` writes `result.md` once and never overwrites it, so this
        counts each logged experiment exactly once either way.
        """
        root = self.project / "experiments"
        if not root.is_dir():
            return set()
        return {p.parent.name for p in root.glob("exp_*/result.md")}

    def _benchmarks(self) -> int:
        path = self.project / ".kopt" / "bench.jsonl"
        try:
            return sum(1 for line in path.open() if line.strip())
        except OSError:
            return 0

    # --- reporting ------------------------------------------------------
    @property
    def spent(self) -> float:
        return sum(i.cost for i in self.history)

    def _should_stop(self) -> str | None:
        if len(self.history) >= self.max_iterations:
            return f"reached max_iterations={self.max_iterations}"
        if self.budget is not None and self.spent >= self.budget:
            return f"reached budget ${self.budget:.2f} (spent ${self.spent:.2f})"
        recent = self.history[-3:]
        if len(recent) == 3 and all(i.experiment is None for i in recent):
            return "3 consecutive iterations logged no experiment"
        return None

    # --- main -----------------------------------------------------------
    def _connect(self, log: Recorder):
        """Open an omp session. Sessions persist across iterations by default so the
        agent keeps its recent experiments in context — `summary.md` is a lossy
        summary of what it just did, and the detail that didn't make the row is
        often what matters next."""
        extra = ["--max-time", self.max_time] if self.max_time else []
        client = RpcClient(
            cwd=str(self.project),
            model=self.model,
            thinking=self.thinking,
            extra_args=tuple(extra),
            request_timeout=self.timeout,
            # The default 10k-event ring is smaller than one verbose turn
            # (high-thinking iterations stream 10M+ tokens); overflow makes
            # prompt_and_wait lose agent_end and raises RpcError mid-run.
            max_event_history=None,
        ).start()  # spawns the process; RpcClient() alone does not
        client.install_headless_ui()
        log.attach(client)
        state = client.get_state()
        stats = client.get_session_stats()
        print(f"omp session {state.session_id} | model {state.model.id}")
        log.write("session_start", session_id=state.session_id, model=state.model.id)
        self._client = client
        self._base = (stats.cost, stats.tokens.total, stats.tool_calls)
        return client

    def _turn(self, idx: int, log: Recorder, prompt: str) -> Iteration:
        client = self._client if self._client is not None else self._connect(log)
        before = self._logged()
        benchmarks_before = self._benchmarks()
        start = time.monotonic()
        turn = client.prompt_and_wait(prompt, timeout=self.timeout)
        elapsed = time.monotonic() - start

        # Stats are cumulative for the session; diff against the last turn.
        stats = client.get_session_stats()
        cost, tokens, calls = self._base
        self._base = (stats.cost, stats.tokens.total, stats.tool_calls)
        logged = sorted(self._logged() - before)
        return Iteration(
            idx=idx,
            experiment=logged[-1] if logged else None,
            benchmarks=max(0, self._benchmarks() - benchmarks_before),
            cost=stats.cost - cost,
            tokens=stats.tokens.total - tokens,
            tool_calls=stats.tool_calls - calls,
            seconds=elapsed,
            assistant_text=turn.assistant_text or "",
        )

    @staticmethod
    def _is_dead(it: Iteration) -> bool:
        """A turn that returns instantly having spent nothing means the session is
        gone — `--max-time` expired, or omp exited. Not a lazy agent."""
        return it.seconds < 5 and it.cost == 0 and it.tool_calls == 0

    def run(self) -> list[Iteration]:
        log = Recorder(new_run_log(self.project))
        print(f"run log: {log.path}")
        log.write("run_start", project=str(self.project), model=self.model or "(default)",
                  thinking=self.thinking or "(default)", max_iterations=self.max_iterations,
                  budget=self.budget, fresh=self.fresh)
        try:
            while True:
                if (reason := self._should_stop()) is not None:
                    print(f"\nstopping: {reason}")
                    log.write("run_end", reason=reason, iterations=len(self.history),
                              spent=self.spent)
                    return self.history

                idx = len(self.history) + 1
                print(f"\n=== iteration {idx} ===")
                if self.fresh:
                    self._close()

                # A turn that benchmarked but never logged (cut off before
                # log-experiment) gets one recovery turn that only writes the record.
                prompt = self.prompt
                last = self.history[-1] if self.history else None
                if last is not None and last.unlogged and not last.recovery:
                    print(f"  previous turn ran {last.benchmarks} benchmark(s) but logged"
                          " no experiment — asking for the record first")
                    prompt = LOG_PROMPT

                it = self._turn(idx, log, prompt)
                it.recovery = prompt == LOG_PROMPT
                if self._is_dead(it):
                    # Session expired (commonly --max-time). Reconnect and retry once;
                    # a fresh session still sees every experiment on disk.
                    print("  session ended — reconnecting")
                    log.write("reconnect", idx=idx)
                    self._close()
                    it = self._turn(idx, log, prompt)
                    it.recovery = prompt == LOG_PROMPT
                    if self._is_dead(it):
                        raise SystemExit("omp unusable after reconnect")

                self.history.append(it)
                log.write("iteration", spent=self.spent, **asdict(it))
                outcome = it.experiment or (
                    f"UNLOGGED ({it.benchmarks} benchmark runs)"
                    if it.unlogged
                    else "NOTHING LOGGED"
                )
                print(
                    f"  {outcome}"
                    f" | {it.seconds:.0f}s | {it.tool_calls} tools"
                    f" | {it.tokens} tok | ${it.cost:.4f} (total ${self.spent:.4f})"
                )
                if last := it.assistant_text.strip():
                    print(f"  {last.splitlines()[-1][:160]}")
        except BaseException as exc:
            # Abort, timeout, RpcError, Ctrl-C: still close the log so `kopt watch`
            # does not show the run as live forever.
            reason = str(exc) if isinstance(exc, SystemExit) else f"error: {exc!r}"
            print(f"\nABORT: {reason}")
            log.write("run_end", reason=reason, iterations=len(self.history), spent=self.spent)
            raise
        finally:
            self._close()

    def _close(self) -> None:
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
