"""Viewer for a project: one continuous log across all its runs, the optimization
chart, and the experiments tree.

Knows nothing about omp or the loop — it reads files. Start it before, during, or
after a run; it replays every run so far, then follows the newest and rolls onto the
next one when `kopt run` starts again.

Stdlib only: SSE rather than WebSockets keeps this dependency-free, and browsers
reconnect automatically.
"""

from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kbench.history import kernel_digest
from kbench.history import load as load_history
from kopt import pages
from kopt.record import list_run_logs, runs_dir

# Only these are browsable, and only under experiments/.
VIEWABLE = {".md", ".py", ".json", ".txt", ".log", ".toml", ".cu", ".cuh", ".jsonl"}
MAX_VIEW_BYTES = 2_000_000


def _tail(path: Path, pos: int = 0):
    """Yield (record, end_pos) for complete lines from pos; (None, pos) idle ticks."""
    while True:
        emitted = False
        if path.exists():
            with path.open("rb") as fh:
                fh.seek(pos)
                while True:
                    line = fh.readline()
                    if not line or not line.endswith(b"\n"):
                        break  # partial write; re-read next pass
                    pos = fh.tell()
                    try:
                        yield json.loads(line), pos
                        emitted = True
                    except json.JSONDecodeError:
                        pass
        if not emitted:
            yield None, pos
        time.sleep(0.25)


# A fresh SSE connection replays at most this much of the log. The logs mirror
# every streaming delta and grow to hundreds of MB; replaying one in full is
# what made EventSource drop and re-replay in a loop.
REPLAY_BYTES = 4 * 1024 * 1024

# Records the log page actually renders — everything else is dead weight on the
# wire (message_update deltas alone are ~90% of the bytes).
def _wanted(rec: dict) -> bool:
    kind = rec.get("kind")
    if kind in ("run_start", "run_end", "reconnect", "iteration"):
        return True
    return kind == "event" and rec.get("type") == "message_end"


DESC = re.compile(r"^\*\*Description:\*\*\s*(.+)$", re.M)


def _experiments(root: Path) -> list[dict]:
    """One entry per exp_N: its description and the hash of its kernel snapshot.

    Hashing the snapshot is what ties an experiment to a measurement — the same
    digest kbench records — so the chart labels points without guessing.
    """
    out = []
    if not root.is_dir():
        return out

    # summary.md's Description column is written as "one phrase" by the
    # log-experiment skill — far better as a chart label than result.md's
    # paragraph, so prefer it and fall back only when a row is missing.
    phrases: dict[str, str] = {}
    index = root / "summary.md"
    if index.exists():
        for line in index.read_text(errors="replace").splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 4 and cells[1].isdigit():
                phrases[f"exp_{cells[1]}"] = cells[3][:70]
    for d in sorted(root.glob("exp_*"), key=lambda p: (len(p.name), p.name)):
        if not d.is_dir():
            continue
        result = d / "result.md"
        desc = phrases.get(d.name, "")
        if not desc and result.exists():
            m = DESC.search(result.read_text(errors="replace"))
            if m:
                desc = " ".join(m.group(1).split())[:90]
        snapshots = [p for p in d.glob("*.py") if p.is_file()]
        out.append({
            "exp": d.name,
            "desc": desc,
            "kernel": kernel_digest(snapshots[0]) if len(snapshots) == 1 else "",
        })
    return out


def resolve(project: Path, run: str | None) -> Path | None:
    """An explicitly pinned run log (path or name), else None = follow all runs."""
    if not run:
        return None
    candidate = Path(run)
    if candidate.exists():
        return candidate
    named = runs_dir(project) / (run if run.endswith(".jsonl") else f"{run}.jsonl")
    if named.exists():
        return named
    raise SystemExit(f"no such run: {run}")


def serve(project: Path, port: int = 8765, run: str | None = None,
          host: str = "127.0.0.1") -> None:
    project = Path(project).resolve()
    experiments = (project / "experiments").resolve()
    pinned_log = resolve(project, run)

    def run_logs() -> list[Path]:
        """Every run of this project, oldest first — or just the pinned one."""
        if pinned_log is not None:
            return [pinned_log]
        # Keep resolving so `kopt watch` can start before the first `kopt run`.
        return list_run_logs(project)

    def safe(rel: str) -> Path | None:
        """Confine reads to experiments/ — the path comes from the browser."""
        try:
            target = (experiments / rel).resolve()
        except OSError:
            return None
        if not target.is_relative_to(experiments) or not target.is_file():
            return None
        if target.suffix.lower() not in VIEWABLE:
            return None
        return target

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _emit(self, path: Path, pos: int, rec: dict) -> None:
            # ids are "<logname>:<pos>" so a resume never seeks into another run's file.
            self.wfile.write(f"id: {path.name}:{pos}\ndata: {json.dumps(rec)}\n\n".encode())

        def _skeleton(self, path: Path, upto: int | None = None) -> None:
            """Replay only the run_start / iteration / run_end rows of a log, so an
            old or huge run still contributes its shape without its megabytes."""
            meta = (b'"kind": "run_start"', b'"kind": "run_end"',
                    b'"kind": "iteration"', b'"kind": "reconnect"')
            with path.open("rb") as fh:
                while (upto is None or fh.tell() < upto) and (line := fh.readline()):
                    if not any(m in line for m in meta):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if _wanted(rec):
                        self._emit(path, upto or fh.tell(), rec)
            self.wfile.flush()

        def _stream(self, last_id: str) -> None:
            """One continuous feed for the project: every run so far, then live."""
            while not (logs := run_logs()):
                time.sleep(0.5)
            name, _, offset = last_id.rpartition(":")
            resume = next((p for p in logs if p.name == name), None)
            if resume is not None and offset.isdigit():
                # Reconnect: the browser already has everything up to this point.
                path, pos = resume, min(int(offset), resume.stat().st_size)
            else:
                # Fresh connect: earlier runs as skeletons, newest run in detail
                # (tail only when it is huge; the skipped region as a skeleton).
                for earlier in logs[:-1]:
                    self._skeleton(earlier)
                path, pos = logs[-1], 0
                if path.stat().st_size > REPLAY_BYTES:
                    with path.open("rb") as fh:
                        fh.seek(path.stat().st_size - REPLAY_BYTES)
                        fh.readline()  # skip into line alignment
                        pos = fh.tell()
                    self._skeleton(path, upto=pos)

            idle = 0.0
            while True:
                for record, at in _tail(path, pos):
                    pos = at
                    if record is None:
                        idle += 0.25
                        if idle >= 15:
                            self.wfile.write(b": ping\n\n")  # keep-alive
                            self.wfile.flush()
                            idle = 0.0
                        # Roll onto the next run once `kopt run` starts a new log,
                        # but only after draining everything from the current one.
                        newer = [p for p in run_logs() if p.name > path.name]
                        if newer:
                            path, pos = newer[0], 0
                            break
                        continue
                    idle = 0.0
                    if _wanted(record):
                        self._emit(path, pos, record)
                        self.wfile.flush()

        def do_GET(self):
            url = urlparse(self.path)
            route = url.path

            if route == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self._stream(self.headers.get("Last-Event-ID", ""))
                except (BrokenPipeError, ConnectionResetError):
                    pass  # EventSource reconnects on its own
                return

            if route == "/api/history":
                return self._send(json.dumps(load_history(project)).encode(), "application/json")

            if route == "/api/experiments":
                return self._send(json.dumps(_experiments(experiments)).encode(),
                                  "application/json")

            if route == "/api/tree":
                files = sorted(
                    str(p.relative_to(experiments))
                    for p in experiments.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIEWABLE
                ) if experiments.is_dir() else []
                return self._send(json.dumps(files).encode(), "application/json")

            if route == "/api/file":
                rel = (parse_qs(url.query).get("path") or [""])[0]
                target = safe(rel)
                if target is None:
                    return self._send(b"not viewable", "text/plain", 404)
                return self._send(target.read_bytes()[:MAX_VIEW_BYTES], "text/plain; charset=utf-8")

            page = {
                "/": pages.log_page,
                "/chart": pages.chart_page,
                "/files": pages.files_page,
            }.get(route)
            if page is None:
                return self._send(b"not found", "text/plain", 404)
            return self._send(page().encode(), "text/html; charset=utf-8")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"http://{host}:{port}")
    if pinned_log is not None:
        print(f"  pinned to run {pinned_log.name}")
    elif (n := len(list_run_logs(project))):
        print(f"  following {n} run(s) — new runs are appended as they start")
    else:
        print("  no run yet — the log fills in when `kopt run` starts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
