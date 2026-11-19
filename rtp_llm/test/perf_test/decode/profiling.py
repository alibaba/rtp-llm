"""Selected-batch profile execution and atomic artifact manifests."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec
from rtp_llm.test.perf_test.errors import error_message
from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.profiling import new_profile_manifest, trace_artifact
from rtp_llm.test.perf_test.storage import atomic_write_json


def discover_profile_artifacts(
    log_path: Optional[str], offset: int, trace_name: str
) -> Tuple[str, ...]:
    if not log_path or not Path(log_path).is_file():
        return ()
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        lines = stream.readlines()
    marker = "profiler trace saved:"
    paths = []
    for line in lines:
        if marker not in line:
            continue
        candidate = line.split(marker, 1)[1].strip()
        if trace_name in candidate and Path(candidate).is_file():
            paths.append(str(Path(candidate).resolve()))
    return tuple(sorted(set(paths)))


def sanitize_error(error: Exception) -> str:
    return error_message(error)


class DecodeProfiler:
    def __init__(
        self,
        batch_executor,
        *,
        run_root: Path,
        log_path: Optional[str],
        trace_directory: Optional[Path] = None,
        world_size: int = 1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        artifact_discoverer: Callable[
            [Optional[str], int, str], Tuple[str, ...]
        ] = discover_profile_artifacts,
    ):
        self._batch_executor = batch_executor
        self._world_size = world_size
        self._run_root = run_root
        self._log_path = log_path
        work_directory = Path(
            os.environ.get(
                "MAGA_SERVER_WORK_DIR",
                os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()),
            )
        )
        trace_directory = Path(
            trace_directory or os.environ.get("TORCH_CUDA_PROFILER_DIR") or "."
        )
        self._trace_directory = (
            trace_directory
            if trace_directory.is_absolute()
            else work_directory / trace_directory
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._artifact_discoverer = artifact_discoverer
        self._requires_session_restart = False

    @property
    def requires_session_restart(self) -> bool:
        return self._requires_session_restart

    @property
    def session_reusable(self) -> bool:
        return not self.requires_session_restart

    def _trace_files(self, trace_name: str) -> Dict[Path, Tuple[int, int]]:
        files = {}
        directories = {self._trace_directory}
        if self._log_path:
            directories.add(Path(self._log_path).parent)
        paths = (
            path
            for directory in directories
            for path in directory.glob(f"{trace_name}*.json")
        )
        for path in paths:
            try:
                stat = path.stat()
                files[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        return files

    def _read_complete_traces(self, candidates, before) -> Dict[Path, dict]:
        traces = {}
        for candidate in sorted(set(candidates)):
            path = Path(candidate).resolve()
            try:
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if signature == before.get(path):
                    continue
                trace = json.loads(path.read_text(encoding="utf-8"))
                after = path.stat()
                if signature != (after.st_mtime_ns, after.st_size):
                    continue
            except (OSError, UnicodeError, ValueError):
                continue
            if isinstance(trace, dict) and isinstance(trace.get("traceEvents"), list):
                traces[path] = trace
        return traces

    def _retain_complete_traces(
        self, traces: Dict[Path, dict], directory: Path
    ) -> Tuple[str, ...]:
        artifacts = []
        for path, trace in traces.items():
            retained = directory / path.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            if retained.exists():
                raise RuntimeError(f"ambiguous profile trace filename: {path.name}")
            atomic_write_json(retained, trace)
            artifacts.append(str(retained.relative_to(self._run_root)))
        return tuple(sorted(artifacts))

    def profile(self, case: DecodeCaseSpec, batch_size: int) -> Dict[str, Any]:
        if not self.session_reusable:
            raise RuntimeError(
                "decode profiling session requires engine shutdown after incomplete capture"
            )
        identity = case.to_identity()
        case_id_value = case_id("decode", identity)
        session_id = "profile-" + uuid.uuid4().hex
        # The engine truncates prefixes at 64 characters. A unique short prefix
        # lets collected ranks join the same session without counter assumptions.
        trace_name = session_id
        relative_manifest = Path("profiles") / session_id / "manifest.json"
        trace_directory = self._run_root / "traces" / session_id / case_id_value
        manifest_path = self._run_root / relative_manifest
        status = "completed"
        error_message = None
        artifacts: Tuple[str, ...] = ()
        artifact_records = []
        profile_invoked = False
        capture_completed = False
        try:
            before_traces = self._trace_files(trace_name)
            for candidate in discover_profile_artifacts(self._log_path, 0, trace_name):
                path = Path(candidate).resolve()
                stat = path.stat()
                before_traces[path] = (stat.st_mtime_ns, stat.st_size)
            log_offset = (
                Path(self._log_path).stat().st_size
                if self._log_path and Path(self._log_path).is_file()
                else 0
            )
            profile_invoked = True
            self._batch_executor.profile(
                case,
                batch_size,
                trace_name=trace_name,
                steps=case.profiling.steps,
                enable_all_ranks=case.profiling.enable_all_ranks,
            )
            deadline = self._monotonic() + case.profiling.flush_timeout_seconds
            while self._monotonic() < deadline:
                candidates = self._artifact_discoverer(
                    self._log_path, log_offset, trace_name
                )
                candidates = tuple(candidates) + tuple(
                    str(path) for path in self._trace_files(trace_name)
                )
                traces = self._read_complete_traces(candidates, before_traces)
                counts = {}
                for path in traces:
                    rank = re.search(r"(?:^|_)wr(\d+)(?:_|\.)", path.name)
                    rank = int(rank.group(1)) if rank else 0
                    counts[rank] = counts.get(rank, 0) + 1
                ranks = (
                    range(self._world_size) if case.profiling.enable_all_ranks else (0,)
                )
                if all(counts.get(rank, 0) >= case.profiling.runs for rank in ranks):
                    # Stable source traces establish capture completion even if
                    # retaining diagnostic copies fails on the run filesystem.
                    capture_completed = True
                    artifacts = self._retain_complete_traces(traces, trace_directory)
                    artifact_records = [
                        dict(
                            trace_artifact(
                                self._run_root,
                                session_id,
                                case_id_value,
                                self._run_root / path,
                                backend="kineto",
                            ),
                            phase="decode",
                        )
                        for path in artifacts
                    ]
                    break
                self._sleep(0.2)
            if not artifacts:
                raise RuntimeError("trace flush timeout")
        except Exception as error:
            status = "failed"
            error_message = sanitize_error(error)
            artifacts = ()
            artifact_records = []
        finally:
            # A lost start acknowledgement can still leave some ranks armed.
            # StartProfile has no cancellation API; only a completed capture
            # establishes that another formal case may use this engine.
            attempted = getattr(self._batch_executor, "profile_start_attempted", True)
            self._requires_session_restart = (
                profile_invoked and attempted is not False and not capture_completed
            )
        manifest = new_profile_manifest(
            "decode",
            session_id,
            [case_id_value],
            backend="kineto",
            capture_scope=f"{case.profiling.steps} forward steps per capture; formal measurement excluded",
        )
        manifest["status"] = status
        manifest["error"] = error_message
        manifest["cases"][0].update(
            trace_name=trace_name,
            status=status,
            error=error_message,
            artifacts=artifact_records,
        )
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(manifest_path, manifest)
        except Exception as error:
            # Profiling is diagnostic: persistence failures must not escape to
            # the case executor and discard the completed formal measurement.
            message = f"profile manifest persistence failed: {sanitize_error(error)}"
            status = "failed"
            error_message = sanitize_error(
                RuntimeError(
                    f"{message}; capture error: {error_message}"
                    if error_message
                    else message
                )
            )
        return {
            "status": status,
            "manifest_path": str(relative_manifest),
            "error": error_message,
        }
