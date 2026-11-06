"""Selected-batch profile execution and atomic artifact manifests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec
from rtp_llm.test.perf_test.plan import case_id
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
    return " ".join(str(error).splitlines())[:500]


class DecodeProfiler:
    def __init__(
        self,
        batch_executor,
        *,
        run_root: Path,
        log_path: Optional[str],
        trace_directory: Optional[Path] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        artifact_discoverer: Callable[
            [Optional[str], int, str], Tuple[str, ...]
        ] = discover_profile_artifacts,
    ):
        self._batch_executor = batch_executor
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

    def _retain_complete_traces(self, candidates, before) -> Tuple[str, ...]:
        artifacts = []
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
                retained = self._run_root / "traces" / path.name
                retained.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(retained, trace)
                artifacts.append(str(retained.resolve()))
        return tuple(sorted(artifacts))

    def profile(self, case: DecodeCaseSpec, batch_size: int) -> Dict[str, Any]:
        identity = case.to_identity()
        case_id_value = case_id("decode", identity)
        trace_name = (
            f"decode_in{case.input_lengths[0]}_out{case.output_length}_bs{batch_size}_"
            f"{case_id_value.rsplit('-', 1)[-1]}"
        )
        relative_manifest = Path("profiles") / case_id_value / "profile_manifest.json"
        manifest_path = self._run_root / relative_manifest
        status = "completed"
        error_message = None
        artifacts: Tuple[str, ...] = ()
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
                artifacts = self._retain_complete_traces(candidates, before_traces)
                if artifacts:
                    break
                self._sleep(0.2)
            if not artifacts:
                raise RuntimeError("trace flush timeout")
        except Exception as error:
            status = "failed"
            error_message = sanitize_error(error)
            artifacts = ()
        manifest = {
            "schema_version": 1,
            "status": status,
            "case_id": case_id_value,
            "batch_size": batch_size,
            "input_lengths": list(case.input_lengths[:batch_size]),
            "output_length": case.output_length,
            "runs": case.profiling.runs,
            "steps": case.profiling.steps,
            "enable_all_ranks": case.profiling.enable_all_ranks,
            "trace_name": trace_name,
            "artifacts": list(artifacts),
            "manifest_path": str(relative_manifest),
            "error": error_message,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest)
        return manifest
