"""Isolated profiling of cases reconstructed from a persisted prefill plan."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Union

import requests

from rtp_llm.test.perf_test.config_schema import DEFAULT_PREFILL_TRANSPORT
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.prefill.planner import require_concurrent_prefill_plan
from rtp_llm.test.perf_test.profiling import new_profile_manifest, trace_artifact
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock
from rtp_llm.test.perf_test.storage import atomic_write_json

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class PrefillProfiler:
    def __init__(self, runner_factory: Callable[[Path], Any]):
        self._runner_factory = runner_factory

    @staticmethod
    def _load_plan(root: Path) -> ExecutionPlan:
        try:
            value = json.loads(
                (root / "execution_plan.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PlanningError(
                "profile requires a valid persisted execution plan"
            ) from error
        plan = ExecutionPlan.from_dict(value)
        if plan.mode != "prefill" or plan.backend != "local":
            raise PlanningError("profile requires a local prefill run")
        require_concurrent_prefill_plan(plan)
        if thaw(plan.resolved_config)["model"]["topology"]["dp"] != 1:
            raise PlanningError(
                "prefill profiling requires model.topology.dp=1; "
                "profile requests are not bound to the armed DP replica"
            )
        return plan

    def run(
        self,
        root: Union[str, Path],
        case_ids: Sequence[str],
        *,
        session_id: str,
    ) -> Dict[str, Any]:
        root = Path(root)
        if not _SESSION_ID.fullmatch(session_id):
            raise PlanningError(f"invalid profile session_id: {session_id!r}")
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise PlanningError("profile case IDs must be non-empty and unique")
        plan = self._load_plan(root)
        by_id = {case.case_id: case for case in plan.cases}
        for case_id in case_ids:
            if case_id not in by_id:
                raise PlanningError(f"unknown profile case: {case_id}")

        with RunDirectoryLock(root, action="profile"):
            profile_dir = root / "profiles" / session_id
            try:
                profile_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError as error:
                raise PlanningError(
                    f"profile session already exists: {session_id}"
                ) from error
            cases_dir = profile_dir / "cases"
            cases_dir.mkdir()
            results = []
            settings = thaw(plan.resolved_config)["workload"].get("profiling", {})
            backend = settings.get("backend", "kineto")
            manifest = new_profile_manifest(
                "prefill",
                session_id,
                case_ids,
                backend=backend,
                capture_scope=capture_scope(backend),
            )
            trace_dir = root / "traces" / session_id
            atomic_write_json(profile_dir / "manifest.json", manifest)
            try:
                trace_dir.mkdir(parents=True, exist_ok=False)
                runner = self._runner_factory(trace_dir)
                for case_id, profile_case in zip(case_ids, manifest["cases"]):
                    case = thaw(by_id[case_id].identity)
                    case["case_id"] = case_id
                    try:
                        if getattr(runner, "session_reusable", True) is False:
                            raise RuntimeError(
                                "profile case was not executed: an incomplete capture "
                                "requires profiling engine shutdown"
                            )
                        result = runner.run_case(case)
                        if result.get("status") == "ok":
                            artifacts = result.get("trace_files") or []
                            if not artifacts:
                                raise RuntimeError(
                                    "profile produced no trace artifacts"
                                )
                            profile_case["artifacts"] = [
                                trace_artifact(
                                    root,
                                    session_id,
                                    case_id,
                                    trace_dir / artifact,
                                    backend=backend,
                                )
                                for artifact in artifacts
                            ]
                            result["trace_files"] = [
                                item["path"] for item in profile_case["artifacts"]
                            ]
                    except Exception as error:
                        result = {
                            "case_id": case_id,
                            "status": "request_failed",
                            "metrics": None,
                            "errors": [
                                {"code": "profile_failed", "message": str(error)}
                            ],
                        }
                    profile_case.update(
                        status=(
                            "completed" if result.get("status") == "ok" else "failed"
                        ),
                        error=(
                            None
                            if result.get("status") == "ok"
                            else str(result.get("errors", []))
                        ),
                    )
                    if profile_case["status"] == "failed":
                        profile_case["artifacts"] = []
                    atomic_write_json(cases_dir / f"{case_id}.json", result)
                    atomic_write_json(profile_dir / "manifest.json", manifest)
                    results.append(result)
                outcome = {
                    "schema_version": 1,
                    "session_id": session_id,
                    "run_id": plan.result_directory,
                    "case_ids": list(case_ids),
                    "status": (
                        "completed"
                        if all(result.get("status") == "ok" for result in results)
                        else "failed"
                    ),
                    "results": results,
                }
                manifest["status"] = outcome["status"]
                atomic_write_json(profile_dir / "manifest.json", manifest)
                atomic_write_json(profile_dir / "profile_result.json", outcome)
                return outcome
            except BaseException as error:
                manifest.update(
                    status=(
                        "interrupted"
                        if isinstance(error, (KeyboardInterrupt, SystemExit))
                        else "failed"
                    ),
                    error=str(error),
                )
                atomic_write_json(profile_dir / "manifest.json", manifest)
                raise


def capture_scope(backend):
    """Describe the backend's actual capture boundary, independently of replay."""
    if backend == "nsys":
        return "one measured concurrent round; seeds excluded"
    return "one forward step; seeds excluded"


class TraceCapture:
    def __init__(
        self,
        directory,
        name,
        *,
        backend="kineto",
        timeout=120,
        tp_size=1,
        port=None,
        nsys_path=None,
        nsys_session=None,
        post=requests.post,
        run=subprocess.run,
    ):
        self.directory = Path(directory).resolve()
        self.name = name
        self.backend = backend
        self.timeout = timeout
        self.tp_size = tp_size
        self.port = port
        self.nsys_path = nsys_path or "nsys"
        self.nsys_session = nsys_session
        self.post = post
        self.run = run
        self.commands = []
        self._lock = threading.Lock()
        self._started = False
        self._attempted = False
        self._error = None
        self._stopped = False

    def _command(self, action, *options):
        argv = [self.nsys_path, action, "--session=" + self.nsys_session, *options]
        record = {"command": argv}
        self.commands.append(record)
        try:
            result = self.run(
                argv, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            record.update(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
            if result.returncode:
                raise RuntimeError(f"nsys {action} failed: {result.stderr}")
        except BaseException as error:
            record["error"] = str(error)
            raise

    def start(self):
        # Every measured member holds this gate until the start acknowledgement.
        with self._lock:
            if self._error is not None:
                raise RuntimeError(str(self._error))
            if self._started:
                return
            self._attempted = True
            try:
                if self.backend == "nsys":
                    self._command(
                        "start",
                        "--sample=process-tree",
                        "--cpuctxsw=process-tree",
                        "--output=" + str(self.directory / self.name),
                    )
                else:
                    response = self.post(
                        f"http://127.0.0.1:{self.port}/start_profile",
                        json={
                            "gen_timeline": True,
                            "trace_name": self.name,
                            "start_step": 0,
                            "num_steps": 1,
                            "enable_all_rank": True,
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    ack = response.json()
                    if ack.get("status") != "ok" or ack.get("error"):
                        raise RuntimeError(f"start_profile failed: {ack}")
                self._started = True
            except BaseException as error:
                self._error = error
                raise

    @property
    def requires_session_restart(self):
        # StartProfile has no cancellation contract. Even a failed HTTP
        # acknowledgement may follow a successful arm on some TP ranks.
        return self.backend != "nsys" and self._attempted and not self._stopped

    def abort(self):
        if self.requires_session_restart:
            state = (
                "armed Kineto capture did not complete"
                if self._started
                else "Kineto start acknowledgement failed; capture state is unknown"
            )
            raise RuntimeError(
                f"{state}; /start_profile cannot cancel capture; "
                "the profiling engine must be stopped before another case"
            )
        if self.backend == "nsys" and self._attempted and not self._stopped:
            self._command("stop")
            self._stopped = True

    def finish(self):
        if not self._started:
            raise RuntimeError("no measured request was traced")
        if self.backend == "nsys":
            self.abort()
            report = self.directory / (self.name + ".nsys-rep")
            if not report.is_file() or report.stat().st_size == 0:
                raise RuntimeError(
                    f"nsys stop did not produce a nonempty report: {report}"
                )
            return [report.name]
        deadline = time.monotonic() + self.timeout
        complete = {}
        while True:
            for rank in range(self.tp_size):
                for path in self.directory.glob(f"{self.name}_wr{rank}_*.json"):
                    try:
                        trace = json.loads(path.read_text())
                        if isinstance(trace, dict) and trace.get("traceEvents"):
                            complete[rank] = path.name
                            break
                    except (OSError, ValueError):
                        pass
            if len(complete) == self.tp_size:
                self._stopped = True
                return [complete[rank] for rank in sorted(complete)]
            if time.monotonic() >= deadline:
                missing = sorted(set(range(self.tp_size)) - complete.keys())
                raise RuntimeError(f"profile trace timeout; missing TP ranks {missing}")
            time.sleep(min(0.1, self.timeout))


class TraceClient:
    def __init__(self, client, capture):
        self.client = client
        self.capture = capture

    def configure_scheduler(self, *args, **kwargs):
        return self.client.configure_scheduler(*args, **kwargs)

    def request(self, **kwargs):
        if kwargs["phase"] == "measure":
            self.capture.start()
        return self.client.request(**kwargs)


class ProfiledRunner:
    def __init__(self, client, tokenizer, directory, config, session_id):
        self.client = client
        self.tokenizer = tokenizer
        self.directory = Path(directory)
        self.config = config
        self.session_id = session_id
        self.session_reusable = True

    def run_case(self, case):
        if not self.session_reusable:
            raise RuntimeError(
                "profiling session requires engine shutdown after incomplete capture"
            )
        from rtp_llm.test.perf_test.prefill.probe import resolve_probe_block_size
        from rtp_llm.test.perf_test.prefill.runner import PrefillRunner

        settings = self.config["workload"].get("profiling", {})
        capture = TraceCapture(
            self.directory,
            case["case_id"],
            backend=settings.get("backend", "kineto"),
            timeout=settings.get("trace_timeout_seconds", 120),
            tp_size=self.config["model"]["topology"]["tp"],
            port=self.client.port,
            nsys_path=settings.get("nsys_path"),
            nsys_session=self.session_id,
        )
        diagnostic = dict(case, measure_runs=1)
        result = None
        try:
            result = PrefillRunner(
                TraceClient(self.client, capture),
                self.tokenizer,
                commit_tail_tokens=self.config["workload"]["commit_tail_tokens"],
                max_batch_tokens=self.config["engine"]["max_batch_tokens"],
                workspace_tokens=self.config["workload"].get("workspace_tokens"),
                input_ids_only=self.config["workload"].get(
                    "transport", DEFAULT_PREFILL_TRANSPORT
                )
                == "dashsc_input_ids",
                cache_block_size=resolve_probe_block_size(self.config),
            ).run_case(diagnostic)
            if result["status"] == "ok":
                files = capture.finish()
                case_dir = self.directory / case["case_id"]
                case_dir.mkdir(exist_ok=False)
                result["trace_files"] = []
                for name in files:
                    (self.directory / name).rename(case_dir / name)
                    result["trace_files"].append(case["case_id"] + "/" + name)
        except Exception as error:
            result = {
                "case_id": case["case_id"],
                "status": "trace_failed",
                "metrics": None,
                "errors": [{"code": "trace_failed", "message": str(error)}],
            }
        finally:
            if capture.requires_session_restart:
                self.session_reusable = False
            try:
                capture.abort()
            except Exception as error:
                # No result means a BaseException (such as EngineInterrupted)
                # is propagating. Cleanup must preserve that interruption.
                if result is not None:
                    result["status"] = "trace_failed"
                    result.setdefault("errors", []).append(
                        {"code": "trace_stop_failed", "message": str(error)}
                    )
        result["nsys_commands"] = capture.commands
        result["capture_scope"] = capture_scope(settings.get("backend", "kineto"))
        return result
