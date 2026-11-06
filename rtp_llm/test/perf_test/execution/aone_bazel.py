"""Production Bazel adapter for AOne prefill execution."""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from rtp_llm.test.perf_test.aone.package import GeneratedAOnePackage
from rtp_llm.test.perf_test.aone.plan import AONE_GPU_PLATFORMS, AONE_RESOURCES
from rtp_llm.test.perf_test.execution.aone import AOneExecutor
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan

_TARGET = re.compile(r"//(?P<package>[A-Za-z0-9_.\-/]+):(?P<name>[A-Za-z0-9_.-]+)\Z")
_MAX_RESULT_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1000.0
# The timeout covers build, scheduling, and the perf run itself.
AONE_TEST_TIMEOUT_S = 7000


@dataclass(frozen=True)
class BazelShardJob:
    target: str
    shard_id: str


def _positive_integer(name: str, value: Union[str, int]) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _target_parts(target: str) -> tuple[str, str]:
    match = _TARGET.fullmatch(target)
    if match is None:
        raise ValueError(f"invalid generated AOne target: {target!r}")
    package_path = PurePosixPath(match.group("package"))
    if package_path.is_absolute() or ".." in package_path.parts:
        raise ValueError(f"invalid generated AOne target: {target!r}")
    return match.group("package"), match.group("name")


def _load_json_bytes(value: bytes, source: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AOne shard result {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"AOne shard result {source} must contain an object")
    return payload


def _safe_zip_member(info: zipfile.ZipInfo, archive: Path) -> None:
    name = info.filename
    normalized = name[:-1] if info.is_dir() and name.endswith("/") else name
    parts = normalized.split("/")
    path = PurePosixPath(normalized)
    file_type = (info.external_attr >> 16) & 0o170000
    if (
        not normalized
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in parts)
        or file_type == stat.S_IFLNK
    ):
        raise ValueError(f"unsafe path in {archive}: {name}")


class BazelAOneIntegration:
    """Build one package, run all shard targets together, and collect outputs."""

    def __init__(
        self,
        *,
        workspace_root: Union[str, Path],
        resource: str,
        gpu_type: str,
        max_parallel: int,
        bazel: str = "bazelisk",
        build_jobs: Optional[int] = None,
        startup_args: Sequence[str] = (),
        common_args: Sequence[str] = (),
        testlogs_root: Optional[Union[str, Path]] = None,
        process_env: Optional[Mapping[str, str]] = None,
        run_command: Callable[..., Any] = subprocess.run,
        popen_command: Callable[..., Any] = subprocess.Popen,
        max_result_bytes: int = _MAX_RESULT_BYTES,
        max_archive_members: int = _MAX_ARCHIVE_MEMBERS,
        max_archive_uncompressed_bytes: int = _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
        max_compression_ratio: float = _MAX_COMPRESSION_RATIO,
    ):
        if gpu_type not in AONE_GPU_PLATFORMS:
            supported = ", ".join(sorted(AONE_GPU_PLATFORMS))
            raise ValueError(
                f"unsupported GPU type {gpu_type!r}; supported GPU types: {supported}"
            )
        if resource not in AONE_RESOURCES:
            supported = ", ".join(AONE_RESOURCES)
            raise ValueError(
                f"unsupported AOne resource {resource!r}; supported resources: {supported}"
            )
        self._workspace_root = Path(workspace_root).resolve()
        self._resource = resource
        self._platform = AONE_GPU_PLATFORMS[gpu_type]
        self._max_parallel = _positive_integer("max_parallel", max_parallel)
        self._bazel = bazel
        self._startup_args = tuple(startup_args)
        self._build_jobs = _positive_integer(
            "build_jobs", build_jobs if build_jobs is not None else os.cpu_count() or 1
        )
        self._common_args = tuple(common_args)
        self._testlogs_root = Path(
            testlogs_root or self._workspace_root / "bazel-testlogs"
        )
        self._process_env = dict(os.environ if process_env is None else process_env)
        self._run_command = run_command
        self._popen_command = popen_command
        self._max_result_bytes = _positive_integer("max_result_bytes", max_result_bytes)
        self._max_archive_members = _positive_integer(
            "max_archive_members", max_archive_members
        )
        self._max_archive_uncompressed_bytes = _positive_integer(
            "max_archive_uncompressed_bytes", max_archive_uncompressed_bytes
        )
        if (
            isinstance(max_compression_ratio, bool)
            or not isinstance(max_compression_ratio, (int, float))
            or max_compression_ratio <= 0
        ):
            raise ValueError("max_compression_ratio must be positive")
        self._max_compression_ratio = float(max_compression_ratio)
        self._package: Optional[GeneratedAOnePackage] = None
        self._process: Any = None
        self._waited = False
        self._test_returncode: Optional[int] = None
        self._jobs: Dict[str, BazelShardJob] = {}

    def build(self, package: GeneratedAOnePackage) -> None:
        if self._package is not None:
            raise ValueError("AOne package was already built")
        if not package.targets:
            raise ValueError("AOne package has no shard targets")
        for target in package.targets:
            _target_parts(target)
        command = [
            self._bazel,
            *self._startup_args,
            "build",
            *package.targets,
            f"--config={self._platform}",
            "--config=cicd",
            f"--config={self._resource}_aone_bazel_cache",
            f"--jobs={self._build_jobs}",
            "--remote_download_outputs=minimal",
            "--keep_going",
            *self._common_args,
        ]
        completed = self._run_command(
            command,
            cwd=self._workspace_root,
            env=self._process_env,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Bazel AOne build exited {completed.returncode}")
        self._package = package

    def _start_test(self) -> None:
        assert self._package is not None
        command = [
            self._bazel,
            *self._startup_args,
            "test",
            *self._package.targets,
            f"--config={self._platform}",
            "--config=cicd",
            f"--config={self._resource}_aone_bazel_remote",
            "--strategy=TestRunner=remote",
            f"--jobs={self._max_parallel}",
            f"--test_timeout={AONE_TEST_TIMEOUT_S}",
            "--test_output=errors",
            "--keep_going",
            "--remote_download_outputs=all",
            "--nocache_test_results",
            *self._common_args,
        ]
        self._process = self._popen_command(
            command,
            cwd=self._workspace_root,
            env=self._process_env,
        )

    def submit(self, target: str, shard_id: str) -> BazelShardJob:
        if self._package is None:
            raise ValueError("AOne package must be built before submission")
        if target not in self._package.targets:
            raise ValueError(f"unknown AOne shard target: {target}")
        if shard_id in self._jobs:
            raise ValueError(f"duplicate AOne shard submission: {shard_id}")
        if self._process is None:
            self._start_test()
        job = BazelShardJob(target=target, shard_id=shard_id)
        self._jobs[shard_id] = job
        return job

    def _wait(self) -> None:
        if self._process is None:
            raise ValueError("AOne shards were not submitted")
        if not self._waited:
            try:
                self._test_returncode = int(self._process.wait())
                self._waited = True
            except BaseException:
                self._terminate_process()
                raise

    def _terminate_process(self) -> None:
        try:
            self._process.terminate()
        except Exception:
            return
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass

    def _missing_result(self, shard_id: str) -> None:
        if self._test_returncode:
            raise RuntimeError(
                f"Bazel remote test exited {self._test_returncode}; "
                f"no result was produced for {shard_id}"
            )
        return None

    def _outputs(self, target: str) -> Path:
        package, name = _target_parts(target)
        return self._testlogs_root / package / name / "test.outputs"

    def _read_archive(
        self, archive: Path, expected: str
    ) -> Optional[Mapping[str, Any]]:
        try:
            with zipfile.ZipFile(archive) as stream:
                members = stream.infolist()
                if len(members) > self._max_archive_members:
                    raise ValueError(
                        f"AOne outputs archive exceeds member limit: {archive}"
                    )
                if (
                    sum(info.file_size for info in members)
                    > self._max_archive_uncompressed_bytes
                ):
                    raise ValueError(
                        f"AOne outputs archive exceeds uncompressed size limit: {archive}"
                    )
                for info in members:
                    _safe_zip_member(info, archive)
                matches = [info for info in members if info.filename == expected]
                if not matches:
                    return None
                if len(matches) != 1:
                    raise ValueError(
                        f"ambiguous AOne shard result in {archive}: {expected}"
                    )
                result = matches[0]
                if result.file_size > self._max_result_bytes:
                    raise ValueError(f"AOne shard result exceeds size limit: {archive}")
                ratio = result.file_size / max(1, result.compress_size)
                if ratio > self._max_compression_ratio:
                    raise ValueError(
                        f"AOne shard result exceeds compression ratio limit: {archive}"
                    )
                with stream.open(result) as result_stream:
                    value = result_stream.read(self._max_result_bytes + 1)
                if len(value) > self._max_result_bytes:
                    raise ValueError(f"AOne shard result exceeds size limit: {archive}")
                return _load_json_bytes(value, str(archive))
        except zipfile.BadZipFile as error:
            raise ValueError(f"invalid Bazel outputs archive: {archive}") from error

    def collect(self, job: BazelShardJob, shard_id: str) -> Optional[Mapping[str, Any]]:
        expected_job = self._jobs.get(shard_id)
        if expected_job != job or job.shard_id != shard_id:
            raise ValueError(f"unknown AOne shard job: {shard_id}")
        self._wait()
        outputs = self._outputs(job.target)
        relative = f"perf_test_{shard_id}/shard_result.json"
        archive = outputs / "outputs.zip"
        if archive.is_file():
            result = self._read_archive(archive, relative)
            result = result if result is not None else self._missing_result(shard_id)
            return self._with_test_outcome(result)
        result = outputs / relative
        if not result.is_file():
            return self._missing_result(shard_id)
        try:
            if result.stat().st_size > self._max_result_bytes:
                raise ValueError(f"AOne shard result exceeds size limit: {result}")
            with result.open("rb") as stream:
                value = stream.read(self._max_result_bytes + 1)
        except OSError as error:
            raise ValueError(
                f"cannot read AOne shard result {result}: {error}"
            ) from error
        if len(value) > self._max_result_bytes:
            raise ValueError(f"AOne shard result exceeds size limit: {result}")
        return self._with_test_outcome(_load_json_bytes(value, str(result)))

    def _with_test_outcome(
        self, result: Optional[Mapping[str, Any]]
    ) -> Optional[Mapping[str, Any]]:
        if result is None or not self._test_returncode:
            return result
        failed = copy.deepcopy(dict(result))
        failed["status"] = "failed"
        failed["failure"] = {
            "type": "BazelTestFailure",
            "message": f"Bazel remote test exited {self._test_returncode}",
        }
        return failed


def _workspace_root() -> Path:
    configured = os.environ.get("PERF_TEST_WORKSPACE_ROOT") or os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY"
    )
    return Path(configured).resolve() if configured else Path.cwd().resolve()


def create_default_aone_executor(plan: ExecutionPlan):
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):
        raise ValueError("AOne execution requires an AOne resource plan")
    if plan.action == "report":
        from rtp_llm.test.perf_test.prefill.report import generate_prefill_report

        return generate_prefill_report(plan.result_directory)
    if resource.machine_count > 1:
        from rtp_llm.test.perf_test.execution.aone_gang import (
            create_default_gang_executor,
        )

        return create_default_gang_executor(plan)
    workspace_root = _workspace_root()
    integration: Optional[BazelAOneIntegration] = None

    def configure() -> BazelAOneIntegration:
        nonlocal integration
        if integration is None:
            integration = BazelAOneIntegration(
                workspace_root=workspace_root,
                resource=resource.resource,
                gpu_type=resource.gpu_type,
                max_parallel=resource.max_parallel,
                bazel=os.environ.get("PERF_TEST_BAZEL", "bazelisk"),
                build_jobs=_positive_integer(
                    "PERF_TEST_BAZEL_BUILD_JOBS",
                    os.environ.get("PERF_TEST_BAZEL_BUILD_JOBS", os.cpu_count() or 1),
                ),
                startup_args=shlex.split(
                    os.environ.get("PERF_TEST_BAZEL_STARTUP_ARGS", "")
                ),
                common_args=shlex.split(os.environ.get("PERF_TEST_BAZEL_ARGS", "")),
                testlogs_root=os.environ.get("PERF_TEST_BAZEL_TESTLOGS"),
            )
        return integration

    return AOneExecutor(
        workspace_root=workspace_root,
        builder=lambda package: configure().build(package),
        submitter=lambda target, shard_id: configure().submit(target, shard_id),
        result_collector=lambda job, shard_id: configure().collect(job, shard_id),
    ).execute(plan)
