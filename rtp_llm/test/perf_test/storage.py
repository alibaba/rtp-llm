"""Atomic JSON storage for mutable perf-test run artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

FaultInjector = Callable[[str, Path, Path], None]
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def atomic_write_json(
    path: Union[str, Path],
    value: Any,
    *,
    fault_injector: Optional[FaultInjector] = None,
) -> None:
    """Commit one complete JSON document with an atomic same-filesystem replace."""
    target = Path(path)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"artifact parent directory does not exist: {target.parent}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if fault_injector is not None:
            fault_injector("before_replace", target, temporary)
        os.replace(temporary, target)
        replaced = True
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if fault_injector is not None:
            fault_injector("after_replace", target, temporary)
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


def _validate_artifact_name(kind: str, name: str) -> str:
    if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
        raise ValueError(f"invalid {kind} name: {name!r}")
    return name


def append_prefill_case(path: Path, case_id: str, result: Mapping[str, Any]) -> None:
    _validate_artifact_name("case", case_id)
    if result.get("case_id") != case_id:
        raise ValueError("case result identity does not match case_id")
    payload = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with Path(path).open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                # Search backwards only as far as the incomplete last record.
                position = end
                while position:
                    start = max(0, position - 8192)
                    stream.seek(start)
                    chunk = stream.read(position - start)
                    index = chunk.rfind(b"\n")
                    if index >= 0:
                        position = start + index + 1
                        break
                    position = start
                stream.truncate(position)
        stream.seek(0, os.SEEK_END)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_prefill_cases(
    path: Path, planned: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    try:
        stream = Path(path).open("rb")
    except FileNotFoundError:
        return cases
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"invalid prefill case record at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"prefill case record at line {line_number} must be an object"
                )
            identifier = value.get("case_id")
            _validate_artifact_name("case", identifier)
            if identifier not in planned:
                raise ValueError(f"unknown prefill case id: {identifier}")
            if value.get("identity") != planned[identifier]:
                raise ValueError(f"prefill case identity mismatch: {identifier}")
            cases[identifier] = value
    return cases


class RunStore:
    """All mutable JSON artifact locations for one initialized run directory."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def write_manifest(
        self, value: Any, *, fault_injector: Optional[FaultInjector] = None
    ) -> None:
        atomic_write_json(
            self.root / "manifest.json", value, fault_injector=fault_injector
        )

    def write_checkpoint(
        self,
        name: str,
        value: Any,
        *,
        fault_injector: Optional[FaultInjector] = None,
    ) -> None:
        name = _validate_artifact_name("checkpoint", name)
        atomic_write_json(
            self.root / "checkpoints" / f"{name}.json",
            value,
            fault_injector=fault_injector,
        )

    def write_case(
        self,
        case_id: str,
        value: Any,
        *,
        fault_injector: Optional[FaultInjector] = None,
    ) -> None:
        case_id = _validate_artifact_name("case", case_id)
        atomic_write_json(
            self.root / "results" / "cases" / f"{case_id}.json",
            value,
            fault_injector=fault_injector,
        )

    def write_prefill_case(self, case_id: str, result: Any) -> None:
        append_prefill_case(self.root / "results" / "cases.jsonl", case_id, result)

    def read_prefill_cases(self, plan: "ExecutionPlan") -> Dict[str, Dict[str, Any]]:
        from rtp_llm.test.perf_test.plan import thaw

        planned = {case.case_id: thaw(case.identity) for case in plan.cases}
        return read_prefill_cases(self.root / "results" / "cases.jsonl", planned)

    def write_final_result(
        self, value: Any, *, fault_injector: Optional[FaultInjector] = None
    ) -> None:
        atomic_write_json(
            self.root / "results" / "result.json",
            value,
            fault_injector=fault_injector,
        )

    def write_summary(
        self, value: Any, *, fault_injector: Optional[FaultInjector] = None
    ) -> None:
        atomic_write_json(
            self.root / "results" / "summary.json",
            value,
            fault_injector=fault_injector,
        )

    def read_committed_cases(self) -> Dict[str, Dict[str, Any]]:
        cases: Dict[str, Dict[str, Any]] = {}
        directory = self.root / "results" / "cases"
        for path in sorted(directory.glob("*.json")):
            case_id = path.stem
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid committed case file {path}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"committed case file must contain an object: {path}")
            if value.get("case_id") != case_id:
                raise ValueError(
                    f"committed case identity does not match filename: {path}"
                )
            cases[case_id] = value
        return cases
