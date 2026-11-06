"""Append and replay prefill case results as newline-delimited JSON."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from rtp_llm.test.perf_test.storage import _validate_artifact_name


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
