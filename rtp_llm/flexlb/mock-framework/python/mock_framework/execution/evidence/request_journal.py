"""Stream Client request records and validate the journal seal."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import ExecutionError

MAX_LINE = 2 * 1024 * 1024
OUTCOMES = {"success", "error", "timeout", "cancelled"}


@dataclass(frozen=True)
class Attempt:
    number: int
    master: str
    started_at_ms: int
    finished_at_ms: int
    grpc_code: str
    error_code: int
    message: str
    response_success: bool | None


@dataclass(frozen=True)
class RequestRecord:
    client: str
    playback: int
    request_id: int
    sent_at_ms: int
    finished_at_ms: int
    ttft_nanos: int | None
    elapsed_nanos: int
    output_tokens: int
    outcome: str
    grpc_code: str
    error_code: int
    message: str
    attempts: list[Attempt]


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate evidence JSONL field: {key}")
        result[key] = value
    return result


def _json_lines(stream, cancelled):
    while line := stream.readline(MAX_LINE + 1):
        if cancelled is not None and cancelled.is_set():
            raise ExecutionError("Client evidence scan cancelled")
        if len(line) > MAX_LINE or not line.endswith(b"\n"):
            raise ValueError("Oversized or truncated evidence JSONL line")
        value = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_fields)
        if not isinstance(value, dict):
            raise ValueError("Expected an evidence JSONL object")
        yield value


HEADER_FIELDS = (
    "version",
    "client",
    "playback",
    "first_request_id",
    "first_index",
    "stride",
    "planned",
    "concurrency",
)
EVENT_NUMBERS = (
    "sequence",
    "request_id",
    "epoch_ms",
    "elapsed_nanos",
    "attempt",
    "error_code",
    "output_tokens",
)
EVENT_TEXT = ("kind", "master", "grpc_code", "message", "outcome")


def _integer(value):
    # JSONL stores 64-bit integers as strings to preserve their precision.
    if isinstance(value, str):
        digits = value.removeprefix("-")
        if digits.isascii() and digits.isdecimal():
            value = int(value)
    if type(value) is not int:
        raise ValueError("Expected an evidence integer")
    return value


def _identity(header):
    if header.keys() - set(HEADER_FIELDS):
        raise ValueError("Unknown evidence header field")
    return tuple(
        header.get(key, "") if key == "client" else _integer(header.get(key, 0))
        for key in HEADER_FIELDS
    )


def _event(row):
    if not isinstance(row, dict) or row.keys() - {
        *EVENT_NUMBERS,
        *EVENT_TEXT,
        "ttft_nanos",
        "response_success",
    }:
        raise ValueError("Invalid evidence event fields")
    event = {key: _integer(row.get(key, 0)) for key in EVENT_NUMBERS}
    if not (0 <= event["attempt"] < 2**32 and 0 <= event["output_tokens"] < 2**32):
        raise ValueError("Invalid evidence attempt or output token count")
    if not -(2**63) <= event["error_code"] < 2**63:
        raise ValueError("Invalid evidence error code")
    for key in EVENT_TEXT:
        value = row.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"Expected evidence text for {key}")
        event[key] = value
    ttft = row.get("ttft_nanos")
    event["ttft_nanos"] = None if ttft is None else _integer(ttft)
    response = row.get("response_success")
    if response is not None and type(response) is not bool:
        raise ValueError("Expected a boolean Schedule response")
    event["response_success"] = response
    return event


def _json_batches(stream, cancelled):
    lines = _json_lines(stream, cancelled)
    header = next(lines, None)
    if header is None:
        return
    for line in lines:
        if "seal" in line:
            if line.keys() != {"seal"} or not isinstance(line["seal"], dict):
                raise ValueError("Invalid evidence JSONL seal")
            yield header, (), line["seal"]
            if next(lines, None) is not None:
                raise ValueError("Trailing evidence after seal")
            return
        yield header, (line,), None


def playback_directories(root, topology, targets=None, playback=None):
    clients = {node["rpc"]: node for node in topology if node["role"] == "client"}
    chosen = list(clients) if targets is None else targets
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - clients.keys():
        raise ExecutionError("Expected nonempty, unique Client addresses")
    if playback is None:
        rounds = root / "components" / "client-0" / "coordinator"
        available = (
            [
                int(path.name)
                for path in rounds.iterdir()
                if path.is_dir() and path.name.isdecimal()
            ]
            if rounds.exists()
            else []
        )
        if not available:
            raise ExecutionError("No Client playback exists")
        playback = max(available)
    return [
        root / "components" / clients[target]["id"] / "playback" / str(playback)
        for target in chosen
    ]


def read_requests(directory: Path, cancelled=None):
    """Yield completed requests with bounded memory; consume through EOF to validate the seal."""
    path = directory / "requests.jsonl"
    active, outcomes = {}, Counter()
    sequence = issued = finished = 0
    sealed = False
    identity = None
    try:
        plan = json.loads((directory / "plan.json").read_text())
        with path.open("rb") as stream:
            for header, events, seal in _json_batches(stream, cancelled):
                if cancelled is not None and cancelled.is_set():
                    raise ExecutionError("Client evidence scan cancelled")
                if sealed:
                    raise ValueError("Trailing evidence after seal")
                current = _identity(header)
                (
                    _,
                    client,
                    playback,
                    first_request_id,
                    first_index,
                    stride,
                    planned,
                    concurrency,
                ) = current
                if identity is None:
                    identity = current
                    expected = (
                        1,
                        directory.parent.parent.name,
                        plan["sequence"],
                        plan["first_request_id"],
                        plan["first_index"],
                        plan["stride"],
                        plan["count"],
                        plan["concurrency"],
                    )
                    if current != expected or not (
                        first_request_id > 0
                        and stride > 0
                        and 0 <= first_index < stride
                        and planned > 0
                        and concurrency > 0
                    ):
                        raise ValueError(
                            "Evidence identity does not match playback plan"
                        )
                for row in events:
                    event = _event(row)
                    sequence += 1
                    if (
                        event["sequence"] != sequence
                        or event["epoch_ms"] <= 0
                        or event["elapsed_nanos"] < 0
                    ):
                        raise ValueError("Invalid event sequence or timestamp")
                    request_id = event["request_id"]
                    if event["kind"] == "ISSUED":
                        expected_id = first_request_id + first_index + issued * stride
                        if (
                            request_id != expected_id
                            or issued >= planned
                            or len(active) >= concurrency
                        ):
                            raise ValueError(
                                "Duplicate, out-of-order or out-of-plan request"
                            )
                        issued += 1
                        active[request_id] = {
                            "sent": event["epoch_ms"],
                            "attempts": [],
                            "pending": None,
                            "first": None,
                            "elapsed": event["elapsed_nanos"],
                        }
                        continue
                    state = active.get(request_id)
                    if state is None or event["elapsed_nanos"] < state["elapsed"]:
                        raise ValueError(
                            "Event without active request or decreasing elapsed time"
                        )
                    state["elapsed"] = event["elapsed_nanos"]
                    if event["kind"] == "SCHEDULE_START":
                        if (
                            state["pending"]
                            or event["attempt"] != len(state["attempts"]) + 1
                            or not event["master"]
                        ):
                            raise ValueError("Invalid Schedule attempt")
                        state["pending"] = event
                    elif event["kind"] == "SCHEDULE_RESULT":
                        start = state["pending"]
                        if (
                            start is None
                            or event["attempt"] != start["attempt"]
                            or event["master"] != start["master"]
                            or not event["grpc_code"]
                        ):
                            raise ValueError("Schedule result without matching attempt")
                        state["attempts"].append(
                            Attempt(
                                event["attempt"],
                                event["master"],
                                start["epoch_ms"],
                                event["epoch_ms"],
                                event["grpc_code"],
                                event["error_code"],
                                event["message"],
                                event["response_success"],
                            )
                        )
                        state["pending"] = None
                    elif event["kind"] == "FIRST_TOKEN":
                        if (
                            state["first"] is not None
                            or state["pending"]
                            or not state["attempts"]
                        ):
                            raise ValueError("Invalid first token event")
                        state["first"] = event["elapsed_nanos"]
                    elif event["kind"] == "TERMINAL":
                        if (
                            state["pending"]
                            or event["outcome"] not in OUTCOMES
                            or not event["grpc_code"]
                        ):
                            raise ValueError("Invalid terminal event")
                        ttft = event["ttft_nanos"]
                        if (ttft is not None) != (state["first"] is not None) or (
                            ttft is not None and not 0 <= ttft <= event["elapsed_nanos"]
                        ):
                            raise ValueError(
                                "Inconsistent first token and terminal result"
                            )
                        if (event["output_tokens"] > 0) != (state["first"] is not None):
                            raise ValueError("Inconsistent output token count")
                        finished += 1
                        outcomes[event["outcome"]] += 1
                        del active[request_id]
                        yield RequestRecord(
                            client,
                            playback,
                            request_id,
                            state["sent"],
                            event["epoch_ms"],
                            ttft,
                            event["elapsed_nanos"],
                            event["output_tokens"],
                            event["outcome"],
                            event["grpc_code"],
                            event["error_code"],
                            event["message"],
                            state["attempts"],
                        )
                    else:
                        raise ValueError("Unknown event kind")
                if seal is not None:
                    if (
                        seal.keys() - {"issued", "finished"}
                        or active
                        or issued != finished
                        or _integer(seal.get("issued", 0)) != issued
                        or _integer(seal.get("finished", 0)) != finished
                    ):
                        raise ValueError("Evidence seal does not reconcile requests")
                    sealed = True
        if not sealed:
            raise ValueError(
                "Missing evidence seal (playback still running, writer failed or JVM exited)"
            )
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (
                not result["done"]
                or result["issued"] != issued
                or result["finished"] != finished
                or result["unsent"] != plan["count"] - issued
                or result["outcomes"] != {key: outcomes[key] for key in OUTCOMES}
            ):
                raise ValueError("Evidence does not reconcile Client result")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f"Invalid Client evidence at {path}: {error}") from error
