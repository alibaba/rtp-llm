"""Java producer control protocol, independent of discovery and traffic source."""

import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class FlowState(str, Enum):
    STARTING = "STARTING"
    SENDING = "SENDING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"
    INCOMPLETE = "INCOMPLETE"

    @property
    def sending_finished(self):
        return self in (FlowState.DRAINING, FlowState.DRAINED, FlowState.INCOMPLETE)

    @property
    def terminal(self):
        return self in (FlowState.DRAINED, FlowState.INCOMPLETE)


class SendingEndReason(str, Enum):
    NATURAL = "natural"
    STOP_COMMAND = "stop_command"


@dataclass(frozen=True)
class FlowStatus:
    state: FlowState
    submitted: int
    started: int
    terminal: int
    outstanding: int
    sending_end_reason: Optional[SendingEndReason] = None
    applied_command_id: Optional[str] = None
    revision: int = 0
    recorded_epoch_ms: int = 0

    @classmethod
    def decode(cls, row):
        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("unsupported flow control schema")
        try:
            result = cls(
                state=FlowState(row["state"]),
                submitted=row["submitted"], started=row["started"],
                terminal=row["terminal"], outstanding=row["outstanding"],
                sending_end_reason=(SendingEndReason(row["sending_end_reason"])
                                    if "sending_end_reason" in row else None),
                applied_command_id=row.get("applied_command_id"),
                revision=row["revision"], recorded_epoch_ms=row["recorded_epoch_ms"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid flow control status") from exc
        for value in (result.submitted, result.started, result.terminal,
                      result.outstanding, result.revision, result.recorded_epoch_ms):
            if type(value) is not int or value < 0:
                raise ValueError("invalid flow control counter")
        if result.terminal > result.submitted or result.started > result.submitted:
            raise ValueError("invalid flow control accounting")
        if result.outstanding != result.submitted - result.terminal:
            raise ValueError("invalid flow outstanding count")
        if result.state.sending_finished:
            if result.sending_end_reason is None:
                raise ValueError("flow sending boundary lacks end reason")
            if result.sending_end_reason == SendingEndReason.STOP_COMMAND:
                if not isinstance(result.applied_command_id, str) or not result.applied_command_id:
                    raise ValueError("flow stop boundary lacks command acknowledgement")
            elif result.applied_command_id is not None:
                raise ValueError("natural boundary cannot acknowledge a command")
        elif result.sending_end_reason is not None or result.applied_command_id is not None:
            raise ValueError("flow acknowledges stop before sending boundary")
        if result.state == FlowState.DRAINED and (result.outstanding or result.started != result.submitted):
            raise ValueError("drained flow retains outstanding requests")
        return result

    def validate_completion(self, terminal_count, command_id):
        if (self.state != FlowState.DRAINED or type(terminal_count) is not int
                or terminal_count < 0 or self.submitted != self.terminal
                or self.terminal != terminal_count):
            raise ValueError("flow did not retain every submitted terminal result")
        if command_id is not None and self.applied_command_id != command_id:
            if self.sending_end_reason != SendingEndReason.NATURAL or self.applied_command_id is not None:
                raise ValueError("flow ended before accepting the stop command")

    def encode(self):
        row = asdict(self)
        row["schema_version"] = 1
        row["state"] = self.state.value
        if self.sending_end_reason is None:
            del row["sending_end_reason"]
        else:
            row["sending_end_reason"] = self.sending_end_reason.value
        if self.applied_command_id is None:
            del row["applied_command_id"]
        return row


@dataclass(frozen=True)
class FlowObservation:
    status: Optional[FlowStatus]
    process_returncode: Optional[int]


@dataclass(frozen=True)
class DrainResult:
    status: FlowStatus
    process_returncode: int
    observed_started: int
    observed_terminal: int
    unfinished_ids: tuple

    def validate(self, command_id):
        if self.process_returncode != 0 or self.status.state != FlowState.DRAINED:
            raise RuntimeError("flow drain incomplete")
        if (self.observed_started != self.status.submitted
                or self.observed_terminal != self.status.terminal or self.unfinished_ids):
            raise RuntimeError("flow drain journal does not account for every submission")
        self.status.validate_completion(self.observed_terminal, command_id)

    def encode(self, identity):
        return dict(identity, **self.status.encode(),
                    process_returncode=self.process_returncode,
                    observed_started=self.observed_started,
                    observed_terminal=self.observed_terminal,
                    unfinished_ids=list(self.unfinished_ids))


class FlowControlProtocol:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity

    def observe(self, process):
        # Poll BEFORE reading: an observed exit guarantees the final file is stable.
        returncode = None if process is None else process.proc.poll()
        row = read_status(self.directory, self.identity)
        status = FlowStatus.decode(row) if row else None
        return FlowObservation(status, returncode)


def read_status(directory, identity):
    path = Path(directory) / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(state, dict) or state and any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("flow control identity mismatch")
    return state


@dataclass(frozen=True)
class StopCommand:
    run_id: str
    group_id: str
    command_id: str
    operation: str = "stop_sending"

    def write(self, directory):
        temporary = Path(directory) / "stop.json.tmp"
        temporary.write_text(json.dumps(asdict(self)))
        temporary.replace(Path(directory) / "stop.json")


def stop_sending(directory, identity, command_id):
    if command_id is not None:
        return command_id
    command_id = uuid.uuid4().hex
    StopCommand(identity["run_id"], identity["group_id"], command_id).write(directory)
    return command_id


def validate_drain(state, identity, *, terminal_count, command_id):
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("flow drain identity mismatch")
    FlowStatus.decode(state).validate_completion(terminal_count, command_id)


def wait_flow_started(control, identity, process, deadline):
    from runtime.startup import wait_ready
    def acknowledged():
        state = read_status(control, identity)
        if not process.alive():
            if state.get('state') != 'DRAINED' or process.proc.poll() != 0:
                raise RuntimeError('client exited before sending acknowledgement\n' + process.tail_log())
        return state if state.get('state') in {'SENDING', 'DRAINING', 'DRAINED', 'INCOMPLETE'} else None
    return wait_ready(deadline, 'client/sending acknowledgement', acknowledged,
                      interval_s=0.1)
