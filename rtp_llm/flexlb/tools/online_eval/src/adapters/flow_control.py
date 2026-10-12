"""Java producer control protocol, independent of discovery and traffic source."""

import json
import uuid
from pathlib import Path


def read_status(directory, identity):
    path = Path(directory) / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(state, dict) or state and any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("flow control identity mismatch")
    return state


def stop_sending(directory, identity, command_id):
    if command_id is not None:
        return command_id
    command_id = uuid.uuid4().hex
    command = dict(run_id=identity["run_id"], group_id=identity["group_id"],
                   operation="stop_sending", command_id=command_id)
    temporary = Path(directory) / "stop.json.tmp"
    temporary.write_text(json.dumps(command))
    temporary.replace(Path(directory) / "stop.json")
    return command_id


def validate_drain(state, identity, *, terminal_count, command_id):
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("flow drain identity mismatch")
    if (state.get("state") != "DRAINED" or type(state.get("submitted")) is not int
            or type(state.get("terminal")) is not int or terminal_count < 0
            or state["submitted"] != state["terminal"] or state["terminal"] != terminal_count):
        raise ValueError("flow did not retain every submitted terminal result")
    if command_id is not None and state.get("applied_command_id") != command_id:
        raise ValueError("flow ended before accepting the stop command")
