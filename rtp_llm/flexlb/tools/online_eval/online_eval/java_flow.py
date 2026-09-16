"""Phase-level Java traffic control; Python never forwards individual requests."""

import hashlib
import json
import time
import uuid
from pathlib import Path

from flexlb_test_framework.ha import LiveClientEvents


class JavaFlowGroup:
    def __init__(self, client, directory, *, run_id, group_id, phase_id, poll_s):
        if not all(isinstance(x, str) and x for x in (run_id, group_id, phase_id)):
            raise ValueError("flow identities must be explicit nonempty strings")
        if poll_s <= 0:
            raise ValueError("flow polling interval must be positive")
        self.client = client
        self.directory = Path(directory)
        self.identity = dict(run_id=run_id, group_id=group_id, phase_id=phase_id)
        self.poll_s = poll_s
        self.proc = None
        self.control = self.directory / "control"
        self.journal = LiveClientEvents(self.directory / "client_lifecycle.jsonl")
        self.stop_command = None
        self.trace_manifest = None

    def start(self, trace, environment, deadline):
        if self.proc is not None or self.directory.exists():
            raise ValueError("flow output directory must be fresh")
        trace = Path(trace).resolve()
        self.trace_manifest = dict(
            path=str(trace), sha256=hashlib.sha256(trace.read_bytes()).hexdigest()
        )
        self.directory.mkdir(parents=True)
        self.control.mkdir()
        env = dict(environment)
        env.update(
            TRACE_FILE=str(trace),
            FLOW_CONTROL_DIR=str(self.control),
            FLOW_RUN_ID=self.identity["run_id"],
            FLOW_GROUP_ID=self.identity["group_id"],
            FLOW_PHASE_ID=self.identity["phase_id"],
            LIVE_CLIENT_EVENTS="true",
        )
        (self.directory / "flow-input.json").write_text(
            json.dumps(
                dict(**self.identity, trace=self.trace_manifest, environment=env),
                indent=2,
            )
        )
        self.proc, _ = self.client.run_async(
            env, self.directory, self.directory / "client.log"
        )
        return self._wait(
            lambda s: s.get("state")
            in {"SENDING", "DRAINING", "DRAINED", "INCOMPLETE"},
            deadline,
            "flow did not acknowledge startup",
        )

    def status(self):
        path = self.control / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        if state and any(state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("flow control identity mismatch")
        self.journal.read()
        state["observed_started"] = len(self.journal.issued)
        state["observed_terminal"] = len(self.journal.terminal)
        state["unfinished_ids"] = sorted(
            self.journal.issued.keys() - self.journal.terminal.keys()
        )
        state["process_returncode"] = (
            None if self.proc is None else self.proc.proc.poll()
        )
        return state

    def _wait(self, predicate, deadline, message):
        while True:
            state = self.status()
            if predicate(state):
                return state
            if state["process_returncode"] is not None:
                raise RuntimeError(message + ": client exited " + str(state))
            remaining = deadline.remaining()
            if remaining <= 0:
                raise TimeoutError(message + ": " + str(state))
            time.sleep(min(self.poll_s, remaining))

    def stop_sending(self, deadline):
        state = self.status()
        if state.get("state") in {"DRAINING", "DRAINED", "INCOMPLETE"}:
            return state
        if self.stop_command is None:
            self.stop_command = uuid.uuid4().hex
            command = dict(
                run_id=self.identity["run_id"],
                group_id=self.identity["group_id"],
                operation="stop_sending",
                command_id=self.stop_command,
            )
            temporary = self.control / "stop.json.tmp"
            temporary.write_text(json.dumps(command))
            temporary.replace(self.control / "stop.json")
        return self._wait(
            lambda s: s.get("state") in {"DRAINING", "DRAINED", "INCOMPLETE"},
            deadline,
            "flow did not stop accepting new trace records",
        )

    def drain(self, deadline):
        state = self._wait(
            lambda s: s.get("state") in {"DRAINED", "INCOMPLETE"}
            and s["process_returncode"] is not None,
            deadline,
            "flow did not drain and exit",
        )
        if state["state"] != "DRAINED" or state["process_returncode"] != 0:
            raise RuntimeError("flow drain incomplete: " + str(state))
        if state["submitted"] != state["observed_terminal"] or state["unfinished_ids"]:
            raise RuntimeError(
                "flow drain journal does not account for every submission"
            )
        return state

    def checkpoint(self, label, deadline, predicate):
        state = self._wait(predicate, deadline, "flow checkpoint failed: " + label)
        record = dict(label=label, epoch_s=time.time(), **self.identity, state=state)
        with (self.directory / "checkpoints.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")
        return record

    def evidence_snapshot(self):
        errors = []
        try:
            state = self.status()
        except Exception as exc:
            state = {}
            errors.append(str(exc))
        complete = (
            state.get("state") == "DRAINED"
            and state.get("process_returncode") == 0
            and state.get("submitted") == len(self.journal.terminal)
            and len(self.journal.issued) == len(self.journal.terminal)
        )
        if not complete:
            errors.append(
                "flow has not proven complete submission and terminal accounting"
            )
        return dict(
            producer_kind="java",
            complete=complete,
            errors=errors,
            **self.identity,
            status=state,
            trace=self.trace_manifest,
            records=[
                self.journal.terminal.get(rid, row)
                for rid, row in self.journal.issued.items()
            ],
            issued=list(self.journal.issued.values()),
            unfinished=[
                r
                for rid, r in self.journal.issued.items()
                if rid not in self.journal.terminal
            ],
        )
