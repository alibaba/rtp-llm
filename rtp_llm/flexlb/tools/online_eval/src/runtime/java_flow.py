"""Phase-level Java traffic control; Python never forwards individual requests."""

import json
import time
import uuid
from pathlib import Path

from runtime.ha import LiveClientEvents
from traffic.traffic_source import sha256_file
from traffic.playback import normalize


class JavaFlowGroup:
    def __init__(
        self,
        client,
        directory,
        *,
        run_id,
        group_id,
        phase_id,
        poll_s,
        max_events=50_000,
        collection_profile="request",
        monitor=None,
    ):
        if not all(isinstance(x, str) and x for x in (run_id, group_id, phase_id)):
            raise ValueError("flow identities must be explicit nonempty strings")
        if poll_s <= 0:
            raise ValueError("flow polling interval must be positive")
        if collection_profile not in {"aggregate", "request", "diagnostic"}:
            raise ValueError("unknown collection profile")
        self.collection_profile = collection_profile
        self.monitor = monitor
        self.monitor_target = "client-" + group_id
        self.client = client
        self.directory = Path(directory)
        self.identity = dict(run_id=run_id, group_id=group_id, phase_id=phase_id)
        self.poll_s = poll_s
        self.proc = None
        self.control = self.directory / "control"
        self.journal = LiveClientEvents(
            self.directory / "client_lifecycle.jsonl", max_events=max_events
        )
        self.stop_command = None
        self.trace_manifest = None

    def start(self, trace, environment, deadline):
        if self.proc is not None or self.directory.exists():
            raise ValueError("flow output directory must be fresh")
        trace = Path(trace).resolve()
        self.trace_manifest = dict(path=str(trace), sha256=sha256_file(trace))
        self.directory.mkdir(parents=True)
        self.control.mkdir()
        env, playback = normalize(environment)
        source_manifest = trace.with_suffix(".manifest.json")
        semantics = (
            json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
        )
        self.trace_manifest.update(semantics)
        self.trace_manifest["playback"] = playback
        env.update(
            TRACE_FILE=str(trace),
            FLOW_CONTROL_DIR=str(self.control),
            FLOW_RUN_ID=self.identity["run_id"],
            FLOW_GROUP_ID=self.identity["group_id"],
            FLOW_PHASE_ID=self.identity["phase_id"],
            LIVE_CLIENT_EVENTS=str(self.collection_profile != "aggregate").lower(),
            COLLECTION_PROFILE=self.collection_profile,
            CLIENT_MONITORING=str(self.monitor is not None).lower(),
        )
        if self.collection_profile != "diagnostic":
            env["SKIP_SERVER_LATENCY"] = "true"
        (self.directory / "flow-input.json").write_text(
            json.dumps(
                dict(**self.identity, trace=self.trace_manifest, environment=env),
                indent=2,
            )
        )
        self.proc, _ = self.client.run_async(
            env, self.directory, self.directory / "client.log"
        )
        if self.monitor is not None:
            target_path = self.directory / "metrics-target.json"
            self._wait(
                lambda state: target_path.exists(),
                deadline,
                "client monitor endpoint absent",
            )
            self.monitor.add_target(
                self.monitor_target, json.loads(target_path.read_text())["url"]
            )
            (self.directory / "metrics-ready").write_text(
                str(int(time.time() * 1000) + 500)
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
        if self.collection_profile == "aggregate":
            state["observed_started"] = state.get("started", 0)
            state["observed_terminal"] = state.get("terminal", 0)
            state["unfinished_ids"] = []
            state["process_returncode"] = (
                None if self.proc is None else self.proc.proc.poll()
            )
            return state
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
        if getattr(self, "monitor", None) is not None:
            self.monitor.end_target(
                self.monitor_target,
                state.get("recorded_epoch_ms", time.time() * 1000) / 1000,
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
        if self.collection_profile == "aggregate":
            complete = (
                state.get("state") == "DRAINED"
                and state.get("process_returncode") == 0
                and state.get("submitted")
                == state.get("started")
                == state.get("terminal")
            )
            return dict(
                producer_kind="java",
                complete=complete,
                errors=[] if complete else ["incomplete flow counters"],
                **self.identity,
                status=state,
                trace=self.trace_manifest,
                records=[],
                issued=[],
                unfinished=[],
            )
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
