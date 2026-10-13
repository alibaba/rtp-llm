"""Phase-level Java traffic control; Python never forwards individual requests."""

import json
import time
from pathlib import Path

from adapters.flow_control import FlowControlProtocol, FlowStatus, DrainResult

from adapters.client_journal import LiveClientEvents
from adapters.playback_evidence import PlaybackEvidence
from file_digest import sha256_file
from adapters.load_client import client_environment, collection_environment, bind_environment

JAVA_FLOW_INPUT_FIELDS = frozenset({"group_id", "phase_id", "poll_s", "source", "client", "jvm_xms", "jvm_xmx"})

RUN_LOCAL_ENVIRONMENT = frozenset({"TRACE_FILE", "FLOW_CONTROL_DIR", "FLOW_RUN_ID",
    "GRPC_TARGET", "GRPC_TARGETS", "MASTER_DISCOVERY_FILE", "OUTPUT_DIR"})


def evidence_environment(environment):
    """Launch producer owns the omission of run-local paths, targets and identities."""
    return {key: value for key, value in environment.items() if key not in RUN_LOCAL_ENVIRONMENT}


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
        self.monitor = monitor if monitor is None or monitor.collects("client") else None
        self.monitor_target = "client-" + group_id
        self.client = client
        self.directory = Path(directory)
        self.identity = dict(run_id=run_id, group_id=group_id, phase_id=phase_id)
        self.poll_s = poll_s
        self.proc = None
        self.control = self.directory / "control"
        self.control_protocol = FlowControlProtocol(self.control, self.identity)
        self.journal = LiveClientEvents(
            self.directory / "client_lifecycle.jsonl", max_events=max_events
        )
        self.stop_command = None
        self.trace_manifest = None
        self._drain_result = None

    def start(self, trace, environment, deadline, *, target):
        if self.proc is not None or self.directory.exists():
            raise ValueError("flow output directory must be fresh")
        deadline.check()
        env, playback = client_environment(environment)
        trace = Path(trace).resolve()
        self.trace_manifest = dict(path=str(trace), sha256=sha256_file(trace))
        self.directory.mkdir(parents=True)
        self.control.mkdir()
        source_manifest = trace.with_suffix(".manifest.json")
        semantics = (
            json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
        )
        if 'sha256' in semantics and semantics['sha256'] != self.trace_manifest['sha256']:
            raise ValueError('trace manifest SHA256 differs from materialized plan')
        self.trace_manifest.update(semantics)
        self.trace_manifest["playback"] = playback
        env = bind_environment(env, dict(
            TRACE_FILE=str(trace), GRPC_TARGET=target,
            FLOW_CONTROL_DIR=str(self.control),
            FLOW_RUN_ID=self.identity["run_id"],
            FLOW_GROUP_ID=self.identity["group_id"],
            FLOW_PHASE_ID=self.identity["phase_id"],
            **collection_environment(self.collection_profile, monitoring=self.monitor is not None),
        ))
        (self.directory / "flow-input.json").write_text(
            json.dumps(
                dict(**self.identity, trace=self.trace_manifest, environment=env),
                indent=2,
            )
        )
        deadline.check()
        self.proc, _ = self.client.run_async(
            env, self.directory, self.directory / "client.log"
        )
        if self.monitor is not None:
            target_path = self.directory / "metrics-target.json"
            self._wait_control(
                lambda observation: target_path.exists(),
                deadline,
                "client monitor endpoint absent",
            )
            self.monitor.add_client_target(
                self.monitor_target, json.loads(target_path.read_text())["url"], deadline=deadline
            )
            (self.directory / "metrics-ready").write_text(
                str(int(time.time() * 1000) + 500)
            )
        from artifacts.startup import startup_phase
        from adapters.flow_control import wait_flow_started
        with startup_phase(self.directory, 'client/sending', deadline):
            started = wait_flow_started(self.control, self.identity, self.proc, deadline)
        snapshot = PlaybackEvidence.read(self.directory / 'playback.json',
                                         self.trace_manifest['sha256'], env)
        self.trace_manifest['java_playback'] = snapshot.to_dict()
        (self.directory / 'flow-input.json').write_text(json.dumps(
            dict(**self.identity, trace=self.trace_manifest, environment=env), indent=2))
        return started

    def observe_control(self):
        return self.control_protocol.observe(self.proc)

    def control_status(self):
        """Serialized control observation for experiment evidence consumers."""
        observation = self.observe_control()
        state = (dict(self.identity, **observation.status.encode())
                 if observation.status is not None else {})
        state["process_returncode"] = observation.process_returncode
        return state

    def status(self):
        state = self.control_status()
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

    def _wait_evidence(self, predicate, deadline, message):
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

    def _wait_control(self, predicate, deadline, message):
        while True:
            observation = self.observe_control()
            if observation.status is not None and predicate(observation):
                return observation
            if observation.process_returncode is not None:
                raise RuntimeError(message + ": client exited before final control state")
            deadline.check()
            time.sleep(min(self.poll_s, deadline.remaining()))

    def stop_sending(self, deadline):
        if self._drain_result is not None:
            return self._drain_result.encode(self.identity)
        observation = self.observe_control()
        if observation.status is None or not observation.status.state.sending_finished:
            from adapters.flow_control import stop_sending
            self.stop_command = stop_sending(self.control, self.identity, self.stop_command)
            observation = self._wait_control(
                lambda o: o.status.state.sending_finished,
                deadline, "flow did not stop accepting new trace records",
            )
        return dict(self.identity, **observation.status.encode(),
                    process_returncode=observation.process_returncode)

    def drain(self, deadline):
        if self._drain_result is not None:
            return self._drain_result.encode(self.identity)
        self._wait_control(
            lambda o: o.status.state.terminal and o.process_returncode is not None,
            deadline, "flow did not drain and exit",
        )
        # Read evidence once after the producer has exited; failures remain retryable.
        state = self.status()
        result = DrainResult(
            FlowStatus.decode(state), state["process_returncode"],
            state["observed_started"], state["observed_terminal"],
            tuple(state["unfinished_ids"]),
        )
        result.validate(self.stop_command)
        if self.monitor is not None:
            self.monitor.end_target(
                self.monitor_target, result.status.recorded_epoch_ms / 1000,
            )
        self._drain_result = result
        return self._drain_result.encode(self.identity)

    def checkpoint(self, label, deadline, predicate):
        state = self._wait_evidence(predicate, deadline, "flow checkpoint failed: " + label)
        record = dict(label=label, epoch_s=time.time(), **self.identity, state=state)
        with (self.directory / "checkpoints.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")
        return record

    def evidence_snapshot(self):
        from adapters.client_journal import request_timing
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
        timing = request_timing(self.journal.issued.values())
        errors.extend(timing["errors"])
        complete = complete and not errors
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
            traffic_start=timing["traffic_start"],
            unfinished=[
                r
                for rid, r in self.journal.issued.items()
                if rid not in self.journal.terminal
            ],
        )
