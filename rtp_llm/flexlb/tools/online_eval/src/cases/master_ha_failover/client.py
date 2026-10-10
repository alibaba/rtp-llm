"""HA discovery and replay resource; actions own registration and orchestration."""

from __future__ import annotations

import json
import uuid
import time
from pathlib import Path

from cases.master_ha_failover.analysis import row_ts_ms
from cases.master_ha_failover.observation import HaMasterStateSampler
from runtime.java_client import ClientOps
from traffic.contracts import source_priority


class HaReplayClient:
    """Own the HA producer and sampler across start, finish and cleanup actions."""

    def __init__(
        self,
        manager,
        env,
        case_dir: Path,
        name: str,
        targets: list,
        *,
        source: dict,
        duration_s: int = 60,
        timeout_ms: int = 30_000,
        enable_fallback: bool = False,
        replay_speed: float = 2.0,
        max_concurrency: int = 8,
        live_events: bool = False,
        source_dir: Path | None = None,
        max_requests: int | None = None,
        loop: bool = False,
        collection_profile="request",
        sampler_limits, clock=time.monotonic, wall_clock=time.time,
    ):
        self.name = name
        self.targets = list(targets)
        self.out_dir = case_dir / f"{name}_out"
        self.log_file = case_dir / f"{name}.log"
        self.state_sampler = HaMasterStateSampler(env, case_dir / "master_states.jsonl",
            limits=sampler_limits, clock=clock, wall_clock=wall_clock)
        specs_by_target = {
            manager.master_instance_target(env, master_name): spec
            for master_name, spec in env.master_specs.items()
        }
        discovery_file = case_dir / "master-discovery.json"
        discovery_file.write_text(json.dumps({"hosts": [
            {"http": f"{specs_by_target[target].bind_ip}:{specs_by_target[target].http_port}",
             "grpc": target}
            for target in targets
        ]}), encoding="utf-8")
        self._client = ClientOps(manager, "8g", "8g")
        from traffic.traffic_source import materialize

        trace = materialize(case_dir / "ha_trace.jsonl", source, name,
                            source_dir, max_requests=max_requests)
        if not loop:
            first_ts = last_ts = None
            with trace.open(encoding="utf-8") as stream:
                for line in stream:
                    ts = json.loads(line)["ts"]
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
            if first_ts is None or (last_ts - first_ts) / replay_speed < duration_s * 1000:
                raise ValueError("one-pass HA trace ends before the requested duration")
        self.control = case_dir / f"{name}_control"
        self.control.mkdir()
        self.flow_identity = dict(run_id=uuid.uuid4().hex, group_id=name, phase_id="ha")
        self.stop_command = None
        from runtime.load_client import client_environment, collection_environment, bind_environment
        settings = {
            "DURATION_S": str(int(duration_s)),
            "MAX_CONCURRENCY": str(max_concurrency),
            "TIMEOUT_MS": str(int(timeout_ms)),
            "REPLAY_UNIQUE_PREFIX": "false", "FETCH_OUTPUT_STREAM": "true",
            "N_CHANNELS": "8",
            "EVENT_LOOP_THREADS": "8",
            "PRIORITY": str(source_priority(source)),
            "ENABLE_FALLBACK": str(enable_fallback).lower(),
            "playback": dict(mode="true-ts", speed=replay_speed,
                             max_laps=0 if loop else 1, identity="structural-relabel"),
        }
        settings, playback = client_environment(settings)
        runtime = dict(
            FLOW_CONTROL_DIR=str(self.control), FLOW_RUN_ID=self.flow_identity["run_id"],
            FLOW_GROUP_ID=name, FLOW_PHASE_ID="ha", TRACE_FILE=str(trace),
            GRPC_TARGETS=",".join(self.targets), MASTER_DISCOVERY_FILE=str(discovery_file),
            **collection_environment(collection_profile, live_events=live_events),
        )
        if enable_fallback:
            runtime["ENDPOINTS_FILE"] = str(env.endpoint_file)
        overrides = bind_environment(settings, runtime)
        self._overrides = overrides
        from traffic.traffic_source import sha256_file
        trace_info = {}
        source_manifest = trace.with_suffix(".manifest.json")
        if source_manifest.is_file():
            trace_info.update(json.loads(source_manifest.read_text()))
        trace_info.update(path=str(trace), sha256=sha256_file(trace), playback=playback)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "flow-input.json").write_text(json.dumps(
            dict(**self.flow_identity, trace=trace_info, environment=overrides), indent=2))
        self.proc = None
        self.finished = False

    def stop_sending(self):
        from runtime.flow_control import stop_sending
        self.stop_command = stop_sending(self.control, self.flow_identity, self.stop_command)

    def validate_drain(self, rows):
        from runtime.flow_control import read_status, validate_drain
        validate_drain(read_status(self.control, self.flow_identity), self.flow_identity,
                       terminal_count=len(rows), command_id=self.stop_command)

    def start(self) -> None:
        self.state_sampler.start()
        try:
            self.proc, self.out_dir = self._client.run_async(
                self._overrides, self.out_dir, self.log_file, label=self.name
            )
        except Exception as start_error:
            try:
                self.state_sampler.stop()
            except Exception as cleanup_error:
                raise start_error from cleanup_error
            raise

    def evidence_snapshot(self):
        """Persist partial rows even when a failed prerequisite skips finish."""
        path = self.out_dir / "client_events.jsonl"
        rows, errors = [], []
        if path.is_file():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("request event is not an object")
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    errors.append(f"line {number}: {exc}")
        else:
            errors.append("missing client_events.jsonl")
        # Java writes the final event file only on natural exit. A failed
        # checkpoint can terminate the producer earlier; keep its live journal
        # instead of discarding all requests observed before the failure.
        lifecycle = self.out_dir / "client_lifecycle.jsonl"
        if not path.is_file() and lifecycle.is_file():
            from runtime.client_journal import LiveClientEvents

            journal = LiveClientEvents(lifecycle)
            try:
                journal.read()
                if journal.pending:
                    errors.append("incomplete live client journal tail")
            except (ValueError, TypeError) as exc:
                errors.append(f"live client journal: {exc}")
            rows = [
                journal.terminal.get(rid, issued)
                for rid, issued in journal.issued.items()
            ]
            path = lifecycle
        if not self.finished:
            errors.append("HA client finish was not validated")
        return dict(
            records=rows,
            complete=self.finished and bool(rows) and not errors,
            errors=errors,
            path=str(path),
        )

    def cleanup(self, deadline):
        from runtime.cleanup import cleanup_all

        from runtime.cleanup import stop_process

        sampler = getattr(self, "state_sampler", None)
        operations = [("HA client", lambda: stop_process(self.proc, deadline))]
        if sampler is not None:
            operations.append(("HA state sampler", sampler.stop))
        cleanup_all(operations)

    def finish(self, deadline, *, stop_sending=False):
        if self.proc is None:
            raise RuntimeError("HA client was not started")
        if stop_sending:
            self.stop_sending()
        rc = self.proc.proc.wait(timeout=deadline.remaining())
        sampler = getattr(self, "state_sampler", None)
        if sampler is not None:
            sampler.stop()
        if rc != 0:
            raise RuntimeError(f"HA client exit code {rc}")
        path = self.out_dir / "client_events.jsonl"
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not rows:
            raise ValueError("HA client produced no request evidence")
        for row in rows:
            if not isinstance(row, dict) or not {
                "rid",
                "route_path",
                "master_target",
                "failover",
                "error_kind",
                "status",
            } <= set(row):
                raise ValueError("HA client evidence lacks required route fields")
            if (
                row["route_path"] not in {"master", "fallback", "failed"}
                or type(row["failover"]) is not bool
            ):
                raise ValueError("invalid HA route evidence")
            row_ts_ms(row)
        if stop_sending:
            self.validate_drain(rows)
        self.finished = True
        return rows, path


