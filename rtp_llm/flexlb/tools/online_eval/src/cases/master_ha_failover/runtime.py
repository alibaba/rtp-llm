"""HA traffic and observation helpers used by the case executor."""

from __future__ import annotations

import json
import math
import threading
import uuid
import time
from pathlib import Path
from typing import Optional

from cases.master_ha_failover.analysis import row_ts_ms
from runtime.java_client import ClientOps
from runtime.network import http_get_json, master_url
from runtime.observation import ObservationClock, SampleBudget
from traffic.contracts import source_priority

HA_TRACE_PRIORITY = 50
HA_TRACE_ROWS = 20
HA_TRACE_SPACING_MS = 100
HA_TRACE_IL = 16
HA_TRACE_OL = 4


def master_state_fields(data):
    """Project a successful response; absent ledger fields are contract errors."""
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("HA state requires finite nonnegative ledger values")
        return value

    def total(endpoints, field):
        if not isinstance(endpoints, list):
            raise ValueError("HA state endpoint ledger must be a list")
        if any(not isinstance(endpoint, dict) or field not in endpoint for endpoint in endpoints):
            raise ValueError("HA state endpoint lacks " + field)
        return sum(number(endpoint[field]) for endpoint in endpoints)

    required = {"scheduler_inflight", "prefill_endpoints", "decode_endpoints"}
    if not isinstance(data, dict) or not required <= set(data):
        raise ValueError("HA state response lacks required ledger fields")
    return dict(scheduler_inflight=number(data["scheduler_inflight"]),
                prefill_inflight_requests=total(data["prefill_endpoints"], "inflight_requests"),
                decode_master_queued=total(data["decode_endpoints"], "master_queued"),
                decode_confirmed_running=total(data["decode_endpoints"], "confirmed_running"))


class HaMasterStateSampler:
    """Record both Masters' HTTP inflight state during the traffic window."""

    def __init__(self, env, path: Path, interval_s: float = 1.0, *,
                 limits=None, clock=time.monotonic, wall_clock=time.time):
        self.path = path
        self.urls = {
            name: master_url(spec.bind_ip, spec.http_port, "inflight")
            for name, spec in env.master_specs.items()
        }
        self.clock, self.wall_clock = clock, wall_clock
        self.budget = SampleBudget(limits if limits is not None else
            dict(max_samples=10000, max_bytes=67108864))
        self.error = None
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ha-master-state", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise TimeoutError("HA Master state sampler did not stop")
            if self.error is not None:
                raise RuntimeError("HA Master state sampling failed: " + self.error)

    def _run(self):
        anchor = ObservationClock(self.wall_clock(), self.clock())
        try:
            with self.path.open("w", encoding="utf-8") as stream:
                while not self._stop.is_set():
                    started = self.clock()
                    for name, url in self.urls.items():
                        if self._stop.is_set():
                            break
                        data = http_get_json(url, timeout=min(0.4, self.interval_s))
                        elapsed = self.clock() - anchor.origin_monotonic_s
                        row = dict(epoch_s=anchor.origin_epoch_s + elapsed, elapsed_s=elapsed,
                                   monotonic_s=self.clock(), master=name, http_up=int(data is not None))
                        if data is not None:
                            try:
                                row.update(master_state_fields(data))
                            except ValueError as exc:
                                row["state_error"] = str(exc)
                        stream.write(self.budget.append(row) + "\n")
                    stream.flush()
                    self._stop.wait(max(0, self.interval_s - (self.clock() - started)))
        except Exception as exc:
            self.error = str(exc)
            self._stop.set()


def write_ha_trace(case_dir: Path) -> Path:
    path = case_dir / "ha_trace.jsonl"
    lines = []
    for i in range(HA_TRACE_ROWS):
        lines.append(
            json.dumps(
                {
                    "ts": i * HA_TRACE_SPACING_MS,
                    "il": HA_TRACE_IL,
                    "ol": HA_TRACE_OL,
                    "bh": [i * 1_000_003 + 7],
                    "priority": HA_TRACE_PRIORITY,
                },
                separators=(",", ":"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class HaTrafficRunner:
    """Background JavaLoadClient using a refreshable Master candidate file.

    The client probes /master/info and chooses the healthy leader, then
    retries a healthy backup only after a Schedule connection failure.

    Phase bookkeeping: the runner stamps wall-clock epoch seconds at
    ``mark()`` call sites; row windows are then sliced offline by
    send_start_epoch_ms (falling back to wall_clock_ts) — the assertions
    compare pre/post-injection WINDOWS. Controlled stop ends submission and
    drains already submitted requests; final row accounting is validated.
    """

    def __init__(
        self,
        manager,
        env,
        case_dir: Path,
        name: str,
        targets: list,
        *,
        duration_s: int = 60,
        timeout_ms: int = 30_000,
        enable_fallback: bool = False,
        replay_speed: float = 2.0,
        max_concurrency: int = 8,
        live_events: bool = False,
        source: dict | None = None,
        source_dir: Path | None = None,
        max_requests: int | None = None,
        loop: bool = False,
        collection_profile="request",
        sampler_limits=None, clock=time.monotonic, wall_clock=time.time,
    ):
        self.manager = manager
        self.env = env
        self.name = name
        self.targets = list(targets)
        self.loop = loop
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
        heap = "8g" if source is not None else "1g"
        self._client = ClientOps(manager, heap, heap)
        if source is None:
            trace = write_ha_trace(case_dir)
        else:
            from traffic.traffic_source import materialize

            trace = materialize(
                case_dir / "ha_trace.jsonl",
                source,
                name,
                source_dir,
                max_requests=max_requests,
            )
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
            "N_CHANNELS": "8" if source is not None else "2",
            "EVENT_LOOP_THREADS": "8" if source is not None else "4",
            "PRIORITY": str(source_priority(source)) if source is not None else str(HA_TRACE_PRIORITY),
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

    def stop_sending(self):
        """Ask the Java producer to drain naturally, without cancelling requests."""
        if self.stop_command is None:
            self.stop_command = uuid.uuid4().hex
            command = dict(run_id=self.flow_identity["run_id"],
                           group_id=self.flow_identity["group_id"],
                           operation="stop_sending", command_id=self.stop_command)
            temporary = self.control / "stop.json.tmp"
            temporary.write_text(json.dumps(command))
            temporary.replace(self.control / "stop.json")

    def validate_drain(self, rows):
        state = json.loads((self.control / "status.json").read_text())
        if any(state.get(key) != value for key, value in self.flow_identity.items()):
            raise ValueError("HA flow drain identity mismatch")
        if (state.get("state") != "DRAINED"
                or state.get("submitted") != state.get("terminal")
                or state.get("terminal") != len(rows)):
            raise ValueError("HA flow did not retain every submitted terminal result")
        if self.stop_command and state.get("applied_command_id") != self.stop_command:
            raise ValueError("HA flow ended before accepting the stop command")

    def start(self) -> None:
        self.state_sampler.start()
        try:
            self.proc, self.out_dir = self._client.run_async(
                self._overrides, self.out_dir, self.log_file, label=self.name
            )
        except Exception:
            self.state_sampler.stop()
            raise

    @staticmethod
    def now() -> float:
        return time.time()

    def rows(self) -> list:
        path = self.out_dir / "client_events.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


def rows_between(rows: list, lo_s: Optional[float], hi_s: Optional[float]) -> list:
    """Rows whose send timestamp falls in [lo_s, hi_s) (epoch seconds)."""
    out = []
    for row in rows:
        ts = row_ts_ms(row)
        if ts is None:
            continue
        if lo_s is not None and ts < lo_s * 1000.0:
            continue
        if hi_s is not None and ts >= hi_s * 1000.0:
            continue
        out.append(row)
    return out
