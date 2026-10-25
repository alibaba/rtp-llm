"""Bounded fixture controls, real discovery withdrawal, and sampled health evidence."""

import copy
import json
from functools import partial
from pathlib import Path

import grpc

from adapters.debug_client import DebugUnavailable
from adapters.java_flow import JavaFlowGroup
from adapters.master_diagnostics import MasterDiagnosticsClient
from adapters.mock_control import control_json
from adapters.resource_evidence import flow_evidence
from analysis.checks import CheckResult
from cases.cache_sync_recovery.analysis import CHECK_IDS
from cases.cache_sync_recovery.health import collateral, converged, healthy
from cases.cache_sync_recovery.telemetry import interval_rates, worker_rows
from file_io import write_json
from file_io import write_text as atomic_write
from runtime.observation import SampleBudget, SampleSchedule, poll_samples
from scenario.contracts import StageHandler, StageOutput


def validate(params, plan):
    if set(params) != {"procedure", "observation", "checks", "traffic", "intervention"}:
        raise ValueError("pressure cycle requires shared background and explicit intervention")
    if type(params["intervention"]) is not bool:
        raise ValueError("intervention must be boolean")
    from adapters.java_flow_spec import JavaFlowSpec

    JavaFlowSpec.read(params["traffic"])
    return copy.deepcopy(params)


def withdraw(original, victims):
    data = json.loads(original)
    if not isinstance(data, dict) or any(not isinstance(value, list) for value in data.values()):
        raise ValueError("unsupported discovery file schema")
    removed = [value for values in data.values() for value in values if value in victims]
    if set(removed) != set(victims) or len(removed) != len(victims):
        raise ValueError("discovery withdrawal must remove every victim exactly once")
    return json.dumps({name: [value for value in values if value not in victims] for name, values in data.items()})


def drain_flow(flow, deadline, sample, interval_s, event):
    """Keep observing on the caller thread until stop acknowledgement and exit."""
    schedule = SampleSchedule(deadline.clock(), interval_s, immediate=True)

    def observe():
        if deadline.clock() >= schedule.next:
            sample("drain")
            schedule.advance(deadline.clock())

    flow.stop_sending(deadline, on_poll=observe)
    event("sending_stopped")
    flow.drain(deadline, on_poll=observe)
    sample("drain")
    event("drain_finished")


def _cache_fixture(ops, deadline, engine: dict, operation: str, **options) -> dict:
    result = control_json(ops, "cache_fixture", deadline, dict(engine=engine["name"], operation=operation, **options))
    if result.get("engine") != engine["name"] or result.get("status") != "ok":
        raise ValueError("fixture identity mismatch")
    return result["fixture"]


def _probe_alive(ctx, deadline, entries: list[dict]) -> list[dict]:
    results = []
    for e in entries:
        with grpc.insecure_channel(e["grpc_addr"]) as channel:
            response = ctx.ops.pb2_grpc.RpcServiceStub(channel).GetWorkerStatus(
                ctx.ops.pb2.StatusVersionPB(), timeout=min(2, deadline.remaining())
            )
        results.append(dict(worker=e["http_addr"], alive=response.alive, epoch_s=ctx.wall_clock()))
        if not response.alive:
            raise ValueError("selected Worker is not independently alive")
    return results


class _CacheRecoverySession:
    """One case-owned observation ledger across baseline, pressure and recovery."""

    def __init__(self, ctx, params, deadline):
        self.ctx, self.params, self.deadline = ctx, params, deadline
        self.procedure, self.observation, self.acceptance_limits = (self.params[k] for k in ("procedure", "observation", "checks"))
        self.intervention = self.params["intervention"]
        self.directory = self.ctx.artifact_dir / "cache-stability"
        self.directory.mkdir()
        self.evidence = dict(
            intervention=self.intervention,
            parameters=copy.deepcopy(self.params),
            environment=self.ctx.instance["environment"],
            events=[],
            intensity={},
            frames=[],
            errors=[],
            healthy_violations=[],
            convergence_samples=0,
            cache_progress=None,
            provenance=dict(env_epoch=self.ctx.env_epoch),
            expiry_scope="discovery withdrawal while RPC remains alive; not a health-timeout injection",
        )
        self.engines = control_json(self.ctx.ops, "snapshot?view=topology", self.deadline)["engines"]
        self.prefill = sorted((e for e in self.engines if e["role"] == "prefill"), key=lambda e: int(e["name"].rsplit("-", 1)[1]))
        if (
            len(self.prefill) != self.ctx.env.spec.n_prefill
            or len(self.engines) != self.ctx.env.spec.n_prefill + self.ctx.env.spec.n_decode
        ):
            raise ValueError("Mock topology differs from frozen case")
        self.chosen = [
            self.prefill[i * len(self.prefill) // self.procedure["expired_workers"]] for i in range(self.procedure["expired_workers"])
        ]
        self.victims = {e["http_addr"] for e in self.chosen} if self.intervention else set()
        self.names = {e["http_addr"] for e in self.engines}
        self.evidence["worker_addresses"] = sorted(self.names)
        self.path = Path(self.ctx.env.discovery_file)
        self.original = self.path.read_text()
        self.ctx.add_cleanup("restore_cache_discovery", lambda d: atomic_write(self.path, self.original))
        self.evidence["victims"] = sorted(self.victims)
        self.baseline, self.last_sample = None, None
        self.affected, self.absent, self.rebuilt = set(), set(), set()
        self.maximum_gap = 0
        self.diagnostics = MasterDiagnosticsClient(
            f"http://127.0.0.1:{self.ctx.ops.master_http_port}", timeout_s=min(5, self.deadline.remaining())
        )
        # Check capabilities before cache construction or traffic launch.
        preflight = self.diagnostics.snapshot(include=("workers",))
        worker_rows(preflight)
        self.instance_id = preflight.payload["instance_id"]
        self.evidence["provenance"].update(master_instance_id=self.instance_id, diagnostic_schema=2)
        self.sample_budget = SampleBudget(self.observation["capture"])

        self.fixture = partial(_cache_fixture, self.ctx.ops, self.deadline)
        self.alive = partial(_probe_alive, self.ctx, self.deadline)

    def stats(self):
        return {e["http_addr"]: self.fixture(e, "inspect") for e in self.prefill}

    def event(self, name):
        e = self.ctx.record_event(name)
        self.evidence["events"].append(dict(e, name=name))
        write_json(self.directory / "events.json", self.evidence["events"])
        print("CACHE_EVENT", name, flush=True)

    def sample(self, phase):
        before = self.ctx.clock()
        self.deadline.check()
        self.diagnostics.timeout_s = min(5, self.deadline.remaining())
        capture = self.diagnostics.snapshot(include=("workers",))
        if capture.payload["instance_id"] != self.instance_id:
            raise DebugUnavailable("Master instance changed during observation")
        rows = worker_rows(capture)
        now = self.ctx.clock()
        if self.last_sample is not None:
            self.maximum_gap = max(self.maximum_gap, now - self.last_sample)
        self.last_sample = now
        if self.baseline is not None:
            excluded = self.victims if phase in {"intervention", "post", "drain", "cache_settle"} else set()
            newly = set(collateral(rows, self.baseline, excluded, self.acceptance_limits["max_status_age_ms"])) - self.affected
            self.affected.update(newly)
            if newly:
                proof = dict(phase=phase, epoch_s=self.ctx.wall_clock(), workers=sorted(newly))
                try:
                    proof["direct_alive"] = self.alive([e for e in self.engines if e["http_addr"] == sorted(newly)[0]])
                except Exception as error:
                    proof["probe_error"] = str(error)
                self.evidence["healthy_violations"].append(proof)
            if phase == "intervention":
                self.absent.update(self.victims - rows.keys())
            if phase == "post":
                self.rebuilt.update(a for a in self.victims if a in rows and rows[a]["generation"] != self.baseline[a]["generation"])
        survivors = {a: w for a, w in rows.items() if a not in self.victims}

        def maximum_age(field):
            values = [w[field] for w in survivors.values()]
            return max(values) if values and all(v is not None for v in values) else None

        poll_age, applied_age = maximum_age("status_age_ms"), maximum_age("applied_status_age_ms")
        frame = dict(
            epoch_s=self.ctx.wall_clock(),
            phase=phase,
            snapshot_read_s=self.ctx.clock() - before,
            snapshot_id=capture.payload["snapshot_id"],
            capture_started_at_ms=capture.payload["capture_started_at_ms"],
            capture_finished_at_ms=capture.payload["capture_finished_at_ms"],
            workers=rows
            if phase == "cache_settle"
            else {
                a: {k: w[k] for k in ("role", "generation", "reported_alive", "lifecycle_state", "status_age_ms", "applied_status_age_ms")}
                for a, w in rows.items()
            },
            missing_workers=len((self.baseline or {}).keys() - rows.keys() - self.victims),
            healthy_max_status_age_ms=max(poll_age, applied_age) if poll_age is not None and applied_age is not None else None,
            healthy_max_rpc_age_ms=poll_age,
            healthy_max_applied_age_ms=applied_age,
            affected_workers=len(self.affected),
        )
        encoded = self.sample_budget.append(frame)
        self.evidence["frames"].append(frame)
        with (self.directory / "observations.jsonl").open("a") as f:
            f.write(encoded + "\n")
        return rows

    def hold(self, phase, seconds):
        start = self.ctx.clock()
        before = self.stats()
        self.event(phase + "_started")
        until = start + seconds
        for _ in poll_samples(
            self.deadline, self.observation["sample_s"], lambda: self.sample(phase), until=until, immediate=True, boundary_sample=False
        ):
            pass
        after = self.stats()
        self.evidence["intensity"][phase] = interval_rates(before, after, self.ctx.clock() - start)
        return after

    def prepare_baseline(self):
        from traffic.traffic_source import materialize

        self.traffic = self.params["traffic"]
        self.trace = materialize(
            self.directory / "traffic.jsonl",
            self.traffic["source"],
            "cache-sync-shared-background",
            Path(self.ctx.instance["source_path"]).parent,
            max_requests=int(self.traffic["client"]["playback"]["duration_s"] * self.traffic["client"]["playback"]["qps"] * 1.02),
        )
        initial_until = self.ctx.clock() + 60
        while self.ctx.clock() < initial_until:
            rows = self.sample("initial_health")
            if set(rows) == self.names and all(healthy(w, self.acceptance_limits["max_status_age_ms"]) for w in rows.values()):
                self.baseline = rows
                break
            self.deadline.sleep(self.observation["sample_s"])
        if self.baseline is None:
            raise ValueError("initial healthy Worker population unavailable")
        self.event("cache_seed_started")
        for i, e in enumerate(self.prefill):
            self.fixture(
                e,
                "configure",
                enabled=True,
                worker=i,
                workers=len(self.prefill),
                unique=self.procedure["unique_keys"],
                shared=self.procedure["shared_keys"],
                delta=0,
                interval_ms=self.procedure["churn_interval_ms"],
            )
            self.sample("cache_construction")
        self.event("cache_seed_finished")
        frozen = self.stats()
        baseline_keys = {a: f["reported_keys"] for a, f in frozen.items()}
        self.evidence["cache_baseline_expected"] = dict(keys=baseline_keys, versions={a: f["version"] for a, f in frozen.items()})
        self.baseline = None
        ready = 0
        until = self.ctx.clock() + self.procedure["warmup_timeout_s"]
        while self.ctx.clock() < until:
            rows = self.sample("warmup")
            self.evidence["cache_baseline_last_workers"] = rows
            good = converged(
                rows, self.names, baseline_keys, {a: f["version"] for a, f in frozen.items()}, self.acceptance_limits["max_status_age_ms"]
            )
            ready = ready + 1 if good else 0
            if ready >= self.observation["converged_samples"]:
                self.baseline = rows
                break
            self.deadline.sleep(self.observation["sample_s"])
        if self.baseline is None:
            raise ValueError("shared cache baseline failed to converge")
        self.last_sample = None
        self.maximum_gap = 0
        self.evidence["cache_baseline"] = dict(
            workers=rows,
            expected_keys=baseline_keys,
            expected_global_keys=len(self.prefill) * self.procedure["unique_keys"] + self.procedure["shared_keys"],
            expected_versions={a: f["version"] for a, f in frozen.items()},
        )
        self.event("cache_baseline_ready")

    def run_background(self):
        for e in self.prefill:
            self.fixture(e, "churn", delta=self.procedure["delta_keys"], interval_ms=self.procedure["churn_interval_ms"])
        self.hold("cache_only", self.procedure["cache_only_s"])
        client = self.ctx.services.environment.client_operations(self.traffic["jvm_xms"], self.traffic["jvm_xmx"])
        self.flow = JavaFlowGroup(
            client,
            self.directory / "flow",
            run_id=str(self.ctx.artifact_dir),
            group_id=self.traffic["group_id"],
            phase_id=self.traffic["phase_id"],
            poll_s=self.traffic["poll_s"],
            collection_profile="aggregate",
        )

        def close_flow(d):
            if self.flow.proc is not None:
                self.flow.stop_sending(d)
                self.flow.drain(d)

        self.ctx.register_resource("java_flow", self.flow, cleanup=close_flow, evidence=flow_evidence, request_stop=self.flow.request_stop)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as startup:
            pending = startup.submit(
                self.flow.start, self.trace, self.traffic["client"], self.deadline, target=f"127.0.0.1:{self.ctx.env.master_http_port + 2}"
            )
            while not pending.done():
                self.sample("traffic_startup")
                self.deadline.sleep(self.observation["sample_s"])
            pending.result()
        self.hold("traffic_warmup", self.procedure["traffic_warmup_s"])
        self.hold("measurement", self.procedure["measurement_s"])
        self.event("measurement_finished")
        if self.intervention:
            self.evidence["victim_alive_before"] = self.alive(self.chosen)
            atomic_write(self.path, withdraw(self.original, self.victims))
            self.event("partial_workers_withdrawn")
            self.hold("intervention", self.procedure["withdraw_s"])
            self.evidence["victim_alive_withdrawn"] = self.alive(self.chosen)
            atomic_write(self.path, self.original)
            self.event("partial_workers_restored")
        # Identical post duration and unchanged background; do not stop at first convergence.
        self.hold("post", self.procedure["post_s"])
        self.event("post_finished")
        drain_flow(self.flow, self.deadline, self.sample, self.observation["sample_s"], self.event)
        self.evidence["client"] = self.flow.evidence_snapshot()
        self.event("performance_observation_finished")

    def settle_cache(self):
        for e in self.prefill:
            self.fixture(e, "churn", delta=0, interval_ms=self.procedure["churn_interval_ms"])
        expected = self.stats()
        versions = {a: f["version"] for a, f in expected.items()}
        reported_keys = {a: f["reported_keys"] for a, f in expected.items()}
        until = self.ctx.clock() + 120
        while self.ctx.clock() < until:
            rows = self.sample("cache_settle")
            good = converged(rows, self.names, reported_keys, versions, self.acceptance_limits["max_status_age_ms"])
            self.evidence["convergence_samples"] = self.evidence["convergence_samples"] + 1 if good else 0
            if self.evidence["convergence_samples"] >= self.observation["converged_samples"]:
                break
            self.deadline.sleep(self.observation["sample_s"])
        self.evidence["cache_progress"] = dict(
            expected_versions=versions,
            expected_keys=reported_keys,
            converged=self.evidence["convergence_samples"] >= self.observation["converged_samples"],
            scope="cache versions, index initialization and counts; no key membership proof",
        )
        self.event("cache_progress_finished")
        self.evidence.update(
            affected_workers=sorted(self.affected),
            expired_workers=sorted(self.absent),
            rebuilt_workers=sorted(self.rebuilt),
            max_sample_gap_s=self.maximum_gap,
        )
        self.event("observation_finished")

    def run(self):
        try:
            self.prepare_baseline()
            self.run_background()
            self.settle_cache()
        except Exception as error:
            self.evidence["errors"].append(repr(error))
            self.evidence.update(affected_workers=sorted(self.affected), runtime_incomplete=True)
            # Freeze partial evidence so an interrupted observation cannot hide collateral failure.
            if not self.affected:
                raise
        finally:
            try:
                atomic_write(self.path, self.original)
            finally:
                write_json(self.directory / "evidence.json", self.evidence)
        from cases.cache_sync_recovery.telemetry import gc_events

        logs = list(self.ctx.artifact_dir.glob("environment/*/master_gc.log"))
        self.evidence["gc"] = gc_events(logs[0]) if len(logs) == 1 else []
        self.evidence["gc_available"] = len(logs) == 1
        from artifacts.gate import commit_gate
        from cases.cache_sync_recovery.analysis import analyze

        result, gate_path = commit_gate(self.ctx.artifact_dir, "cache_stability", self.evidence, analyze)
        return StageOutput({}, [CheckResult(**c) for c in result["checks"]], [str(gate_path), str(self.directory / "evidence.json")])


def execute(ctx, params, deadline):
    return _CacheRecoverySession(ctx, params, deadline).run()


HANDLERS = [StageHandler("cache_recovery_cycle", validate, execute, {}, checks=CHECK_IDS)]
