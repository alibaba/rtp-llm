"""Bounded fixture controls, real discovery withdrawal, and sampled health evidence."""

import copy
import json
from functools import partial
from pathlib import Path

import grpc

from adapters.debug_client import DebugUnavailable
from adapters.master_diagnostics import MasterDiagnosticsClient
from adapters.mock_control import control_json
from analysis.checks import CheckResult
from cases.cache_sync_recovery.analysis import CHECK_IDS
from cases.cache_sync_recovery.health import collateral, converged, healthy
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


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".cache-recovery.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def execute(ctx, params, deadline):
    from adapters.java_flow import JavaFlowGroup
    from adapters.resource_evidence import flow_evidence
    from cases.cache_sync_recovery.telemetry import interval_rates, worker_rows
    from traffic.traffic_source import materialize

    procedure, observation, acceptance_limits = (params[k] for k in ("procedure", "observation", "checks"))
    intervention = params["intervention"]
    directory = ctx.artifact_dir / "cache-stability"
    directory.mkdir()
    evidence = dict(
        intervention=intervention,
        parameters=copy.deepcopy(params),
        environment=ctx.instance["environment"],
        events=[],
        intensity={},
        frames=[],
        errors=[],
        healthy_violations=[],
        convergence_samples=0,
        cache_progress=None,
        provenance=dict(env_epoch=ctx.env_epoch),
        expiry_scope="discovery withdrawal while RPC remains alive; not a health-timeout injection",
    )
    engines = control_json(ctx.ops, "snapshot?view=topology", deadline)["engines"]
    prefill = sorted((e for e in engines if e["role"] == "prefill"), key=lambda e: int(e["name"].rsplit("-", 1)[1]))
    if len(prefill) != ctx.env.spec.n_prefill or len(engines) != ctx.env.spec.n_prefill + ctx.env.spec.n_decode:
        raise ValueError("Mock topology differs from frozen case")
    chosen = [prefill[i * len(prefill) // procedure["expired_workers"]] for i in range(procedure["expired_workers"])]
    victims = {e["http_addr"] for e in chosen} if intervention else set()
    names = {e["http_addr"] for e in engines}
    evidence["worker_addresses"] = sorted(names)
    path = Path(ctx.env.discovery_file)
    original = path.read_text()
    ctx.add_cleanup("restore_cache_discovery", lambda d: atomic_write(path, original))
    evidence["victims"] = sorted(victims)
    baseline, last_sample = None, None
    affected, absent, rebuilt = set(), set(), set()
    maximum_gap = 0
    diagnostics = MasterDiagnosticsClient(f"http://127.0.0.1:{ctx.ops.master_http_port}", timeout_s=min(5, deadline.remaining()))
    # Check capabilities before cache construction or traffic launch.
    preflight = diagnostics.snapshot(include=("workers",))
    worker_rows(preflight)
    instance_id = preflight.payload["instance_id"]
    evidence["provenance"].update(master_instance_id=instance_id, diagnostic_schema=2)
    sample_budget = SampleBudget(observation["capture"])

    fixture = partial(_cache_fixture, ctx.ops, deadline)

    def stats():
        return {e["http_addr"]: fixture(e, "inspect") for e in prefill}

    def event(name):
        e = ctx.record_event(name)
        evidence["events"].append(dict(e, name=name))
        (directory / "events.json").write_text(json.dumps(evidence["events"]))
        print("CACHE_EVENT", name, flush=True)

    alive = partial(_probe_alive, ctx, deadline)

    def sample(phase):
        nonlocal last_sample, maximum_gap
        before = ctx.clock()
        deadline.check()
        diagnostics.timeout_s = min(5, deadline.remaining())
        capture = diagnostics.snapshot(include=("workers",))
        if capture.payload["instance_id"] != instance_id:
            raise DebugUnavailable("Master instance changed during observation")
        rows = worker_rows(capture)
        now = ctx.clock()
        if last_sample is not None:
            maximum_gap = max(maximum_gap, now - last_sample)
        last_sample = now
        if baseline is not None:
            excluded = victims if phase in {"intervention", "post", "drain", "cache_settle"} else set()
            newly = set(collateral(rows, baseline, excluded, acceptance_limits["max_status_age_ms"])) - affected
            affected.update(newly)
            if newly:
                proof = dict(phase=phase, epoch_s=ctx.wall_clock(), workers=sorted(newly))
                try:
                    proof["direct_alive"] = alive([e for e in engines if e["http_addr"] == sorted(newly)[0]])
                except Exception as error:
                    proof["probe_error"] = str(error)
                evidence["healthy_violations"].append(proof)
            if phase == "intervention":
                absent.update(victims - rows.keys())
            if phase == "post":
                rebuilt.update(a for a in victims if a in rows and rows[a]["generation"] != baseline[a]["generation"])
        survivors = {a: w for a, w in rows.items() if a not in victims}

        def maximum_age(field):
            values = [w[field] for w in survivors.values()]
            return max(values) if values and all(v is not None for v in values) else None

        poll_age, applied_age = maximum_age("status_age_ms"), maximum_age("applied_status_age_ms")
        frame = dict(
            epoch_s=ctx.wall_clock(),
            phase=phase,
            snapshot_read_s=ctx.clock() - before,
            snapshot_id=capture.payload["snapshot_id"],
            capture_started_at_ms=capture.payload["capture_started_at_ms"],
            capture_finished_at_ms=capture.payload["capture_finished_at_ms"],
            workers=rows
            if phase == "cache_settle"
            else {
                a: {k: w[k] for k in ("role", "generation", "reported_alive", "lifecycle_state", "status_age_ms", "applied_status_age_ms")}
                for a, w in rows.items()
            },
            missing_workers=len((baseline or {}).keys() - rows.keys() - victims),
            healthy_max_status_age_ms=max(poll_age, applied_age) if poll_age is not None and applied_age is not None else None,
            healthy_max_rpc_age_ms=poll_age,
            healthy_max_applied_age_ms=applied_age,
            affected_workers=len(affected),
        )
        encoded = sample_budget.append(frame)
        evidence["frames"].append(frame)
        with (directory / "observations.jsonl").open("a") as f:
            f.write(encoded + "\n")
        return rows

    def hold(phase, seconds):
        start = ctx.clock()
        before = stats()
        event(phase + "_started")
        until = start + seconds
        for _ in poll_samples(deadline, observation["sample_s"], lambda: sample(phase), until=until, immediate=True, boundary_sample=False):
            pass
        after = stats()
        evidence["intensity"][phase] = interval_rates(before, after, ctx.clock() - start)
        return after

    try:
        traffic = params["traffic"]
        trace = materialize(
            directory / "traffic.jsonl",
            traffic["source"],
            "cache-sync-shared-background",
            Path(ctx.instance["source_path"]).parent,
            max_requests=int(traffic["client"]["playback"]["duration_s"] * traffic["client"]["playback"]["qps"] * 1.02),
        )
        initial_until = ctx.clock() + 60
        while ctx.clock() < initial_until:
            rows = sample("initial_health")
            if set(rows) == names and all(healthy(w, acceptance_limits["max_status_age_ms"]) for w in rows.values()):
                baseline = rows
                break
            deadline.sleep(observation["sample_s"])
        if baseline is None:
            raise ValueError("initial healthy Worker population unavailable")
        event("cache_seed_started")
        for i, e in enumerate(prefill):
            fixture(
                e,
                "configure",
                enabled=True,
                worker=i,
                workers=len(prefill),
                unique=procedure["unique_keys"],
                shared=procedure["shared_keys"],
                delta=0,
                interval_ms=procedure["churn_interval_ms"],
            )
            sample("cache_construction")
        event("cache_seed_finished")
        frozen = stats()
        baseline_keys = {a: f["reported_keys"] for a, f in frozen.items()}
        baseline = None
        ready = 0
        until = ctx.clock() + procedure["warmup_timeout_s"]
        while ctx.clock() < until:
            rows = sample("warmup")
            good = converged(
                rows, names, baseline_keys, {a: f["version"] for a, f in frozen.items()}, acceptance_limits["max_status_age_ms"]
            )
            ready = ready + 1 if good else 0
            if ready >= observation["converged_samples"]:
                baseline = rows
                break
            deadline.sleep(observation["sample_s"])
        if baseline is None:
            raise ValueError("shared cache baseline failed to converge")
        last_sample = None
        maximum_gap = 0
        evidence["cache_baseline"] = dict(
            workers=rows,
            expected_keys=baseline_keys,
            expected_global_keys=len(prefill) * procedure["unique_keys"] + procedure["shared_keys"],
            expected_versions={a: f["version"] for a, f in frozen.items()},
        )
        event("cache_baseline_ready")
        for e in prefill:
            fixture(e, "churn", delta=procedure["delta_keys"], interval_ms=procedure["churn_interval_ms"])
        hold("cache_only", procedure["cache_only_s"])
        client = ctx.services.environment.client_operations(traffic["jvm_xms"], traffic["jvm_xmx"])
        flow = JavaFlowGroup(
            client,
            directory / "flow",
            run_id=str(ctx.artifact_dir),
            group_id=traffic["group_id"],
            phase_id=traffic["phase_id"],
            poll_s=traffic["poll_s"],
            collection_profile="aggregate",
        )

        def close_flow(d):
            if flow.proc is not None:
                flow.stop_sending(d)
                flow.drain(d)

        ctx.register_resource("java_flow", flow, cleanup=close_flow, evidence=flow_evidence, request_stop=flow.request_stop)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as startup:
            pending = startup.submit(flow.start, trace, traffic["client"], deadline, target=f"127.0.0.1:{ctx.env.master_http_port + 2}")
            while not pending.done():
                sample("traffic_startup")
                deadline.sleep(observation["sample_s"])
            pending.result()
        hold("traffic_warmup", procedure["traffic_warmup_s"])
        hold("measurement", procedure["measurement_s"])
        event("measurement_finished")
        if intervention:
            evidence["victim_alive_before"] = alive(chosen)
            atomic_write(path, withdraw(original, victims))
            event("partial_workers_withdrawn")
            hold("intervention", procedure["withdraw_s"])
            evidence["victim_alive_withdrawn"] = alive(chosen)
            atomic_write(path, original)
            event("partial_workers_restored")
        # Identical post duration and unchanged background; do not stop at first convergence.
        hold("post", procedure["post_s"])
        event("post_finished")
        drain_flow(flow, deadline, sample, observation["sample_s"], event)
        evidence["client"] = flow.evidence_snapshot()
        event("performance_observation_finished")
        for e in prefill:
            fixture(e, "churn", delta=0, interval_ms=procedure["churn_interval_ms"])
        expected = stats()
        versions = {a: f["version"] for a, f in expected.items()}
        reported_keys = {a: f["reported_keys"] for a, f in expected.items()}
        until = ctx.clock() + 120
        while ctx.clock() < until:
            rows = sample("cache_settle")
            good = converged(rows, names, reported_keys, versions, acceptance_limits["max_status_age_ms"])
            evidence["convergence_samples"] = evidence["convergence_samples"] + 1 if good else 0
            if evidence["convergence_samples"] >= observation["converged_samples"]:
                break
            deadline.sleep(observation["sample_s"])
        evidence["cache_progress"] = dict(
            expected_versions=versions,
            expected_keys=reported_keys,
            converged=evidence["convergence_samples"] >= observation["converged_samples"],
            scope="cache versions, index initialization and counts; no key membership proof",
        )
        event("cache_progress_finished")
        evidence.update(
            affected_workers=sorted(affected), expired_workers=sorted(absent), rebuilt_workers=sorted(rebuilt), max_sample_gap_s=maximum_gap
        )
        event("observation_finished")
    except Exception as error:
        evidence["errors"].append(repr(error))
        evidence.update(affected_workers=sorted(affected), runtime_incomplete=True)
        # Freeze partial evidence so an interrupted observation cannot hide collateral failure.
        if not affected:
            raise
    finally:
        try:
            atomic_write(path, original)
        finally:
            (directory / "evidence.json").write_text(json.dumps(evidence))
    from cases.cache_sync_recovery.telemetry import gc_events

    logs = list(ctx.artifact_dir.glob("environment/*/master_gc.log"))
    evidence["gc"] = gc_events(logs[0]) if len(logs) == 1 else []
    evidence["gc_available"] = len(logs) == 1
    from artifacts.gate import commit_gate
    from cases.cache_sync_recovery.analysis import analyze

    result, path = commit_gate(ctx.artifact_dir, "cache_stability", evidence, analyze)
    return StageOutput({}, [CheckResult(**c) for c in result["checks"]], [str(path), str(directory / "evidence.json")])


HANDLERS = [StageHandler("cache_recovery_cycle", validate, execute, {}, checks=CHECK_IDS)]
