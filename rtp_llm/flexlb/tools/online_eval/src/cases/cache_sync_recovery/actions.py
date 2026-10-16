"""Bounded fixture controls, real discovery withdrawal, and sampled health evidence."""

import copy
import json
import math
import urllib.request
from pathlib import Path

import grpc

from adapters.master_control import master_url
from adapters.mock_control import control_json
from analysis.checks import CheckResult, evaluate
from scenario.contracts import StageHandler, StageOutput

CHECK_IDS = frozenset({"healthy_survivors", "victims_withdrawn", "victims_rebuilt", "cache_converged",
    "cache_consistency", "actual_sync_intensity", "response_shape", "clock_input_intensity", "sampling_complete", "client_complete"})


def validate(params, plan):
    if set(params) != {"procedure", "observation", "checks", "traffic", "intervention"}:
        raise ValueError("pressure cycle requires shared background and explicit intervention")
    if type(params["intervention"]) is not bool:
        raise ValueError("intervention must be boolean")
    from adapters.java_flow_spec import JavaFlowSpec
    JavaFlowSpec.read(params["traffic"])
    return copy.deepcopy(params)


def master_rows(ctx, deadline):
    url = master_url("127.0.0.1", ctx.ops.master_http_port, "inflight")
    with urllib.request.urlopen(url, timeout=min(2, deadline.remaining())) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Master health snapshot exceeds byte budget")
    payload = json.loads(raw)
    rows = {}
    for role in ("prefill", "decode"):
        for row in payload[role + "_endpoints"]:
            address = row["ip_port"]
            if address in rows:
                raise ValueError("duplicate Worker in Master snapshot")
            for field in ("generation", "status_age_ms", "reported_alive"):
                if field not in row:
                    raise ValueError("Master lacks Worker poll-health evidence: " + field)
            if (type(row["generation"]) is not int or row["generation"] <= 0
                    or type(row["reported_alive"]) is not bool
                    or type(row["status_age_ms"]) not in (int, float)
                    or not math.isfinite(row["status_age_ms"])):
                raise ValueError("invalid Worker poll-health evidence")
            rows[address] = dict(row, role=role)
    return rows


def healthy(row, maximum):
    return (row["reported_alive"] and 0 <= row["status_age_ms"] <= maximum
            and 0 <= row.get("applied_status_age_ms", row["status_age_ms"]) <= maximum
            and row.get("scheduling_status_fresh", True))


def collateral(rows, baseline, victims, maximum):
    """Absence, generation replacement and stale status all affect healthy survivors."""
    return sorted(address for address, old in baseline.items() if address not in victims
                  and (address not in rows or rows[address]["generation"] != old["generation"]
                       or not healthy(rows[address], maximum)))


def converged(rows, names, keys, versions, maximum):
    if set(rows) != set(names):
        return False
    return all(healthy(row, maximum) and (row["role"] != "prefill" or (
        row.get("cache_indexed") is True and row.get("cache_key_size") == keys
        and row.get("cache_version") == versions[address]
        and row.get("cache_indexed_version") == versions[address]))
               for address, row in rows.items())


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
    return json.dumps({name: [value for value in values if value not in victims]
                       for name, values in data.items()})


def execute(ctx, params, deadline):
    from adapters.java_flow import JavaFlowGroup
    from adapters.resource_evidence import flow_evidence
    from traffic.traffic_source import materialize
    from cases.cache_sync_recovery.telemetry import (
        fetch, metric_values, metric_sum, resource, fingerprint_matches, interval_rates,
    )
    p, o, limits = (params[k] for k in ("procedure", "observation", "checks"))
    intervention = params["intervention"]
    directory = ctx.artifact_dir / "cache-stability"
    directory.mkdir()
    evidence = dict(intervention=intervention, parameters=copy.deepcopy(params),
        environment=ctx.instance["environment"], events=[], intensity={}, frames=[], errors=[],
        healthy_violations=[], convergence_samples=0, consistency=None,
        expiry_scope="discovery withdrawal while RPC remains alive; not a health-timeout injection")
    engines = control_json(ctx.ops, "snapshot?view=topology", deadline)["engines"]
    prefill = sorted((e for e in engines if e["role"] == "prefill"), key=lambda e: int(e["name"].rsplit("-", 1)[1]))
    if len(prefill) != ctx.env.spec.n_prefill or len(engines) != ctx.env.spec.n_prefill + ctx.env.spec.n_decode:
        raise ValueError("Mock topology differs from frozen case")
    chosen = [prefill[i*len(prefill)//p["expired_workers"]] for i in range(p["expired_workers"])]
    victims = {e["http_addr"] for e in chosen} if intervention else set()
    names = {e["http_addr"] for e in engines}
    path = Path(ctx.env.discovery_file)
    original = path.read_text()
    ctx.add_cleanup("restore_cache_discovery", lambda d: atomic_write(path, original))
    evidence["victims"] = sorted(victims)
    keys = p["unique_keys"] + 2*p["shared_keys"]
    baseline, flow, last_sample = None, None, None
    affected, absent, rebuilt = set(), set(), set()
    maximum_gap = 0
    url = master_url("127.0.0.1", ctx.ops.master_http_port, "inflight")
    metrics_url = f"http://127.0.0.1:{ctx.env.master_management_port}/prometheus"

    def fixture(e, operation, **options):
        result = control_json(ctx.ops, "cache_fixture", deadline,
                              dict(engine=e["name"], operation=operation, **options))
        if result.get("engine") != e["name"] or result.get("status") != "ok":
            raise ValueError("fixture identity mismatch")
        return result["fixture"]

    def stats():
        return {e["http_addr"]: fixture(e, "inspect") for e in prefill}

    def event(name):
        e = ctx.record_event(name)
        evidence["events"].append(dict(e, name=name))
        (directory/"events.json").write_text(json.dumps(evidence["events"]))
        print("CACHE_EVENT", name, flush=True)

    def alive(entries):
        results = []
        for e in entries:
            with grpc.insecure_channel(e["grpc_addr"]) as channel:
                response = ctx.ops.pb2_grpc.RpcServiceStub(channel).GetWorkerStatus(
                    ctx.ops.pb2.StatusVersionPB(), timeout=min(2, deadline.remaining()))
            results.append(dict(worker=e["http_addr"], alive=response.alive, epoch_s=ctx.wall_clock()))
            if not response.alive:
                raise ValueError("selected Worker is not independently alive")
        return results

    def sample(phase):
        nonlocal last_sample, maximum_gap
        before = ctx.clock()
        raw = json.loads(fetch(url, min(5, deadline.remaining())))
        now = ctx.clock()
        if last_sample is not None:
            maximum_gap = max(maximum_gap, now-last_sample)
        last_sample = now
        rows = {w["ip_port"]: dict(w, role=role) for role in ("prefill", "decode")
                for w in raw[role+"_endpoints"]}
        if len(rows) != len(raw["prefill_endpoints"]) + len(raw["decode_endpoints"]):
            raise ValueError("duplicate Worker identity")
        for row in rows.values():
            if not {"generation", "status_age_ms", "reported_alive"} <= row.keys():
                raise ValueError("Master lacks observational health adapter")
        if baseline is not None:
            excluded = victims if phase in {"intervention", "post"} else set()
            newly = set(collateral(rows, baseline, excluded, limits["max_status_age_ms"])) - affected
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
        if phase == "post":
            ready = set(rows) == names and all(healthy(w, limits["max_status_age_ms"])
                and (w["role"] != "prefill" or w.get("cache_key_size") == keys)
                for w in rows.values()) and rebuilt == victims
            evidence["convergence_samples"] = evidence["convergence_samples"] + 1 if ready else 0
        meters = metric_values(fetch(metrics_url, min(5, deadline.remaining())))
        frame = dict(epoch_s=ctx.wall_clock(), phase=phase, snapshot_read_s=ctx.clock()-before,
            workers=rows, p_workers=len(raw["prefill_endpoints"]), d_workers=len(raw["decode_endpoints"]),
            max_status_age_ms=max((w["status_age_ms"] for w in rows.values()), default=None),
            missing_workers=len((baseline or {}).keys()-rows.keys()-victims),
            healthy_max_status_age_ms=max((max(w["status_age_ms"], w.get("applied_status_age_ms", w["status_age_ms"]))
                for a, w in rows.items() if a not in victims), default=None),
            healthy_max_rpc_age_ms=max((w["status_age_ms"] for a,w in rows.items()
                if a not in victims and "applied_status_age_ms" in w), default=None),
            healthy_max_applied_age_ms=max((w["applied_status_age_ms"]
                for a,w in rows.items() if a not in victims and "applied_status_age_ms" in w), default=None),
            stale_applied_workers=sum(w.get("applied_status_age_ms",w["status_age_ms"]) > limits["max_status_age_ms"]
                for a,w in rows.items() if a not in victims),
            routable_workers=(sum(w["scheduling_status_fresh"] for a,w in rows.items() if a not in victims)
                if all("scheduling_status_fresh" in w for w in rows.values()) else None),
            stale_workers=sum(not healthy(w, limits["max_status_age_ms"]) for w in rows.values()),
            replaced_workers=sum(a in baseline and w["generation"] != baseline[a]["generation"] for a, w in rows.items()) if baseline else 0,
            global_keys=metric_sum(meters, "cache_global_total_count"),
            affected_workers=len(affected), reported_dead=sum(not w["reported_alive"] for w in rows.values()),
            process_allocated_bytes=raw.get("process_allocated_bytes"), scheduler_inflight=raw["scheduler_inflight"],
            heap_bytes=metric_sum(meters, "jvm_memory_used_bytes", {"area": "heap"}),
            direct_bytes=metric_sum(meters, "jvm_buffer_memory_used_bytes", {"id": "direct"}),
            diff_scanned_keys=raw.get("diagnostic_diff_scanned_keys", metric_sum(meters, "cache_diff_scanned_keys_total", {"role": "PREFILL"})),
            worker_status_commits=metric_sum(meters, "engine_health_check_visitor_success_qps_total"),
            cache_sync_commits=metric_sum(meters, "cache_status_check_visitor_success_qps_total", {"role": "PREFILL"}),
            master=resource(ctx.env.master), mock=resource(ctx.env.mock))
        evidence["frames"].append(frame)
        with (directory/"observations.jsonl").open("a") as f:
            f.write(json.dumps(frame)+"\n")
        return rows

    def hold(phase, seconds):
        start = ctx.clock()
        before = stats()
        event(phase+"_started")
        until = start+seconds
        while ctx.clock() < until:
            deadline.check()
            sample(phase)
            deadline.sleep(min(o["sample_s"], max(0, until-ctx.clock())))
        after = stats()
        evidence["intensity"][phase] = interval_rates(before, after, ctx.clock()-start)
        return after

    try:
        t = params["traffic"]
        trace = materialize(directory/"traffic.jsonl", t["source"], "cache-sync-shared-background",
                            Path(ctx.instance["source_path"]).parent,
                            max_requests=int(t["client"]["playback"]["duration_s"]*t["client"]["playback"]["qps"]*1.02))
        initial_until = ctx.clock()+60
        while ctx.clock() < initial_until:
            rows = sample("initial_health")
            if set(rows) == names and all(healthy(w, limits["max_status_age_ms"]) for w in rows.values()):
                baseline = rows
                break
            deadline.sleep(o["sample_s"])
        if baseline is None:
            raise ValueError("initial healthy Worker population unavailable")
        event("cache_seed_started")
        for i, e in enumerate(prefill):
            fixture(e, "seed", worker=i, workers=len(prefill), unique=1, shared=0)
            sample("cache_construction")
            if i+1 < len(prefill):
                deadline.sleep(p["seed_interval_s"])
        event("cache_growth_started")
        for step in range(1, p["baseline_steps"]+1):
            for e in prefill:
                fixture(e, "grow", unique=max(1, p["unique_keys"]*step//p["baseline_steps"]),
                        shared=p["shared_keys"]*step//p["baseline_steps"])
            sample("cache_construction")
            deadline.sleep(p["baseline_interval_s"])
        event("cache_growth_finished")
        frozen = stats()
        baseline = None
        ready = 0
        until = ctx.clock()+p["warmup_timeout_s"]
        while ctx.clock() < until:
            rows = sample("warmup")
            good = set(rows) == names and all(healthy(w, limits["max_status_age_ms"]) and (
                w["role"] != "prefill" or w.get("cache_key_size") == keys
                and w.get("cache_version") == frozen[a]["version"]) for a, w in rows.items())
            ready = ready+1 if good else 0
            if ready >= o["converged_samples"]:
                baseline = rows
                break
            deadline.sleep(o["sample_s"])
        if baseline is None:
            raise ValueError("shared cache baseline failed to converge")
        last_sample = None
        maximum_gap = 0
        event("cache_baseline_ready")
        for e in prefill:
            fixture(e, "clock", interval_ms=p["churn_interval_ms"])
            fixture(e, "churn", delta=p["delta_keys"], every=1)
        hold("cache_only", p["cache_only_s"])
        client = ctx.services.environment.client_operations(t["jvm_xms"], t["jvm_xmx"])
        flow = JavaFlowGroup(client, directory/"flow", run_id=str(ctx.artifact_dir), group_id=t["group_id"],
                             phase_id=t["phase_id"], poll_s=t["poll_s"], collection_profile="aggregate")

        def close_flow(d):
            if flow.proc is not None:
                flow.stop_sending(d)
                flow.drain(d)

        ctx.register_resource("java_flow", flow, cleanup=close_flow, evidence=flow_evidence, request_stop=flow.request_stop)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as startup:
            pending = startup.submit(flow.start, trace, t["client"], deadline,
                                     target=f"127.0.0.1:{ctx.env.master_http_port+2}")
            while not pending.done():
                sample("traffic_startup")
                deadline.sleep(o["sample_s"])
            pending.result()
        hold("traffic_warmup", p["traffic_warmup_s"])
        hold("measurement", p["measurement_s"])
        event("measurement_finished")
        if intervention:
            evidence["victim_alive_before"] = alive(chosen)
            atomic_write(path, withdraw(original, victims))
            event("partial_workers_withdrawn")
        hold("intervention", p["withdraw_s"])
        if intervention:
            evidence["victim_alive_withdrawn"] = alive(chosen)
            atomic_write(path, original)
            event("partial_workers_restored")
        # Identical post duration and unchanged background; do not stop at first convergence.
        hold("post", p["post_s"])
        event("post_finished")
        flow.stop_sending(deadline)
        event("sending_stopped")
        flow.drain(deadline)
        event("drain_finished")
        evidence["client"] = flow.evidence_snapshot()
        event("performance_observation_finished")
        for e in prefill:
            fixture(e, "churn", delta=0, every=0)
        expected = stats()
        until = ctx.clock()+120
        verified = None
        while ctx.clock() < until:
            verified = json.loads(fetch(url.rsplit("/", 1)[0]+"/cache_fingerprint", min(90, deadline.remaining())))
            if fingerprint_matches(expected, verified):
                break
            deadline.sleep(o["sample_s"])
        consistent = bool(verified and fingerprint_matches(expected, verified))
        consistent = consistent and verified.get("global_unique_keys") == len(prefill)*(p["unique_keys"]+p["shared_keys"])
        consistent = consistent and verified.get("global_mappings") == len(prefill)*keys
        evidence["consistency"] = dict(consistent=consistent, expected=expected, observed=verified,
            scope="frozen fingerprints and exhaustive status/global membership; counts guard extra mappings")
        event("consistency_finished")
        last = evidence["frames"][-1]["workers"]
        healthy_end = (set(last) == names and all(healthy(w, limits["max_status_age_ms"]) for w in last.values())
                       and evidence["convergence_samples"] >= o["converged_samples"])
        evidence.update(affected_workers=sorted(affected), expired_workers=sorted(absent), rebuilt_workers=sorted(rebuilt),
                        max_sample_gap_s=maximum_gap)
        measured = evidence["intensity"]["measurement"]
        checks = [evaluate("healthy_survivors", len(affected), "eq", 0),
            evaluate("cache_converged", healthy_end, "eq", True),
            evaluate("cache_consistency", consistent, "eq", True),
            evaluate("actual_sync_intensity", measured["responses"], "ge", limits["min_cache_responses_per_s"]),
            evaluate("response_shape", measured["totals"]["entries"] == measured["totals"]["responses"]*keys
                     and measured["totals"]["full_responses"] == measured["totals"]["responses"], "eq", True),
            evaluate("clock_input_intensity", limits["min_added_keys_per_s"] <= measured["added_keys"] <= limits["max_added_keys_per_s"], "eq", True),
            evaluate("sampling_complete", maximum_gap, "le", o["max_gap_s"]),
            evaluate("client_complete", evidence["client"]["complete"], "eq", True)]
        if intervention:
            checks.extend([evaluate("victims_withdrawn", len(absent), "eq", len(victims)),
                           evaluate("victims_rebuilt", len(rebuilt), "eq", len(victims))])
        else:
            checks.extend(CheckResult(id=k, status="SKIP", detail="steady has no withdrawal event")
                          for k in ("victims_withdrawn", "victims_rebuilt"))
        evidence["checks"] = [dict(id=c.id, status=c.status, actual=c.actual) for c in checks]
        event("observation_finished")
        return StageOutput({}, checks, [str(directory/"evidence.json")])
    except Exception as error:
        evidence["errors"].append(repr(error))
        # A later collection error cannot erase an already observed collateral failure.
        # Remaining checks stay unevaluated; reports retain the incomplete runtime boundary.
        if affected:
            evidence["affected_workers"] = sorted(affected)
            evidence["runtime_incomplete"] = True
            checks = [evaluate("healthy_survivors", len(affected), "eq", 0)]
            checks.extend(CheckResult(id=k, status="SKIP", actual=None, expected=None,
                detail="observation interrupted") for k in sorted(CHECK_IDS-{"healthy_survivors"}))
            evidence["checks"] = [dict(id=c.id, status=c.status, actual=c.actual) for c in checks]
            return StageOutput({}, checks, [str(directory/"evidence.json")])
        raise
    finally:
        atomic_write(path, original)
        (directory/"evidence.json").write_text(json.dumps(evidence))


HANDLERS = [StageHandler("cache_recovery_cycle", validate, execute, {}, checks=CHECK_IDS)]
