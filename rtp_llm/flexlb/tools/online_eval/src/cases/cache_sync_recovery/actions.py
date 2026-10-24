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
                       "stable_actual_sync_rate", "churn_actual_sync_rate", "small_diff_observed"})


class HealthIsolationFailure(Exception):
    """A recorded healthy-survivor violation is already a definitive failed gate."""


def validate(params, plan):
    if set(params) != {"procedure", "observation", "checks"}:
        raise ValueError("cache recovery requires procedure, observation and checks")
    for section in params.values():
        if not isinstance(section, dict) or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in section.values()):
            raise ValueError("invalid recovery parameters")
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
    return row["reported_alive"] and 0 <= row["status_age_ms"] <= maximum


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
    p, observation, limits = (params[name] for name in ("procedure", "observation", "checks"))
    engines = control_json(ctx.ops, "snapshot?view=topology", deadline)["engines"]
    prefill = sorted((entry for entry in engines if entry["role"] == "prefill"),
                     key=lambda entry: int(entry["name"].rsplit("-", 1)[1]))
    if len(prefill) != ctx.env.spec.n_prefill or len(engines) != ctx.env.spec.n_prefill + ctx.env.spec.n_decode:
        raise ValueError("Mock topology differs from configured population")
    # Evenly spaced victims; never stop their real RPC service.
    selected = [prefill[i * len(prefill) // p["expired_workers"]] for i in range(p["expired_workers"])]
    victim_addresses = {entry["http_addr"] for entry in selected}
    expected_names = {entry["http_addr"] for entry in engines}
    keys = p["unique_keys"] + 2 * p["shared_keys"]
    path = Path(ctx.env.discovery_file)
    original = path.read_text()
    hidden = withdraw(original, {entry["http_addr"] for entry in selected})
    # LIFO cleanup restores discovery even when cancellation interrupts this action.
    ctx.add_cleanup("restore_cache_recovery_discovery", lambda cleanup_deadline: atomic_write(path, original))
    evidence_path = ctx.artifact_dir / "cache-recovery.jsonl"
    summary_path = ctx.artifact_dir / "cache-recovery-summary.json"
    evidence = dict(keys_per_worker=keys, expected_unique_keys=len(prefill) * (p["unique_keys"] + p["shared_keys"]),
                    victims=sorted(victim_addresses), fixture="synthetic ring; separate from inference KV",
                    identity_scope="Master discovery ip:httpPort; alive probes use grpc_addr",
                    byte_scope="fixture protobuf map and version; excludes other RPC fields and framing",
                    expiry_scope="controlled discovery withdrawal while RPC remains alive",
                    errors=[], events=[], intensity={})

    def fixture(entry, operation, **options):
        response = control_json(ctx.ops, "cache_fixture", deadline,
                                {"engine": entry["name"], "operation": operation, **options})
        if response.get("status") != "ok" or response.get("engine") != entry["name"]:
            raise ValueError("fixture lacks acknowledged engine identity")
        return response["fixture"]

    def event(name):
        evidence["events"].append(ctx.record_event(name))

    def stats():
        return {entry["http_addr"]: fixture(entry, "inspect") for entry in prefill}

    def alive(entries):
        for entry in entries:
            with grpc.insecure_channel(entry["grpc_addr"]) as channel:
                result = ctx.ops.pb2_grpc.RpcServiceStub(channel).GetWorkerStatus(
                    ctx.ops.pb2.StatusVersionPB(), timeout=min(2, deadline.remaining()))
            if not result.alive:
                raise ValueError("fixture Worker did not independently report alive: " + entry["name"])

    last_sample = None
    max_gap = 0.0
    peak_age = 0.0
    affected = set()
    baseline = None
    expired = set()

    with evidence_path.open("w") as output_file:
        def sample(phase):
            nonlocal last_sample, max_gap, peak_age
            rows = master_rows(ctx, deadline)
            now = ctx.clock()
            if last_sample is not None:
                max_gap = max(max_gap, now - last_sample)
                if now - last_sample > observation["max_gap_s"]:
                    raise ValueError("health sampling gap exceeds evidence budget")
            last_sample = now
            if baseline is not None:
                excluded = victim_addresses if phase in {"withdrawn", "recovery"} else set()
                affected.update(collateral(rows, baseline, excluded, limits["max_status_age_ms"]))
                for address in set(baseline) - victim_addresses:
                    if address in rows:
                        peak_age = max(peak_age, rows[address]["status_age_ms"])
            if phase == "withdrawn":
                expired.update(victim_addresses - rows.keys())
            output_file.write(json.dumps(dict(epoch_s=ctx.wall_clock(), phase=phase, workers=rows)) + "\n")
            output_file.flush()
            if affected:
                # Stop on positive evidence before a later stall hides it behind an I/O error.
                address = sorted(affected)[0]
                entry = next(entry for entry in engines if entry["http_addr"] == address)
                proof = dict(address=address, grpc_addr=entry["grpc_addr"], phase=phase,
                             epoch_s=ctx.wall_clock(), master=rows.get(address))
                try:
                    with grpc.insecure_channel(entry["grpc_addr"]) as channel:
                        response = ctx.ops.pb2_grpc.RpcServiceStub(channel).GetWorkerStatus(
                            ctx.ops.pb2.StatusVersionPB(), timeout=min(1, deadline.remaining()))
                    proof["rpc_alive"] = response.alive
                except Exception as error:
                    proof["probe_error"] = str(error)
                proof["probe_end_epoch_s"] = ctx.wall_clock()
                evidence["healthy_violation_probe"] = proof
                raise HealthIsolationFailure()
            return rows

        def hold(phase, seconds):
            end = ctx.clock() + seconds
            while ctx.clock() < end:
                sample(phase)
                deadline.sleep(min(observation["sample_s"], end - ctx.clock()) if ctx.clock() < end else 0)

        try:
            event("cache_seed_started")
            for i, entry in enumerate(prefill):
                fixture(entry, "seed", worker=i, workers=len(prefill), unique=p["unique_keys"], shared=p["shared_keys"])
                if p["seed_interval_s"] and i + 1 < len(prefill):
                    deadline.sleep(p["seed_interval_s"])
            initial = stats()
            versions = {address: value["version"] for address, value in initial.items()}
            end = ctx.clock() + p["warmup_timeout_s"]
            consecutive = 0
            while ctx.clock() < end:
                rows = sample("warmup")
                consecutive = consecutive + 1 if converged(rows, expected_names, keys, versions, limits["max_status_age_ms"]) else 0
                if consecutive >= observation["converged_samples"]:
                    baseline = rows
                    break
                deadline.sleep(observation["sample_s"])
            if baseline is None:
                raise ValueError("cache baseline did not converge")
            alive(engines)
            # Baseline establishment is excluded from the intervention sampling budget.
            last_sample = None
            event("cache_baseline_ready")
            for phase, duration in (("stable", p["stable_s"]), ("churn", p["churn_s"])):
                if phase == "churn":
                    for entry in prefill:
                        fixture(entry, "churn", delta=p["delta_keys"], every=p["change_every_responses"])
                before = stats()
                started = ctx.clock()
                event("cache_" + phase + "_started")
                hold(phase, duration)
                after = stats()
                elapsed = ctx.clock() - started
                totals = {field: sum(after[address][field] - before[address][field] for address in after)
                          for field in ("responses", "entries", "key_payload_bytes", "changed_responses", "added_keys", "removed_keys")}
                evidence["intensity"][phase] = dict(totals, observed_s=elapsed,
                    responses_per_s=totals["responses"] / elapsed, entries_per_s=totals["entries"] / elapsed,
                    key_payload_bytes_per_s=totals["key_payload_bytes"] / elapsed)
            alive(selected)
            atomic_write(path, hidden)
            event("partial_workers_withdrawn")
            hold("withdrawn", p["withdraw_s"])
            alive(selected)
            atomic_write(path, original)
            restored_at = ctx.clock()
            event("partial_workers_restored")
            for entry in prefill:
                fixture(entry, "churn", delta=0, every=0)
            frozen = stats()
            versions = {address: value["version"] for address, value in frozen.items()}
            end = ctx.clock() + p["recovery_timeout_s"]
            consecutive = 0
            recovered = False
            rebuilt = set()
            while ctx.clock() < end:
                rows = sample("recovery")
                rebuilt.update(address for address in victim_addresses if address in rows
                               and rows[address]["generation"] != baseline[address]["generation"])
                good = converged(rows, expected_names, keys, versions, limits["max_status_age_ms"])
                consecutive = consecutive + 1 if good and rebuilt == victim_addresses else 0
                if consecutive >= observation["converged_samples"]:
                    recovered = True
                    break
                deadline.sleep(observation["sample_s"])
            event("cache_recovery_finished")
            evidence.update(affected_workers=sorted(affected), max_healthy_status_age_ms=peak_age,
                max_sample_gap_s=max_gap, expired_workers=sorted(expired), rebuilt_workers=sorted(rebuilt),
                recovered=recovered, recovery_s=ctx.clock() - restored_at)
            check_evidence = dict(artifact=str(evidence_path), summary=str(summary_path),
                                  expiry_is_observed_absence=True, generation_is_observed_replacement=True)
            checks = [
                evaluate("healthy_survivors", len(affected), "eq", 0, evidence=check_evidence),
                evaluate("victims_withdrawn", len(expired), "eq", len(selected), evidence=check_evidence),
                evaluate("victims_rebuilt", len(rebuilt), "eq", len(selected), evidence=check_evidence),
                evaluate("cache_converged", recovered, "eq", True, evidence=check_evidence),
            ]
            for phase, intensity in evidence["intensity"].items():
                checks.append(evaluate(phase + "_actual_sync_rate", intensity["responses_per_s"], "ge",
                                       limits["min_cache_responses_per_s"], evidence=check_evidence))
            checks.append(evaluate("small_diff_observed", evidence["intensity"]["churn"]["changed_responses"],
                                   "ge", 1, evidence=check_evidence))
            return StageOutput({}, checks, [str(evidence_path), str(summary_path)])
        except HealthIsolationFailure:
            evidence["aborted_after_health_violation"] = True
            checks = [evaluate("healthy_survivors", len(affected), "eq", 0,
                evidence=dict(artifact=str(evidence_path), summary=str(summary_path)))]
            checks.extend(CheckResult(name, "SKIP", detail="stopped after recorded health isolation failure")
                          for name in sorted(CHECK_IDS - {"healthy_survivors"}))
            return StageOutput({}, checks, [str(evidence_path), str(summary_path)])
        except Exception as error:
            evidence["errors"].append(str(error))
            raise
        finally:
            evidence.update(affected_workers=sorted(affected), max_healthy_status_age_ms=peak_age,
                            max_sample_gap_s=max_gap)
            # Do not leave a withdrawal in place on a failed gate, evidence error or timeout.
            atomic_write(path, original)
            summary_path.write_text(json.dumps(evidence, indent=2))


HANDLERS = [StageHandler("cache_recovery_cycle", validate, execute, {}, checks=CHECK_IDS)]
