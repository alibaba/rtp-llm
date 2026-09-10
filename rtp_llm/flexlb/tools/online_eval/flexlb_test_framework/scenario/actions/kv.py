"""Per-engine cache evidence and routing checks, composed explicitly by YAML."""

import copy
import json
import math
import uuid

from ...grade import GradeReport
from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import request_success
from .engine_control import ENGINE_NAME, _engines, _http


def _fields(params, allowed, required):
    if (
        not isinstance(params, dict)
        or set(params) - set(allowed)
        or set(required) - set(params)
    ):
        raise ValueError("invalid KV action parameters")
    return copy.deepcopy(params)


def _engine(value, plan):
    if isinstance(value, dict):
        plan.reference(value, "string")
    elif not isinstance(value, str) or not ENGINE_NAME.fullmatch(value):
        raise ValueError("KV engine must be an explicit name or typed string reference")


def _keys(value):
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 4096
        or any(type(k) is not int or not -(2**63) <= k < 2**63 for k in value)
    ):
        raise ValueError("KV keys must be a nonempty bounded int64 list")


def _target(ctx, value):
    name = ctx.resolve(value)
    if not isinstance(name, str) or not ENGINE_NAME.fullmatch(name):
        raise ValueError("resolved KV target is not an engine name")
    return name


def _artifact(ctx, prefix, value):
    path = ctx.artifact_dir / f"{prefix}-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(value, indent=2))
    return str(path)


def _snapshot(ctx, targets, deadline):
    entries = _engines(_http(ctx.ops, "snapshot", deadline), targets)
    for row in entries.values():
        # An absent key-set is unavailable evidence, never an empty cache.
        value = row.get("cache_key_set")
        if not isinstance(value, list) or any(
            type(k) is not int or not -(2**63) <= k < 2**63 for k in value
        ):
            raise ValueError("snapshot lacks a typed cache key set")
    return dict(engines=entries, sampled_s=ctx.clock(), env_epoch=ctx.env_epoch)


def _snapshot_validate(params, plan):
    p = _fields(params, {"targets", "quiet_s", "exclude"}, {"targets"})
    if not isinstance(p["targets"], list) or not 1 <= len(p["targets"]) <= 32:
        raise ValueError("KV snapshot needs 1..32 explicit targets")
    for name in p["targets"]:
        _engine(name, plan)
    p.setdefault("exclude", [])
    if not isinstance(p["exclude"], list) or len(p["exclude"]) > 31:
        raise ValueError("snapshot exclusions must be a bounded engine list")
    for name in p["exclude"]:
        _engine(name, plan)
    p.setdefault("quiet_s", 0)
    if (
        type(p["quiet_s"]) not in (int, float)
        or not math.isfinite(p["quiet_s"])
        or not 0 <= p["quiet_s"] <= 30
    ):
        raise ValueError("quiet window must be finite and at most 30 seconds")
    return p


def snapshot(ctx, params, deadline):
    targets = [_target(ctx, value) for value in params["targets"]]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate resolved KV targets")
    excluded = [_target(ctx, value) for value in params["exclude"]]
    if len(set(excluded)) != len(excluded) or not set(excluded) <= set(targets):
        raise ValueError("snapshot exclusions must be distinct declared targets")
    targets = [name for name in targets if name not in excluded]
    if not targets:
        raise ValueError("KV snapshot cannot exclude every target")
    samples, previous, changed = [], None, ctx.clock()
    try:
        while True:
            deadline.check()
            current = _snapshot(ctx, targets, deadline)
            samples.append(current)
            sets = {
                name: frozenset(row["cache_key_set"])
                for name, row in current["engines"].items()
            }
            if sets != previous:
                previous, changed = sets, ctx.clock()
            if ctx.clock() - changed >= params["quiet_s"]:
                break
            deadline.sleep(0.5)
    finally:
        path = _artifact(ctx, "kv-key-observations", samples)
    current["quiet_s"] = params["quiet_s"]
    current["sample_count"] = len(samples)
    return StageOutput(
        {"snapshot": ctx.register_resource("kv_snapshot", current)}, artifacts=[path]
    )


def _landing_validate(params, plan):
    p = _fields(params, {"requests", "phase", "identity_snapshot"}, {"requests"})
    if "identity_snapshot" in p:
        plan.reference(p["identity_snapshot"], "kv_snapshot")
    plan.reference(p["requests"], "requests")
    p.setdefault("phase", "terminal")
    if p["phase"] not in ("scheduled", "terminal"):
        raise ValueError("landing phase must be scheduled or terminal")
    return p


def landing(ctx, params, deadline):
    records = ctx.resource(params["requests"], "requests").snapshot_records()
    if len(records) != 1 or records[0]["schedule"]["status"] != "OK":
        raise ValueError("landing requires exactly one successful Schedule record")
    record = records[0]
    if params["phase"] == "terminal" and (
        not request_success(record)
        or record.get("consumer_completion_verified") is not True
    ):
        raise ValueError("terminal landing lacks successful consumer completion")
    deadline.check()
    if "identity_snapshot" in params:
        # Only resolve a stable endpoint identity here; never reuse cached KV counters.
        fleet = ctx.resource(params["identity_snapshot"], "kv_snapshot")
        raw = {"engines": list(fleet["engines"].values())}
    else:
        raw = _http(ctx.ops, "snapshot", deadline)
    matches = [
        row
        for row in raw.get("engines", [])
        if row.get("grpc_addr") == record["prefill_addr"]
        and row.get("role") == "prefill"
    ]
    if len(matches) != 1:
        raise ValueError("landing address has no unique prefill engine identity")
    name = matches[0]["name"]
    _engines(raw, [name])
    return StageOutput(
        {"engine": name},
        artifacts=[
            _artifact(
                ctx,
                "kv-landing",
                dict(record=record, engine=matches[0], phase=params["phase"]),
            )
        ],
    )


def _evict_validate(params, plan):
    p = _fields(params, {"engine", "keys"}, {"engine", "keys"})
    _engine(p["engine"], plan)
    _keys(p["keys"])
    return p


def _distinct_validate(params, plan):
    p = _fields(params, {"first", "second"}, {"first", "second"})
    _engine(p["first"], plan)
    _engine(p["second"], plan)
    return p


def distinct(ctx, params, deadline):
    deadline.check()
    first, second = (_target(ctx, params[key]) for key in ("first", "second"))
    passed = first != second
    return StageOutput(
        {"distinct": passed},
        [
            CheckResult(
                "distinct_holders",
                "PASS" if passed else "FAIL",
                actual=[first, second],
                expected="two distinct prefill engines",
                evidence=dict(complete=True, sample_count=2, min_samples=2),
            )
        ],
    )


def same(ctx, params, deadline):
    deadline.check()
    first, second = (_target(ctx, params[key]) for key in ("first", "second"))
    passed = first == second
    return StageOutput(
        {"same": passed},
        [
            CheckResult(
                "same_holder",
                "PASS" if passed else "FAIL",
                actual=[first, second],
                expected="same prefill engine",
                evidence=dict(complete=True, sample_count=2, min_samples=2),
            )
        ],
    )


def evict(ctx, params, deadline):
    name = _target(ctx, params["engine"])
    evidence = dict(
        before=_snapshot(ctx, [name], deadline), keys=params["keys"], response=None
    )
    try:
        response = _http(
            ctx.ops, "cache_evict", deadline, {"engine": name, "keys": params["keys"]}
        )
        evidence["response"] = response
        if response.get("status") != "ok" or response.get("engine") != name:
            raise ValueError("cache eviction lacks owned target acknowledgement")
        after = _snapshot(ctx, [name], deadline)
        evidence["after"] = after
        if (
            after["engines"][name]["grpc_addr"]
            != evidence["before"]["engines"][name]["grpc_addr"]
        ):
            raise ValueError("cache eviction endpoint changed")
        remaining = set(params["keys"]) & set(after["engines"][name]["cache_key_set"])
        evidence.update(complete=True, sample_count=2, min_samples=2)
    finally:
        path = _artifact(ctx, "kv-eviction", evidence)
    return StageOutput(
        {"snapshot": ctx.register_resource("kv_snapshot", after)},
        [
            CheckResult(
                "evicted",
                "PASS" if not remaining else "FAIL",
                actual=sorted(remaining),
                expected=[],
                evidence=evidence,
            )
        ],
        [path],
    )


def _prefix_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "engine", "keys", "expected"},
        {"snapshot", "engine", "keys", "expected"},
    )
    plan.reference(p["snapshot"], "kv_snapshot")
    _engine(p["engine"], plan)
    _keys(p["keys"])
    if type(p["expected"]) is not int or not 0 <= p["expected"] <= len(p["keys"]):
        raise ValueError("expected contiguous run outside key count")
    return p


def prefix(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "kv_snapshot")
    name = _target(ctx, params["engine"])
    keys = set(data["engines"][name]["cache_key_set"])
    run = 0
    for key in params["keys"]:
        if key not in keys:
            break
        run += 1
    return StageOutput(
        {"run": run},
        [
            CheckResult(
                "contiguous_prefix",
                "PASS" if run == params["expected"] else "FAIL",
                actual=run,
                expected=params["expected"],
                evidence=dict(
                    complete=True,
                    sample_count=1,
                    min_samples=1,
                    snapshot=data,
                    engine=name,
                ),
            )
        ],
    )


def _affinity_validate(params, plan, upper=False):
    p = _fields(
        params,
        {"requests", "holder", "min_samples", "bands"},
        {"requests", "holder", "min_samples", "bands"},
    )
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 100:
        raise ValueError("affinity requires explicit bounded request cohorts")
    for value in p["requests"]:
        plan.reference(value, "requests")
    _engine(p["holder"], plan)
    if type(p["min_samples"]) is not int or p["min_samples"] < 1:
        raise ValueError("affinity requires positive sample floor")
    bands = p["bands"]
    if (
        not isinstance(bands, dict)
        or set(bands) != {"strict", "normal", "loose"}
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in bands.values()
        )
        or not (
            bands["strict"] <= bands["normal"] <= bands["loose"]
            if upper
            else bands["strict"] >= bands["normal"] >= bands["loose"]
        )
    ):
        raise ValueError("affinity requires ordered explicit concentration bands")
    return p


def affinity(ctx, params, deadline, property_id="M3"):
    rows = [
        row
        for ref in params["requests"]
        for row in ctx.resource(ref, "requests").snapshot_records()
    ]
    if not rows or len({row["wire_request_id"] for row in rows}) != len(rows):
        raise ValueError("affinity cohort empty or duplicated")
    if any(
        row.get("consumer_completion_verified") is not True
        or not isinstance(row.get("prefill_addr"), str)
        or not row["prefill_addr"]
        for row in rows
    ):
        raise ValueError("affinity cohort lacks consumer exit evidence")
    name = _target(ctx, params["holder"])
    holder = _engines(_http(ctx.ops, "snapshot", deadline), [name])[name]
    share = sum(row["prefill_addr"] == holder["grpc_addr"] for row in rows) / len(rows)
    complete = len(rows) >= params["min_samples"] and all(
        request_success(row) for row in rows
    )
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    passed = report.check(property_id, share, bands=params["bands"])
    evidence = dict(
        complete=True,
        sample_count=len(rows),
        min_samples=params["min_samples"],
        records=rows,
        holder=holder,
        grade=report.run_grade,
        achieved=report.achieved,
        bands=params["bands"],
    )
    return StageOutput(
        {"share": share},
        [
            CheckResult(
                "P6",
                "PASS" if complete else "FAIL",
                actual=complete,
                expected=True,
                evidence=evidence,
            ),
            CheckResult(
                property_id,
                "PASS" if passed and len(rows) >= params["min_samples"] else "FAIL",
                actual=share,
                expected=params["bands"][report.run_grade],
                evidence=evidence,
            ),
        ],
        [_artifact(ctx, "kv-affinity", evidence)],
    )


def fidelity(ctx, params, deadline):
    return affinity(ctx, params, deadline, property_id="P9")


def _holder_share_validate(params, plan):
    return _affinity_validate(params, plan, upper=True)


def holder_share(ctx, params, deadline):
    return affinity(ctx, params, deadline, property_id="M2")


def _off_holder_validate(params, plan):
    p = _fields(
        params,
        {"requests", "holder", "min_samples"},
        {"requests", "holder", "min_samples"},
    )
    # Reuse the same cohort/holder/sample contract; this action has no graded band.
    validated = _affinity_validate(
        dict(p, bands=dict(strict=0, normal=0, loose=0)), plan
    )
    validated.pop("bands")
    return validated


def off_holder(ctx, params, deadline):
    rows = [
        row
        for ref in params["requests"]
        for row in ctx.resource(ref, "requests").snapshot_records()
    ]
    if (
        not rows
        or len({row["wire_request_id"] for row in rows}) != len(rows)
        or any(
            row.get("consumer_completion_verified") is not True
            or not isinstance(row.get("prefill_addr"), str)
            or not row["prefill_addr"]
            for row in rows
        )
    ):
        raise ValueError("off-holder cohort lacks distinct terminal landing evidence")
    name = _target(ctx, params["holder"])
    engine = _engines(_http(ctx.ops, "snapshot", deadline), [name])[name]
    count = sum(row["prefill_addr"] != engine["grpc_addr"] for row in rows)
    complete = len(rows) >= params["min_samples"] and all(
        request_success(row) for row in rows
    )
    evidence = dict(
        complete=True,
        sample_count=len(rows),
        min_samples=params["min_samples"],
        records=rows,
        holder=engine,
    )
    return StageOutput(
        {"count": count},
        [
            CheckResult(
                "P2",
                "PASS" if complete and count >= 1 else "FAIL",
                actual=count,
                expected="at least one off-holder landing",
                evidence=evidence,
            ),
            CheckResult(
                "P6",
                "PASS" if complete else "FAIL",
                actual=complete,
                expected=True,
                evidence=evidence,
            ),
        ],
        [_artifact(ctx, "kv-off-holder", evidence)],
    )


def _spread_validate(params, plan):
    p = _fields(
        params,
        {"requests", "min_samples", "bands"},
        {"requests", "min_samples", "bands"},
    )
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 100:
        raise ValueError("spread requires explicit bounded request cohorts")
    for ref in p["requests"]:
        plan.reference(ref, "requests")
    if type(p["min_samples"]) is not int or p["min_samples"] < 1:
        raise ValueError("spread needs a positive sample floor")
    bands = p["bands"]
    if (
        not isinstance(bands, dict)
        or set(bands) != {"strict", "normal", "loose"}
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in bands.values()
        )
        or not bands["strict"] <= bands["normal"] <= bands["loose"]
    ):
        raise ValueError("spread requires ordered upper bands")
    return p


def spread(ctx, params, deadline):
    from collections import Counter

    deadline.check()
    rows = [
        row
        for ref in params["requests"]
        for row in ctx.resource(ref, "requests").snapshot_records()
    ]
    if not rows or len({row["wire_request_id"] for row in rows}) != len(rows):
        raise ValueError("spread cohort empty or duplicated")
    if any(
        row.get("consumer_completion_verified") is not True
        or not isinstance(row.get("prefill_addr"), str)
        or not row["prefill_addr"]
        for row in rows
    ):
        raise ValueError("spread cohort lacks terminal or landing evidence")
    distribution = Counter(row["prefill_addr"] for row in rows)
    maximum = max(distribution.values()) / len(rows)
    complete = len(rows) >= params["min_samples"] and all(
        request_success(row) for row in rows
    )
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    passed = report.check("P1", maximum, bands=params["bands"])
    evidence = dict(
        complete=True,
        sample_count=len(rows),
        min_samples=params["min_samples"],
        records=rows,
        distribution=dict(distribution),
        grade=report.run_grade,
        achieved=report.achieved,
        bands=params["bands"],
    )
    return StageOutput(
        {"max_share": maximum},
        [
            CheckResult(
                "P6",
                "PASS" if complete else "FAIL",
                actual=complete,
                expected=True,
                evidence=evidence,
            ),
            CheckResult(
                "P1",
                "PASS" if passed and len(rows) >= params["min_samples"] else "FAIL",
                actual=maximum,
                expected=params["bands"][report.run_grade],
                evidence=evidence,
            ),
        ],
        [_artifact(ctx, "kv-spread", evidence)],
    )


def _membership_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "engine", "keys", "relation"},
        {"snapshot", "engine", "keys", "relation"},
    )
    plan.reference(p["snapshot"], "kv_snapshot")
    _engine(p["engine"], plan)
    _keys(p["keys"])
    if p["relation"] not in ("all", "none", "exact"):
        raise ValueError("unknown key membership relation")
    return p


def membership(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "kv_snapshot")
    name = _target(ctx, params["engine"])
    actual = set(data["engines"][name]["cache_key_set"])
    wanted = set(params["keys"])
    passed = (
        wanted <= actual
        if params["relation"] == "all"
        else (not wanted & actual if params["relation"] == "none" else wanted == actual)
    )
    evidence = dict(
        complete=True,
        sample_count=1,
        min_samples=1,
        snapshot=data,
        engine=name,
        requested_keys=params["keys"],
    )
    return StageOutput(
        {"matched": passed},
        [
            CheckResult(
                "key_membership",
                "PASS" if passed else "FAIL",
                actual=sorted(actual & wanted),
                expected=params["relation"],
                evidence=evidence,
            )
        ],
    )


def _holders_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "keys", "holders", "match"},
        {"snapshot", "keys", "holders", "match"},
    )
    plan.reference(p["snapshot"], "kv_snapshot")
    _keys(p["keys"])
    if not isinstance(p["holders"], list) or len(p["holders"]) > 32:
        raise ValueError("holders must be a bounded explicit engine list")
    for value in p["holders"]:
        _engine(value, plan)
    if p["match"] not in ("full_family", "any_key"):
        raise ValueError("holder match must be full_family or any_key")
    return p


def holders(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "kv_snapshot")
    expected = [_target(ctx, value) for value in params["holders"]]
    if len(set(expected)) != len(expected) or not set(expected) <= set(data["engines"]):
        raise ValueError("holder expectation must name distinct observed engines")
    wanted = set(params["keys"])
    actual = sorted(
        name
        for name, row in data["engines"].items()
        if (
            wanted <= set(row["cache_key_set"])
            if params["match"] == "full_family"
            else bool(wanted & set(row["cache_key_set"]))
        )
    )
    evidence = dict(
        complete=True,
        sample_count=len(data["engines"]),
        min_samples=len(data["engines"]),
        snapshot=data,
        keys=params["keys"],
        match=params["match"],
    )
    return StageOutput(
        {"matched": actual == sorted(expected)},
        [
            CheckResult(
                "holder_set",
                "PASS" if actual == sorted(expected) else "FAIL",
                actual=actual,
                expected=sorted(expected),
                evidence=evidence,
            )
        ],
    )


def _union_validate(params, plan):
    p = _fields(
        params,
        {"requests", "holders", "min_samples", "min_used"},
        {"requests", "holders", "min_samples", "min_used"},
    )
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 100:
        raise ValueError("holder union requires bounded request cohorts")
    for value in p["requests"]:
        plan.reference(value, "requests")
    if not isinstance(p["holders"], list) or not 1 <= len(p["holders"]) <= 32:
        raise ValueError("holder union requires explicit holders")
    for value in p["holders"]:
        _engine(value, plan)
    if (
        type(p["min_samples"]) is not int
        or p["min_samples"] < 1
        or type(p["min_used"]) is not int
        or not 1 <= p["min_used"] <= len(p["holders"])
    ):
        raise ValueError("invalid holder union sample or worker floor")
    return p


def union(ctx, params, deadline):
    names = [_target(ctx, value) for value in params["holders"]]
    if len(set(names)) != len(names):
        raise ValueError("holder union contains duplicate engine identity")
    engines = _engines(_http(ctx.ops, "snapshot", deadline), names)
    rows = [
        row
        for ref in params["requests"]
        for row in ctx.resource(ref, "requests").snapshot_records()
    ]
    if (
        not rows
        or len({row["wire_request_id"] for row in rows}) != len(rows)
        or any(
            row.get("consumer_completion_verified") is not True
            or not isinstance(row.get("prefill_addr"), str)
            or not row["prefill_addr"]
            for row in rows
        )
    ):
        raise ValueError("holder union lacks distinct terminal landing evidence")
    addresses = {row["grpc_addr"] for row in engines.values()}
    used = {row["prefill_addr"] for row in rows}
    complete = len(rows) >= params["min_samples"] and all(
        request_success(row) for row in rows
    )
    evidence = dict(
        complete=True,
        sample_count=len(rows),
        min_samples=params["min_samples"],
        records=rows,
        engines=engines,
    )
    checks = [
        CheckResult(
            "P2",
            "PASS" if complete and len(used) >= params["min_used"] else "FAIL",
            actual=len(used),
            expected=params["min_used"],
            evidence=evidence,
        ),
        CheckResult(
            "P6",
            "PASS" if complete and used <= addresses else "FAIL",
            actual=sorted(used),
            expected=sorted(addresses),
            evidence=evidence,
        ),
    ]
    return StageOutput(
        {"used": len(used)}, checks, [_artifact(ctx, "kv-holder-union", evidence)]
    )


def _alive_validate(params, plan):
    p = _fields(params, {"role", "count"}, {"role", "count"})
    if (
        p["role"] not in ("PREFILL", "DECODE")
        or type(p["count"]) is not int
        or not 0 <= p["count"] <= 32
    ):
        raise ValueError("master alive requires a role and bounded exact count")
    return p


def master_alive(ctx, params, deadline):
    from ...harness import http_post_json

    samples = []
    try:
        while True:
            deadline.check()
            status, body = http_post_json(
                f"http://127.0.0.1:{ctx.ops.master_http_port}/rtp_llm/master/info",
                {},
                timeout=min(2, deadline.remaining()),
            )
            samples.append(dict(sampled_s=ctx.clock(), status=status, body=body))
            count = (
                body.get("worker_summary", {}).get(params["role"], {}).get("alive")
                if isinstance(body, dict)
                else None
            )
            if status != 200 or type(count) is not int or count < 0:
                raise ValueError("master alive count lacks typed observation")
            if count == params["count"]:
                break
            deadline.sleep(0.5)
    finally:
        path = _artifact(ctx, "kv-master-alive", samples)
    return StageOutput(
        {"count": count},
        [
            CheckResult(
                "master_alive",
                "PASS",
                actual=count,
                expected=params["count"],
                evidence=dict(
                    complete=True,
                    sample_count=len(samples),
                    min_samples=1,
                    samples=samples,
                ),
            )
        ],
        [path],
    )


HANDLERS = [
    StageHandler(
        "kv_holder_share_check",
        _holder_share_validate,
        holder_share,
        {"share": "number"},
        checks=frozenset({"P6", "M2"}),
    ),
    StageHandler(
        "kv_off_holder_check",
        _off_holder_validate,
        off_holder,
        {"count": "integer"},
        checks=frozenset({"P2", "P6"}),
    ),
    StageHandler(
        "kv_holders_check",
        _holders_validate,
        holders,
        {"matched": "boolean"},
        checks=frozenset({"holder_set"}),
    ),
    StageHandler(
        "kv_union_check",
        _union_validate,
        union,
        {"used": "integer"},
        checks=frozenset({"P2", "P6"}),
    ),
    StageHandler(
        "kv_master_alive",
        _alive_validate,
        master_alive,
        {"count": "integer"},
        checks=frozenset({"master_alive"}),
    ),
    StageHandler(
        "kv_same",
        _distinct_validate,
        same,
        {"same": "boolean"},
        checks=frozenset({"same_holder"}),
    ),
    StageHandler(
        "kv_fidelity_check",
        _affinity_validate,
        fidelity,
        {"share": "number"},
        checks=frozenset({"P6", "P9"}),
    ),
    StageHandler(
        "kv_spread_check",
        _spread_validate,
        spread,
        {"max_share": "number"},
        checks=frozenset({"P6", "P1"}),
    ),
    StageHandler(
        "kv_membership_check",
        _membership_validate,
        membership,
        {"matched": "boolean"},
        checks=frozenset({"key_membership"}),
    ),
    StageHandler(
        "kv_snapshot", _snapshot_validate, snapshot, {"snapshot": "kv_snapshot"}
    ),
    StageHandler("kv_landing", _landing_validate, landing, {"engine": "string"}),
    StageHandler(
        "kv_distinct",
        _distinct_validate,
        distinct,
        {"distinct": "boolean"},
        checks=frozenset({"distinct_holders"}),
    ),
    StageHandler(
        "kv_evict",
        _evict_validate,
        evict,
        {"snapshot": "kv_snapshot"},
        checks=frozenset({"evicted"}),
    ),
    StageHandler(
        "kv_prefix_check",
        _prefix_validate,
        prefix,
        {"run": "integer"},
        checks=frozenset({"contiguous_prefix"}),
    ),
    StageHandler(
        "kv_affinity_check",
        _affinity_validate,
        affinity,
        {"share": "number"},
        checks=frozenset({"P6", "M3"}),
    ),
]
