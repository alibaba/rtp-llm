"""Audit the full MTP smoke against current shared-engine execution events."""

import argparse
from bisect import bisect_left
import json
import hashlib
from pathlib import Path
import re

if __package__:
    from . import audit_k31_last_n, k31_last_n_precision
    from .k31_smoke_capacity import device_pool_usable_blocks
    from . import result_check
    from .k31_full_smoke import cache_pressure_plan, decode_batch_sizes, request_budget
    from .smoke_request_identity import resolve as resolve_request_identity
else:
    import audit_k31_last_n
    import k31_last_n_precision
    from k31_smoke_capacity import device_pool_usable_blocks
    import result_check
    from k31_full_smoke import cache_pressure_plan, decode_batch_sizes, request_budget
    from smoke_request_identity import resolve as resolve_request_identity


MARKER = "[K3_SMOKE_EVENT] "
GRAPH_ROLE = {"mtp_target_verify_forward": 2, "mtp_draft_decode_forward": 1,
              "mtp_draft_update_forward": 4}


class IndexedEvents(tuple):
    """Immutable log snapshot; avoid rescanning an idle DP owner's history."""
    def __new__(cls, rows):
        return super().__new__(cls, rows)

    def __init__(self, rows):
        self.groups, self.replays, self.requests = {}, {}, {}
        self.forwards = []
        for row in self:
            event = row.get("event")
            if "execution_step" in row:
                self.groups.setdefault((event, row["execution_step"]), []).append(row)
                self.forwards.append(row)
            if event == "cuda_graph_replay":
                self.replays.setdefault(row["role"], []).append(row)
            if "request_id" in row:
                self.requests.setdefault((event, row["request_id"]), []).append(row)
        self.forward_times = [row["time_ns"] for row in self.forwards]
        self.replay_times = {role: [row["time_ns"] for row in events]
                             for role, events in self.replays.items()}


def request_events(rows, event, request_id):
    if isinstance(rows, IndexedEvents):
        return rows.requests.get((event, request_id), ())
    return [row for row in rows if row.get("event") == event
            and row.get("request_id") == request_id]


def read_events(directory, engine, ranks=range(8)):
    events = {rank: [] for rank in ranks}
    for rank in ranks:
        path = Path(directory) / f"main_{rank}.log"
        for line in path.read_text(errors="replace").splitlines():
            if MARKER in line:
                events[rank].append(json.loads(line.split(MARKER, 1)[1]))
    for line in Path(engine).read_text(errors="replace").splitlines():
        if MARKER not in line:
            continue
        match = re.search(r"\[RANK (\d+)\]", line.split(MARKER, 1)[0])
        if match is None or int(match[1]) not in events:
            raise ValueError("native evidence lacks an actual rank")
        events[int(match[1])].append(json.loads(line.split(MARKER, 1)[1]))
    for rows in events.values():
        if any(type(row.get("time_ns")) is not int for row in rows):
            raise ValueError("execution evidence lacks timestamps")
        rows.sort(key=lambda row: row["time_ns"])
    return {rank: IndexedEvents(rows) for rank, rows in events.items()}


def group_evidence(events, root, tp_size):
    """Root owns request identities; every TP rank must execute that group."""
    owner = root["dp_rank"]
    expected = root["request_ids"]
    if (not root.get("logical_authoritative") or root.get("fake")
            or not expected or len(set(expected)) != len(expected)
            or root.get("logical_batch") != len(expected)):
        raise ValueError("non-authoritative, fake or inconsistent request group")
    for rank in range(owner * tp_size, (owner + 1) * tp_size):
        rows = events[rank]
        matches = (rows.groups.get((root["event"], root["execution_step"]), ())
                   if isinstance(rows, IndexedEvents) else
                   [row for row in rows if row.get("event") == root["event"]
                    and row.get("execution_step") == root["execution_step"]])
        if len(matches) != 1:
            raise ValueError(f"rank{rank}: actual group execution missing or duplicated")
        row = matches[0]
        if (row.get("fake") or row.get("dp_rank") != owner
                or row.get("world_rank") != rank or row.get("tp_rank") != rank % tp_size
                or row.get("input_rows") != root["input_rows"]
                or row.get("token_rows") != root["token_rows"]):
            raise ValueError(f"rank{rank}: synchronized input geometry differs")
    return expected


def replay_evidence(rows, forward):
    role = GRAPH_ROLE[forward["event"]]
    if isinstance(rows, IndexedEvents):
        begin = bisect_left(rows.forward_times, forward["time_ns"])
        end = next((rows.forwards[i]["time_ns"] for i in range(begin, len(rows.forwards))
                    if rows.forwards[i]["execution_step"] > forward["execution_step"]), None)
        times = rows.replay_times.get(role, ())
        left = bisect_left(times, forward["time_ns"])
        right = len(times) if end is None else bisect_left(times, end)
        replays = rows.replays.get(role, ())[left:right]
    else:
        after = [row for row in rows if row["time_ns"] >= forward["time_ns"]]
        end = next((row["time_ns"] for row in after
                    if row.get("execution_step", -1) > forward["execution_step"]), None)
        replays = [row for row in after if row.get("event") == "cuda_graph_replay"
                   and row.get("role") == role
                   and (end is None or row["time_ns"] < end)]
    if not replays or any(row["bucket"] < row["real_batch"] for row in replays):
        raise ValueError(f"{forward['event']}: actual Graph replay missing")


def host_cancel_evidence(rows):
    """Require a native pending observation after the Cancel latch is installed.

    The weak RPC acknowledgement precedes downstream joining and scheduler
    cleanup. That later cleanup may legitimately see an already-finished load.
    A start line without a completion line alone is insufficient evidence.
    """
    kinds = {"host_cache_cancelled_during_load",
             "host_cache_priority_cancel_requested_while_pending"}
    for cancel in rows:
        if cancel.get("event") not in kinds or cancel.get("host_reuse_len", 0) <= 0:
            continue
        starts = request_events(rows, "host_cache_load_started", cancel["request_id"])
        done = request_events(rows, "host_cache_load_done", cancel["request_id"])
        if (any(e["time_ns"] <= cancel["time_ns"] for e in starts)
                and not any(e["time_ns"] <= cancel["time_ns"] for e in done)):
            return cancel
    raise ValueError("no native cancellation observation while actual Host load remained pending")


def large_batch_evidence(stages, case_ids, decode, tp_size):
    """Prove the complete cohort, including both owners at one native step."""
    proofs = []
    for stage in stages:
        match = re.fullmatch(r"orthogonal_decode_\d{2}_batch_(63|64)", stage["name"])
        if match is None:
            continue
        size = int(match[1])
        cohort = [case_ids[name] for name in stage["case_names"]]
        if len(cohort) != size or any(len(ids) != 1 for ids in cohort):
            raise ValueError("large batch requires explicit native request identities")
        owners = 8 // tp_size
        by_owner = [{ids[0] for index, ids in enumerate(cohort) if index % owners == owner}
                    for owner in range(owners)]
        expected_sizes = ([32, 31] if size == 63 else [32, 32]) if owners == 2 else [size]
        if list(map(len, by_owner)) != expected_sizes:
            raise ValueError("large batch owner distribution differs")
        steps = []
        for owner, expected in enumerate(by_owner):
            steps.append({e["execution_step"] for e in decode[owner * tp_size]
                          if e.get("event") == "mtp_target_verify_forward"
                          and not e.get("fake") and set(e.get("request_ids", ())) == expected})
        common = set.intersection(*steps)
        if not common:
            raise ValueError(f"BS{size}: no same-step complete owner cohorts")
        proofs.append(dict(batch=size, owner_batches=expected_sizes,
                           matching_cycles=len(common), witness_step=min(common)))
    return proofs


def audit(result, prefill, decode, tp_size, *, prefill_full_pages=None, last_n_precision=None):
    errors, proofs = [], {}
    declared_profile = result.get("validation_profile", "smoke")
    last5 = declared_profile == 'last5-chunked-smoke'
    profile = 'smoke' if last5 else declared_profile
    if last5:
        geometry = result.get('geometry', {})
        page = geometry.get('page_size')
        if (tp_size != 8 or page not in (64, 128) or result.get('last_n_checkpoints') != 5
                or result.get('scheduler_chunk_profile') is not True
                or any(geometry.get(key) != expected for key, expected in dict(
                    kernel_page_size=page, source_shards=8, checkpoint_span=8*page,
                    reuse_cache_tail_blocks=5, runtime_layers=93, runtime_verified=True).items())):
            raise ValueError('Last-5 result lacks verified B64 or B128, CP8, TP8 and full93 geometry')
    rows = result.get("cases", [])
    plan = result.get("cache_pressure_plan")
    if profile == "smoke" and plan is not None and "requests" in plan:
        expected = cache_pressure_plan(prefill_full_pages, result["reuse_unit_tokens"], scheduler_chunking=last5)
        if plan != expected:
            raise ValueError("cache pressure plan does not match the runtime pool")
        probes = [row for row in rows if row["name"].startswith("tier-pressure-")]
        if ({row["name"] for row in probes} !=
                {f"tier-pressure-{i:02d}" for i in range(plan["requests"])}
                or any(row.get("phase") != "preparation"
                       or row.get("input_len") != plan["input_tokens"] for row in probes)):
            raise ValueError("cache pressure preparation coverage differs from its plan")
    budget = request_budget(profile, plan.get("requests", 15) if plan else 15)
    counts = {"formal": sum(row.get("phase") == "formal" for row in rows),
              "preparation": sum(row.get("phase") == "preparation" for row in rows)}
    if (not result.get("passed") or result.get("target_only") is not False
            or counts != {key: budget[key] for key in ("formal", "preparation")}
            or len({row["name"] for row in rows}) != budget["total"]):
        raise ValueError(f"{profile} MTP functional result is incomplete")
    stages = result["stages"]
    if any(not stage.get("passed") or stage.get("skipped") for stage in stages):
        raise ValueError("unfinished required smoke stage")
    stage_names = {stage["name"] for stage in stages}
    required = {f"orthogonal_decode_{i:02d}_batch_{size}" for i, size in
                enumerate(decode_batch_sizes(profile))}
    if profile == "smoke":
        required |= {f"batch_shape_{i}_{size}" for i, size in
                     enumerate((1, 2, 3, 7, 8, 9, 3, 1, 9))}
        required |= {"rolling_refill", "historical_four_squares", "orthogonal_kv_final",
                     "orthogonal_cancel_recovery", "prefix_AB_mixed"}
    if not required <= stage_names:
        raise ValueError(f"missing full reference stages:{sorted(required-stage_names)}")
    if profile == "smoke" and not any(stage.get("cache_tier_checks")
               and all(stage["cache_tier_checks"].values()) for stage in stages):
        raise ValueError("Device/Host/partial/miss mixed-forward proof absent")

    explicit = {name: rid for stage in stages if stage.get("transport") == "pd_group_rpc"
                for name, rid in stage.get("request_ids", {}).items()}
    raw = {}
    for row in rows:
        name = row["name"]
        item = dict(result=row, passed=True, transport="http")
        if name in explicit or "request_id" in row:
            rid = explicit.get(name, row.get("request_id"))
            if "request_id" in row and row["request_id"] != rid:
                raise ValueError("summary and grouped native request IDs disagree")
            item.update(transport="pd_group_rpc", request_id=rid)
        raw[name] = item
        errors.extend(result_check.independent_answer_check(row, scheduler_chunking=last5))
        errors.extend(result_check._cache_errors(row, result["block_size"]))
    try:
        request_ids, methods, barriers = resolve_request_identity(
            stages, raw, prefill, decode, tp_size=tp_size)
    except AssertionError as error:
        raise ValueError(f"native request identity audit failed: {error}") from error
    proofs["request_identity"] = dict(methods=methods, stage_barriers=barriers)

    prefill_roots = [e for e in prefill[0]
                     if e.get("event") == "mtp_target_prefill_forward" and not e.get("fake")]
    if last5 and any(not e.get('context_lengths') or any(q > 65536 for q in e['context_lengths'])
                     for e in prefill_roots):
        raise ValueError('Last-5 scheduled Prefill exceeds the actual 64K chunk budget')
    seen_prefill = set()
    for root in prefill_roots:
        seen_prefill.update(group_evidence(prefill, root, 8))
    if not {rid for ids in request_ids.values() for rid in ids} <= seen_prefill:
        raise ValueError("not every request has an all8-rank Prefill forward")
    for stage in stages:
        if (stage["name"].startswith("batch_shape_")
                or stage["name"].startswith("orthogonal_cache_mixed_")
                or stage["name"] == "prefix_AB_mixed"):
            cohort = {rid for name in stage["case_names"] for rid in request_ids[name]}
            if not any(set(e["request_ids"]) == cohort for e in prefill_roots):
                raise ValueError(f"{stage['name']}: requests did not share one actual forward")

    verified = {owner: set() for owner in range(8 // tp_size)}
    batches = {owner: set() for owner in verified}
    for owner in verified:
        roots = [e for e in decode[owner * tp_size]
                 if e.get("event") == "mtp_target_verify_forward" and not e.get("fake")]
        for root in roots:
            verified[owner].update(group_evidence(decode, root, tp_size))
            batches[owner].add(root["logical_batch"])
        for rank in range(owner * tp_size, (owner + 1) * tp_size):
            for event in decode[rank]:
                if event.get("event") in GRAPH_ROLE and not event.get("fake"):
                    replay_evidence(decode[rank], event)
            for phase in GRAPH_ROLE:
                if not any(e.get("event") == phase and not e.get("fake") for e in decode[rank]):
                    raise ValueError(f"rank{rank}: {phase} did not execute")
    for row in rows:
        candidates, owner = request_ids[row["name"]], row["decode_owner_rank"]
        if row["output_len"] > 1 and not set(candidates) <= verified[owner]:
            raise ValueError(f"{row['name']}: actual owner verify missing")
        for rid in candidates:
            for rank in range(8):
                loaded = request_events(decode[rank], "pd_cache_loaded", rid)
                if rank // tp_size != owner:
                    if loaded:
                        raise ValueError(f"{row['name']}: cache reached another owner")
                    continue
                transfers = request_events(decode[rank], "pd_transfer_submitted", rid)
                if not loaded or {e["source_tp_rank"] for e in transfers} != set(range(8)):
                    raise ValueError(f"{row['name']}: owner-local load/eight-source plan incomplete")
                for e in transfers:
                    if (e["source_tp"] != 8 or e["destination_tp"] != tp_size
                            or e["dp_rank"] != owner or e["attn_tp_rank"] != rank % tp_size):
                        raise ValueError("PD source/head/owner dimensions mixed")
                    if e["mla_pages"] and (e["first_mla_page"] % 8 != e["source_tp_rank"]
                                           or e["last_mla_page"] % 8 != e["source_tp_rank"]):
                        raise ValueError("PageRR cache loaded from the wrong owner")
    proofs["large_batches"] = large_batch_evidence(stages, request_ids, decode, tp_size)
    if tp_size == 8 and not set(decode_batch_sizes(profile)) <= batches[0]:
        raise ValueError(f"actual Decode {profile} batch coverage incomplete")
    owner_batches = ({0: {1,2,4,32}, 1: {2,4,31}} if profile == "smoke"
                     else {0: {32}, 1: {32}})
    if tp_size == 4 and any(not expected <= batches[owner]
                            for owner, expected in owner_batches.items()):
        raise ValueError(f"actual DP2 {profile} owner batches incomplete")

    for rank in range(8) if profile == "smoke" else ():
        for phase in ("indexer", "sfa"):
            chunks = [e for e in prefill[rank] if e.get("event") == "dsa_kv_chunks_completed"
                      and e.get("phase") == phase and e.get("fp8_indexer")]
            if not chunks or not any(e["completed_chunks"] >= 2 for e in chunks):
                raise ValueError(f"rank{rank}: actual target {phase} multi-chunk execution missing")
            if any(not e["query_cp"] or e["shard_size"] != 8 for e in chunks):
                raise ValueError("target Prefill query CP/source-shard policy differs")
    if profile == "smoke":
        cancel = host_cancel_evidence(prefill[0])
        proofs["host_cancel_while_actual_load_pending"] = cancel["request_id"]
    if last5:
        proofs['last_n'] = audit_last5(result, prefill, request_ids, last_n_precision)
    return {"passed": not errors, "errors": errors, "proofs": proofs,
            "counts": counts, "decode_batches": {str(k):sorted(v) for k,v in batches.items()},
            "validation_profile": declared_profile, "request_budget": budget,
            "complete_smoke_passed": not errors and profile == "smoke",
            "native_K31_MTP": bool(result.get("draft_contract") and
                result["draft_contract"].get("kind") == "native_k3_mla_mtp"),
            "draft_contract": result.get("draft_contract"),
            "golden_alignment": "pending"}


def audit_last5(result, prefill, request_ids, precision):
    def ids(name):
        values = request_ids.get(name, [])
        if len(values) != 1:
            raise ValueError(f'{name}: Last-5 proof requires one actual native request identity')
        return values

    geometry = result['geometry']
    span = geometry['checkpoint_span']
    c7 = 7 * span
    if (result.get('last_n_c7_fixture') or {}).get('common_tokens') != c7:
        raise ValueError('C7 negative control does not declare its exact shared token boundary')
    rows = {row['name']: row for row in result['cases']}
    c7_inputs = [rows[name].get('input_tokens') for name in ('prefix-A-seed', 'prefix-B-partial')]
    if (any(not isinstance(tokens, list) or len(tokens) != 12 * span + 128 for tokens in c7_inputs)
            or c7_inputs[0][:c7] != c7_inputs[1][:c7]
            or c7_inputs[0][c7] == c7_inputs[1][c7]):
        raise ValueError('C7 negative control lacks actual identical-prefix / divergent-suffix input token IDs')
    for field, tokens in zip(('seed_token_sha256', 'probe_token_sha256'), c7_inputs):
        if hashlib.sha256(json.dumps(tokens).encode()).hexdigest() != result['last_n_c7_fixture'].get(field):
            raise ValueError('C7 actual input token IDs differ from its declared fixture')
    proof = dict(lifecycle={}, tiers={}, c7={})
    # TP0 owns allocation, matching and publication. Execution and conv input
    # snapshots still require evidence from every TP rank; cache settlement
    # follows MultiRankTransferAsyncContext's all-worker status merge.
    for rank in (0,):
        proof['lifecycle'][rank] = audit_k31_last_n.audit_prefill_states(
            prefill, root_rank=rank, page_tokens=geometry['page_size'], checkpoint_tokens=span)
        events = {0: prefill[rank]}
        proof['tiers'][rank] = {
            tier: audit_k31_last_n.audit_checkpoint_reuse(
                events, ids('tier-memory-seed-0'), ids(name), tier, checkpoint_span=span)
            for tier, name in (('DEVICE', 'tier-device-probe-0'),
                               ('HOST', 'tier-mixed-0-memory'), ('DISK', 'tier-memory-probe'))}
        proof['c7'][rank] = audit_k31_last_n.audit_c7_miss(
            events, ids('prefix-A-seed'), ids('prefix-B-partial'), checkpoint_span=span)
    if precision is None:
        raise ValueError('Last-5 requires actual P/D logits and a pre-locked precision contract')
    proof['precision'] = k31_last_n_precision.audit_cases(result, request_ids, *precision)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("result", "prefill-logs", "prefill-engine", "decode-logs", "decode-engine", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--decode-tp", type=int, choices=(4,8), required=True)
    for key in ('prefill-logits', 'decode-logits', 'precision-reference', 'precision-contract'):
        parser.add_argument('--'+key, type=Path)
    args = parser.parse_args()
    try:
        result = json.loads(args.result.read_text())
        plan = result.get("cache_pressure_plan") or {}
        prefill_full_pages = None
        if "requests" in plan:
            prefill_full_pages = device_pool_usable_blocks(
                args.prefill_engine.read_text(errors="replace"), "full")
        precision_args = (args.prefill_logits, args.decode_logits, args.precision_reference, args.precision_contract)
        if any(value is not None for value in precision_args) and not all(value is not None for value in precision_args):
            raise ValueError('precision capture directories, reference and locked contract must be supplied together')
        precision = ((args.prefill_logits, args.decode_logits,
                      k31_last_n_precision.read_descriptor(args.precision_reference),
                      json.loads(args.precision_contract.read_text())) if args.precision_contract is not None else None)
        proof = audit(result, read_events(args.prefill_logs,args.prefill_engine),
                      read_events(args.decode_logs,args.decode_engine), args.decode_tp,
                      prefill_full_pages=prefill_full_pages, last_n_precision=precision)
        rows = {row["name"]:row for row in result["cases"]}
        proof["errors"].extend(result_check.audit_raw_answers(args.result.parent/"requests",rows))
        proof["errors"].extend(result_check._raw_consistency(args.result.parent/"requests",rows))
        proof["passed"] = not proof["errors"]
        proof["complete_smoke_passed"] = proof["passed"] and proof["validation_profile"] in ('smoke', 'last5-chunked-smoke')
    except (OSError, ValueError, KeyError, TypeError) as error:
        proof = {"passed": False, "errors": [f"{type(error).__name__}: {error}"]}
    with args.output.open("x") as output:
        json.dump(proof,output,ensure_ascii=False,indent=2)
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
