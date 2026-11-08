"""Native31 workload evidence on the shared CP8-to-DP8 DSpARK service."""

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import re

from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_dspark_modes import (
    validate_decode_commit_layout,
)


POINTS = (129, 257, 1025, 8192, 32768)
REUSE_POINTS = (129, 257, 513, 1025)
REUSE_WARM = (19, 20, 21, 22, 23, 24, 25, 26, 59, 63, 67, 71, 75, 79, 83, 87)
CP_REUSE_PAGE_TOKENS = 2048
CP_REUSE_PROBE_TOKENS = CP_REUSE_PAGE_TOKENS + 257
CP_REUSE_PROBE_REQUESTS = 17
OUTPUT_TOKENS = 8
SEQ_CAPACITY = 65536
REQUESTS = 98
POOL_TAGS = {"csa_kv", "hca_kv", "indexer_kv"}
CACHE_FIELDS = ("positions", "sha256", "bytes", "nonzero_bytes")


def admission_capacity(log, role, gamma):
    """Bind the original workload's headroom to actually initialized pools."""
    text = log.read_text(errors="replace")
    if f"max_seq_len: {SEQ_CAPACITY}" not in text or "seq_size_per_block=256" not in text:
        return None
    metrics = {}
    for rank, tag, total, free, reserve in re.findall(
        r"\[RANK (\d+)\][^\n]*kvc raw pool\[DEVICE/([^\]]+)\]: [^\n]*?"
        r"\btotal=(\d+) free=(\d+) [^\n]*?\breserve=(\d+)\b",
        text,
    ):
        if tag in POOL_TAGS:
            metrics[(int(rank), tag)] = dict(total=int(total), free=int(free), reserve=int(reserve))
    ranks = range(8) if role == "decode" else (0,)
    keys = {(rank, tag) for rank in ranks for tag in POOL_TAGS}
    if not keys <= metrics.keys():
        return None
    block = 256
    headroom = max(POINTS) + 256 + gamma + 1
    worst_wave = sum(
        math.ceil((POINTS[index % len(POINTS)]
                   + (max(32, OUTPUT_TOKENS * 4) if index % 3 == 0 else gamma + 2)
                   + gamma + 1) / block)
        for index in range(16)
    )
    assert SEQ_CAPACITY >= headroom
    assert all(metrics[key]["free"] - metrics[key]["reserve"] >= worst_wave for key in keys)
    return {
        "role": role, "max_seq_len": SEQ_CAPACITY, "required_tokens": headroom,
        "full_block_span": block, "required_full_blocks": worst_wave,
        "full_pools": {f"{rank}/{tag}": metrics[(rank, tag)] for rank, tag in sorted(keys)},
    }


def _domains(state, row):
    return {name.rsplit(":", 1)[0]: value for name, value in state.items()
            if name.rsplit(":", 1)[1] == str(row)}


def _signature(domains):
    assert domains and any(value["nonzero_bytes"] for value in domains.values())
    canonical = {name: [value[field] for field in CACHE_FIELDS] for name, value in domains.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _assert_source_fragments(shards):
    assert set(shards) == set(range(8))
    canonical = shards[0]
    assert all(state.keys() == canonical.keys() for state in shards.values())
    for key, value in canonical.items():
        width, remainder = divmod(value["bytes"], len(value["positions"]))
        assert width > 0 and remainder == 0
        fragments = {position: [] for position in value["positions"]}
        for rank, state in shards.items():
            item = state[key]
            assert all(item[field] == value[field] for field in CACHE_FIELDS)
            owner = item["source_owner"]
            assert (owner["cp_rank"], owner["cp_size"], owner["matches_canonical"]) == (rank, 8, True)
            total = 0
            for fragment in owner["fragments"]:
                start, end = fragment["begin"], fragment["end"]
                assert 0 <= start < end <= width
                fragments[fragment["position"]].append((start, end))
                total += end - start
            assert total == owner["bytes"]
        for spans in fragments.values():
            position = 0
            for start, end in sorted(spans):
                assert start == position
                position = end
            assert position == width
    return _signature(canonical)


def validate_prefill(reports, drafts, cases, *, reuse=False, reuse_page_probe=False, output=None):
    target = defaultdict(dict)
    draft = {}
    for row in drafts:
        assert not row["idle"] and row["commit_only"] and row["event"] == "commit"
        assert row["executed_layers"] == [0, 1, 2] and row["feature_finite"]
        assert row["native_aliases"] and row["output_finite"] and row["native_writes"]
        key = row["rank"], row["call"]
        assert key not in draft
        draft[key] = row
    for row in reports:
        assert not row["idle"] and row["prefill"] and row["output_finite"]
        assert row["yoco"]["executed_layers"] == list(range(40))
        assert row["cache_protocol"]["source_layout"] == "per-tag-native-cp-sharded"
        assert row["profile"]["mode"] == ("compact" if reuse else "full") and row["yoco"]["cp_size"] == 8
        assert row["cache_placement"] and all(item["sharded"] for item in row["cache_placement"])
        rank, call = row["rank"], row["call"]
        assert rank not in target[call]
        target[call][rank] = row
    expected = CP_REUSE_PROBE_REQUESTS if reuse_page_probe else 89 if reuse else REQUESTS
    assert len(target) == len(cases) == expected
    handoffs, by_request = [], {}
    for call, shards in target.items():
        length, = shards[0]["row_layout"]["lengths"]
        commits = {rank: draft.pop((rank, call)) for rank in shards}
        for rank, row in shards.items():
            assert row["row_layout"]["lengths"] == [length] and row["row_layout"]["prefixes"] == [0]
            assert commits[rank]["input_feature_sha256"] == row["aux_feature_sha256"]
        handoffs.append([
            length,
            _assert_source_fragments({rank: _domains(row["native_state_after"], 0) for rank, row in shards.items()}),
            _assert_source_fragments({rank: _domains(row["native_state"], 0) for rank, row in commits.items()}),
        ])
        if reuse:
            request_id, = shards[0]["request_ids"]
            assert all(row["request_ids"] == [request_id] for row in shards.values())
            assert request_id not in by_request
            prefix, = commits[0]["prefixes"]
            assert all(item["prefixes"] == [prefix] and item["lengths"] == [length - prefix] for item in commits.values())
            by_request[request_id] = {"handoff": handoffs[-1], "prefix": prefix,
                                      "proof": reuse_proof(shards, commits, length, prefix)}
    assert not draft
    assert Counter(item[0] for item in handoffs) == Counter(case["aux_info"]["input_len"] for case in cases)
    result = {"requests": len(cases), "cp_ranks": list(range(8)), "exact_aux_handoff": True, "handoffs": handoffs}
    if reuse:
        fixed_inputs = {}
        fixed_indices = range(CP_REUSE_PROBE_REQUESTS) if reuse_page_probe else (0, *REUSE_WARM)
        for index in fixed_indices:
            case = cases[index]
            payload = json.loads((output / "contract-http" / f"request-{index}.http.txt").read_text())
            tokens, = payload["input_ids"]
            assert len(tokens) == case["aux_info"]["input_len"]
            fixed_inputs[case["trace_id"]] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        result.update(handoffs_by_request=by_request, fixed_inputs=fixed_inputs)
    return result


def reuse_proof(shards, commits, length, prefix):
    if not prefix:
        assert all(row["reuse_prefix"] is None for row in shards.values())
        return None
    history = {}
    before, after = {}, {}
    for rank, row in shards.items():
        yoco = row["yoco"]
        assert yoco["encoder_append_starts"] == {"0": prefix}
        assert row["immutable_global_prefix"] == "EXACT_BYTES_UNCHANGED"
        assert row["aux_feature_rows"] == commits[rank]["feature_rows"]
        before[rank] = _domains(row["reuse_prefix"]["before"], 0)
        after[rank] = _domains(row["reuse_prefix"]["after"], 0)
        assert before[rank] == after[rank]
        history_row = row["replay_history"]
        assert len(history_row["positions"]) == len(history_row["tokens"])
        for position, token in zip(history_row["positions"], history_row["tokens"]):
            assert position not in history
            history[position] = token
    assert set(history) == set(range(length))
    for name, row in before[0].items():
        tag = name.rsplit(":", 1)[1]
        assert tag in ("global_kv_1", "global_kv_2", "indexer_kv")
        positions = row["positions"]
        stride = positions[1] - positions[0]
        assert stride in (1, 2) and positions == list(range(0, prefix // stride * stride, stride))
    assert _assert_source_fragments(before) == _assert_source_fragments(after)
    expected_swa = list(range(max(0, length - 128), length))
    assert all(set(_domains(item["native_state"], 0)) == {f"{layer}:swa_kv" for layer in range(3)}
               and all(row["positions"] == expected_swa for row in item["native_state"].values())
               for item in commits.values())
    return {"replayed_input_sha256": hashlib.sha256(json.dumps([history[i] for i in range(length)]).encode()).hexdigest(),
            "unchanged_global_indexer_bytes": sum(row["bytes"] for row in before[0].values()),
            "fresh_suffix_tokens": length - prefix, "draft_live_SWA_positions": expected_swa}


def target_round_events(rank, call, request_ids, prefixes, widths, positions, tokens, counts):
    rows = [(rid, offset) for rid, width in zip(request_ids, widths) for offset in range(width)]
    assert len(rows) == len(tokens) == len(positions)
    assert positions == [p + j for p, width in zip(prefixes, widths) for j in range(width)]
    identity = {"model": "target", "dp_rank": rank}
    stages = defaultdict(list)
    for (rid, stage), position, token in zip(rows, positions, tokens):
        stages[stage].append((rid, position, token))
    for stage, inputs in stages.items():
        ids, stage_positions, stage_tokens = map(list, zip(*inputs))
        yield dict(event="compute", kind="verify", identity=identity, capture_id=call,
                   request_ids=ids, stage=stage, positions=stage_positions, tokens=stage_tokens)
    yield dict(event="finish", kind="verify", identity=identity, capture_id=call,
               request_ids=request_ids, prefixes=prefixes, accepted_input_rows=counts)


def validate_decode(output, peer, reports, drafts, accepts, aborts, mode, native_text, *,
                    reuse=False, reuse_page_probe=False, audit_original_geometry=False):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import (
        access_records, assert_cancellation, assert_first_anchors, read_model_inputs,
    )

    def read(path):
        return json.loads(path.read_text())

    cases = read(peer / "http-cases.json")["cases"]
    assert len({case["trace_id"] for case in cases}) == len(cases)
    if reuse_page_probe:
        assert len(cases) == CP_REUSE_PROBE_REQUESTS
        assert [case["case_kind"] for case in cases] == ["cold"] + ["warm"] * (len(cases) - 1)
        assert all(case["aux_info"]["input_len"] == CP_REUSE_PROBE_TOKENS for case in cases)
        cancellation = None
    else:
        assert len(cases) == (89 if reuse else REQUESTS)
        assert Counter(case["case_kind"] for case in cases) == {
            "greedy": 36 if reuse else 45, "random": 18, "eos": 1, "cancel": 1, "recovery": 1,
            "concurrent-0": 16, "concurrent-1": 16,
        }
        assert next(case for case in cases if case["case_kind"] == "eos")["eos_observed"]
        cancellation = assert_cancellation(
            next(case for case in cases if case["case_kind"] == "cancel"),
            next(case for case in cases if case["case_kind"] == "recovery"),
            access_records(peer / "logs"),
            native_text,
        )
    outcomes = {}
    for row in (*accepts, *aborts):
        if row.get("idle"):
            assert row["native_rows_published"] == 0
            continue
        key = row["rank"], row["call"]
        assert key not in outcomes
        outcomes[key] = row
    proposals, commits = defaultdict(deque), defaultdict(deque)
    for row in sorted(drafts, key=lambda item: (item["rank"], item["call"])):
        assert not row["idle"] and row["native_aliases"] and row["output_finite"]
        assert row["executed_layers"] == [0, 1, 2] and row["dspark_mode"] == mode.protocol()
        key = row["rank"], tuple(row["prefixes"])
        if row["event"] == "proposal":
            assert row["proposal_native_unchanged"] and not row["native_writes"]
            assert row["confidence_hidden"] and all(n == mode.query_width for n in row["lengths"])
            proposals[key].append(row)
        else:
            assert row["event"] == "commit" and row["feature_finite"]
            commits[(*key, row["input_feature_sha256"])].append(row)
    events, first = [], {}
    classes, batches = Counter({"zero": 0, "partial": 0, "all": 0}), Counter()
    committed_rounds = 0
    for verify in sorted(reports, key=lambda item: (item["rank"], item["call"])):
        rank, call = verify["rank"], verify["call"]
        assert not verify["idle"] and not verify["prefill"] and verify["target_verify"]
        assert verify["output_finite"] and verify["verify_native_unchanged"]
        assert all(pool["native_alias"] for pool in verify["native_pool_bindings"])
        outcome = outcomes.pop((rank, call))
        request_ids = outcome["request_ids"]
        layout = verify["row_layout"]
        prefixes, widths = layout["prefixes"], layout["lengths"]
        assert len(request_ids) == len(prefixes) == len(widths)
        if reuse:
            assert all(1 <= n <= mode.max_verified_rows for n in widths)
            assert sum(widths) <= 2 * len(widths) and verify["packed_target_verify"]
        else:
            assert all(n == mode.max_verified_rows for n in widths) and not verify["packed_target_verify"]
        key = rank, tuple(prefixes)
        proposal = proposals[key].popleft()
        assert len(proposal["lengths"]) == len(prefixes)
        queue = commits[(*key, verify["aux_feature_sha256"])]
        has_commit = bool(queue)
        if has_commit:
            validate_decode_commit_layout([verify], [queue.popleft()])
            committed_rounds += 1
        aborted = outcome["event"] == "abort"
        if aborted:
            assert outcome["native_rows_published"] == 0
            counts = [0] * len(request_ids)
        else:
            assert outcome["status"] == "ACCEPTED" and outcome["native_aliases_preserved"]
            assert outcome["prefixes"] == prefixes and outcome["verified_rows"] == widths
            counts = outcome["accepted_input_rows"]
            assert outcome["accepted_feature_rows"] == sum(counts)
            assert outcome["accepted_ends"] == [p + n for p, n in zip(prefixes, counts)]
            assert has_commit
        positions, tokens = verify["yoco"]["local_positions"], verify["input_token_ids"]
        cursor = 0
        for index, (rid, prefix, count, width) in enumerate(zip(request_ids, prefixes, counts, widths)):
            assert 0 <= count <= width
            if not aborted:
                assert outcome["accepted_input_token_ids"][index] == tokens[cursor:cursor + count]
            cursor += width
            classes["zero" if count == 0 else "all" if count == width else "partial"] += 1
            if (rank, rid) not in first:
                first[(rank, rid)] = (
                    prefix,
                    _signature(_domains(verify["native_state_before"], index)),
                    _signature(_domains(proposal["native_state"], index)),
                )
        events.extend(target_round_events(rank, call, request_ids, prefixes, widths, positions, tokens, counts))
        batches[len(prefixes)] += 1
    assert not outcomes and not any(proposals.values()) and not any(commits.values())
    # A real client disconnect can land between verify calls. The direct
    # target transaction fixture independently exercises a zero publication.
    assert committed_rounds and classes["partial"] and classes["all"]
    model_inputs = read_model_inputs(output / "logs/model_inputs")
    anchors = assert_first_anchors(model_inputs, events, cases)
    assert {(item["owner"], item["request_id"]) for item in anchors} == first.keys()
    p_evidence = read(peer / "summary.json")["evidence"]
    assert Counter(first.values()) == Counter(tuple(row) for row in p_evidence["handoffs"])
    result = {
        "requests": len(anchors), "anchors": anchors, "cancellation": cancellation,
        "model_input_records": len(model_inputs), "exact_target_and_draft_restores": len(first),
        "actual_verify_batch_sizes": dict(batches), "acceptance_classes": dict(classes),
        "full_verify_draft_commits": committed_rounds,
    }
    if reuse:
        expected = p_evidence["handoffs_by_request"]
        assert len(expected) == len(first) == len(anchors)
        assert all(tuple(expected[str(rid)]["handoff"]) == digest for (_, rid), digest in first.items())
        by_trace = {row["trace_id"]: row for row in anchors}
        warm, misses = [], []
        fixed_indices = range(CP_REUSE_PROBE_REQUESTS) if reuse_page_probe else (0, *REUSE_WARM)
        warm_length = CP_REUSE_PROBE_TOKENS if reuse_page_probe else 513
        for index in fixed_indices:
            case = cases[index]
            trace = case["trace_id"]
            anchor = by_trace[trace]
            item = expected[str(anchor["request_id"])]
            aux = case["aux_info"]
            assert aux["prefill_total_reuse_len"] == item["prefix"]
            if index == 0:
                cold_length = CP_REUSE_PROBE_TOKENS if reuse_page_probe else 129
                assert aux["input_len"] == cold_length and item["prefix"] == 0 and item["proof"] is None
                continue
            proof = item["proof"]
            assert aux["input_len"] == warm_length
            if audit_original_geometry and not item["prefix"]:
                assert not reuse_page_probe and proof is None and warm_length < CP_REUSE_PAGE_TOKENS
                misses.append({"case_index": index, **anchor, "prefix": 0, "prompt_tokens": warm_length})
                continue
            assert item["prefix"] > 0 and proof["fresh_suffix_tokens"] >= 128
            if reuse_page_probe:
                assert item["prefix"] == CP_REUSE_PAGE_TOKENS
            assert proof["replayed_input_sha256"] == p_evidence["fixed_inputs"][trace]
            assert proof["unchanged_global_indexer_bytes"] > 0
            warm.append({"case_index": index, **anchor, "prefix": item["prefix"], **proof})
        result["fixed_reuse_probes"] = {"cold_index": 0, "warm": warm, "misses": misses}
    return result
