"""Cumulative native CED, same-visible-domain and reuse test fixture."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.ppu.test.native_ced_role import write_json
from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_contract import (
    _assert_source_fragments, _domains, _signature,
)
from rtp_llm.tools.v41.native_smoke import make_query, run_request


MODULE = "rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_ced_suite"
PROFILES = {"full": "off", "exact": "exact", "compact": "bounded",
            "visible": "bounded", "reuse": "bounded"}


def configure(argv, env, profile, parallel, role, output, capacity):
    assert parallel in ("cp", "dp")
    argv[3] = MODULE
    argv[argv.index("--ced-mode") + 1] = PROFILES[profile]
    argv.extend(("--prefill-parallel", parallel, "--max-seq-len", str(capacity)))
    if profile == "reuse":
        argv.append("--encoder-swa-replay")
    if os.environ.get("RTP_V41_TEST_CED_PREFILL_MHC") == "legacy":
        assert role == "prefill" and profile == "full"
        env["RTP_V41_PPU_MHC_BACKEND"] = "legacy"
    env.update(RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE="1", DISABLE_ACCESS_LOG="0")
    if role == "decode":
        from rtp_llm.models_py.modules.dsv41.ppu.test.native_dspark_role import configure_contract_logging

        retention = configure_contract_logging(env, "contracts", output, (257,), 8, 0)
        env.pop("RTP_V41_DSPARK_OBSERVE_DIR", None)
        write_json(output / "model-input-retention.json", retention)


def launch():
    from rtp_llm.tools.v41.launch import main

    execute = os.execvpe

    def exec_fixture(program, command, env):
        assert command[-1] == "rtp_llm.start_server"
        return execute(program, [*command[:-1], MODULE, "--server"],
                       {**env, "RTP_V41_TEST_CED_SUITE_SERVER": "1"})

    with patch.object(os, "execvpe", exec_fixture):
        return main()


def install_reference_and_capture():
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.cp import cp_swa_replay_starts
    from rtp_llm.models_py.modules.dsv41.ppu import shared_attention
    from rtp_llm.models_py.modules.dsv41.ppu.shared_prefill import SharedForwardState
    from rtp_llm.models_py.modules.dsv41.ppu.shared_runtime import SharedNativeRuntime

    if getattr(SharedNativeRuntime, "_n10_ced_fixture", False):
        return
    SharedNativeRuntime._n10_ced_fixture = True
    profile = os.environ["RTP_V41_TEST_SHARED_CED_SUITE"]
    if profile == "visible":
        prepare = SharedForwardState.prepare_plan
        workspace = shared_attention._build_reference_workspace

        def prepare_reference(state):
            prepare(state)
            if state.plan is not None:
                starts = cp_swa_replay_starts(state.plan.context)
                assert starts is not None and len(starts) == len(state.query_prefixes)
                state.cache.fixture_visible_starts = {
                    rid: prefix + start for rid, (prefix, start) in enumerate(zip(state.query_prefixes, starts))
                }
                state.plan = None
                state.plan_reason = "same_visible_full_rows"

        def reference_workspace(**kwargs):
            kv, indices, lengths = workspace(**kwargs)
            starts = getattr(kwargs["cache"], "fixture_visible_starts", None)
            if starts is None or kwargs["layer_id"] < 21:
                return kv, indices, lengths
            # Keep the full-shaped reference causal for rows absent from the
            # compact plan. Only retained queries use the compact SWA floor.
            context = kwargs["context"]
            original = kwargs.get("swa_starts") or {}
            window = kwargs["window"]
            rows = zip(context.request_ids.tolist(), context.positions.tolist(), lengths.tolist())
            for index, (request, position, count) in enumerate(rows):
                start = starts[request]
                if position < start:
                    continue
                first = max(0, original.get(request, 0), position - window + 1)
                drop = start - first
                if drop <= 0:
                    continue
                global_count = count - (position - first + 1)
                assert global_count >= 0
                row = indices[index, 0]
                row[global_count:count - drop] = row[global_count + drop:count].clone()
                row[count - drop:count] = -1
                lengths[index] = count - drop
            return kv, indices, lengths

        SharedForwardState.prepare_plan = prepare_reference
        shared_attention._build_reference_workspace = reference_workspace

    report = SharedNativeRuntime.report
    dispatch = SharedNativeRuntime.dispatch

    def note_state(runtime, state):
        report(runtime, state)
        if state.prefill and not state.idle:
            runtime.fixture_ced_state = state

    def capture_anchor(runtime, inputs, fmha_impl=None):
        try:
            result = dispatch(runtime, inputs, fmha_impl)
            state = getattr(runtime, "fixture_ced_state", None)
            if state is None:
                return result
            directory = Path(os.environ["RTP_V41_SHARED_EVIDENCE_DIR"])
            rank = dist.get_rank() if dist.is_initialized() else 0
            if os.environ["RTP_V41_TEST_CED_PARALLEL"] == "dp":
                request_id, = inputs.request_id.reshape(-1).tolist()
                write_json(directory / f"binding-rank{rank}-call{runtime.calls}.json",
                           {"rank": rank, "call": runtime.calls, "request_id": request_id})
            if profile == "reuse":
                return result
            length, = state.lengths
            prefix, = state.query_prefixes
            hidden = result.hidden_states.reshape(-1, runtime.geometry.hidden_size)
            context = state.original_context
            if context is not None and hidden.shape[0] == context.chunk_length:
                mask = context.local_is_real & (context.global_positions == prefix + length - 1)
                index = mask.nonzero().flatten().tolist()
            elif hidden.shape[0] == length:
                index = [length - 1]
            else:
                raise ValueError("cannot identify the actual restored Prefill anchor row")
            if index:
                row, = index
                starts = getattr(state.cache, "fixture_visible_starts", None)
                if state.plan is not None:
                    offsets = cp_swa_replay_starts(state.plan.context)
                    starts = {0: prefix + offsets[0]} if offsets else None
                write_json(directory / f"anchor-rank{rank}-call{runtime.calls}.json",
                           {"rank": rank, "call": runtime.calls, "position": prefix + length - 1,
                            "visible_starts": starts, "values": hidden[row].float().tolist()})
            return result
        finally:
            if hasattr(runtime, "fixture_ced_state"):
                del runtime.fixture_ced_state

    SharedNativeRuntime.report = note_state
    SharedNativeRuntime.dispatch = capture_anchor


def exercise(output, address, profile, parallel, lengths):
    host, port = address.rsplit(":", 1)
    cases = []
    workload = [(index % 8 if parallel == "dp" else 0, length)
                for index, length in enumerate(lengths)]
    if parallel == "dp":
        # Each P frontend has its own Decode selection cursor. Keep all eight
        # original P listeners and advance one cursor across all D owners.
        extra = max(0, 8 - sum(owner == 0 for owner, _ in workload))
        workload.extend([(0, lengths[0])] * extra)
    for index, (owner, length) in enumerate(workload):
        trace = f"{output.name}-{index}"
        query = make_query("hello " * (length - 1), 4, reuse_cache=profile == "reuse",
                           timeout=7200, trace_id=trace)
        item = run_request(f"http://{host}:{int(port) + owner * 9}", query, output / "contract-http",
                           request=index, timeout=7200, expected_input_tokens=length)
        cases.append({**item, "prefill_owner": owner, "query": query})
    write_json(output / "http-cases.json", {"cases": cases})
    return cases


def validate_prefill(reports, cases, profile, parallel):
    assert all(row["prefill"] for row in reports)
    groups = defaultdict(dict)
    for row in reports:
        assert row["profile"]["mode"] == (
            profile if profile in ("full", "exact") else "compact"
        )
        rank = row["rank"]
        key = (row["call"],) if parallel == "cp" else (rank, row["call"])
        assert rank not in groups[key]
        groups[key][rank] = row
    assert len(groups) == len(cases)
    bindings, traces = {}, {}
    if parallel == "dp":
        from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import access_records

        output = Path(os.environ["RTP_V41_SHARED_OUTPUT"])
        for path in (output / "forwards").glob("binding-rank*.json"):
            item = json.loads(path.read_text())
            bindings[item["rank"], item["call"]] = item["request_id"]
        assert bindings.keys() == groups.keys()
        expected_traces = {case["trace_id"] for case in cases}
        for row in access_records(output / "logs"):
            trace = row["request"]["request_json"].get("generate_config", {}).get("trace_id")
            if trace in expected_traces:
                traces[row["id"]] = trace
        assert set(traces) == set(bindings.values()) and set(traces.values()) == expected_traces
    handoffs, exact_lengths, visible_lengths, reused = [], [], [], []
    for key, peers in groups.items():
        report = next(iter(peers.values()))
        yoco = report["yoco"]
        length, = yoco["input_lengths"]
        if profile == "exact":
            assert yoco["exact_compacted"] == (length >= 32768)
            if yoco["exact_compacted"]:
                assert yoco["restored"]
                exact_lengths.append(length)
        if profile in ("compact", "visible", "reuse") and length > 140:
            if profile == "visible":
                assert yoco["plan_reason"] == "same_visible_full_rows" and not yoco["compacted"]
            else:
                assert yoco["compacted"] and yoco["restored"] and yoco["logical_retained_rows"] == 140
            visible_lengths.append(length)
        if yoco["encoder_append_starts"]:
            assert profile == "reuse"
            assert report["immutable_global_prefix"] == "EXACT_BYTES_UNCHANGED"
            reused.append({"owners": sorted(peers), "length": length, "starts": yoco["encoder_append_starts"]})
        domains = {rank: _domains(row["native_state_after"], 0) for rank, row in peers.items()}
        if parallel == "cp":
            assert all(row["yoco"]["cp_size"] == 8 for row in peers.values())
            cache = _assert_source_fragments(domains)
        else:
            owner, = domains
            assert yoco["cp_size"] == 1
            assert all("source_owner" not in item for item in domains[owner].values())
            cache = _signature(domains[owner])
        handoffs.append({"length": length, "cache": cache, "ranks": sorted(peers), "call": key[-1]})
        if parallel == "dp":
            request_id = bindings[key]
            handoffs[-1].update(request_id=request_id, trace_id=traces[request_id])
    if parallel == "dp":
        assert {row["ranks"][0] for row in handoffs} == set(range(8))
    if profile == "reuse":
        assert reused and any(any(start > 0 for start in row["starts"].values()) for row in reused)
        if parallel == "dp":
            assert {row["owners"][0] for row in reused} == set(range(8))
    return {"requests": len(cases), "handoffs": handoffs, "exact_lengths": exact_lengths,
            "same_visible_lengths": visible_lengths, "encoder_reused": reused}


def validate_decode(output, peer, reports):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import read_model_inputs

    cases = json.loads((peer / "http-cases.json").read_text())["cases"]
    record_groups, forwards = defaultdict(list), defaultdict(list)
    for record in read_model_inputs(output / "logs/model_inputs"):
        assert not record["is_fake_stream"] and not record["warmup"]
        assert record["execution_stage"] == "decode"
        record_groups[record["native_owner"]].append(record)
    for report in reports:
        assert not report["prefill"]
        forwards[report["rank"]].append(report)
    assert forwards.keys() == record_groups.keys()
    chains = defaultdict(list)
    restores = {}
    for owner, records in record_groups.items():
        rows = sorted(forwards[owner], key=lambda item: item["call"])
        assert len(records) == len(rows)
        for record, report in zip(sorted(records, key=lambda row: row["record_sequence"]), rows):
            positions = record["sequence_lengths"].reshape(-1).tolist()
            tokens = record["combo_tokens"].reshape(-1).tolist()
            assert positions == report["row_layout"]["prefixes"]
            assert tokens == report["input_token_ids"]
            assert len(tokens) == len(positions) == len(record["trace_ids"])
            for index, (trace, prefix, token) in enumerate(zip(record["trace_ids"], positions, tokens)):
                if trace not in restores:
                    restores[trace] = (prefix, _signature(_domains(report["native_state_before"], index)))
                chains[trace].append((owner, prefix, token))
    assert chains.keys() == {case["trace_id"] for case in cases}
    anchors = []
    for case in cases:
        trace, length = case["trace_id"], case["aux_info"]["input_len"]
        rows = chains[trace]
        owner, = {row[0] for row in rows}
        tokens = case["output_ids"][0][:-1]
        if case.get("intentionally_disconnected"):
            assert len(rows) >= len(tokens)
        else:
            assert len(rows) == len(tokens)
        assert [row[1] for row in rows] == list(range(length, length + len(rows)))
        assert [row[2] for row in rows[:len(tokens)]] == tokens
        anchors.append({"trace_id": trace, "owner": owner})
    p_summary = json.loads((peer / "summary.json").read_text())["evidence"]
    assert Counter(restores.values()) == Counter((row["length"], row["cache"]) for row in p_summary["handoffs"])
    if all("trace_id" in row for row in p_summary["handoffs"]):
        for row in p_summary["handoffs"]:
            trace = row["trace_id"]
            assert restores[trace] == (row["length"], row["cache"])
    if len(cases) >= 8:
        assert {row["owner"] for row in anchors} == set(range(8))
    return {"requests": len(cases), "anchors": anchors, "exact_P_to_D_restores": len(restores)}


def wait_peer(control, deadline):
    while not (control / "peer-ready").exists():
        if (control / "abort").exists():
            raise RuntimeError("L0 closed this CED attempt before peer validation")
        if time.monotonic() >= deadline:
            raise TimeoutError("CED peer evidence did not arrive after stop")
        time.sleep(1)
    return control / "peer-evidence"


if os.environ.get("RTP_V41_TEST_CED_SUITE_SERVER") == "1":
    install_reference_and_capture()


if __name__ == "__main__":
    if sys.argv[1:] == ["--server"]:
        from rtp_llm.start_server import main

        sys.argv[1:] = []
        main()
    else:
        raise SystemExit(launch())
