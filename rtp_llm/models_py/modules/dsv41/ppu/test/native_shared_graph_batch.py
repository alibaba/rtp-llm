"""Concurrent native RDMA workload and receipts for served DSpARK Graph."""

from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from threading import Barrier
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.ppu.test.native_ced_role import wait_for, write_json
from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_contract import target_round_events
from rtp_llm.tools.v41.native_smoke import make_query, run_request


BUCKETS = (1, 2, 4)
WAVE = 24
REQUESTS = WAVE + 2
LAUNCHER = "rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_graph_batch"


def launch_multibucket():
    """Keep the native launcher checks and set the public server bucket list."""
    from rtp_llm.tools.v41.launch import main

    if os.environ.get("RTP_V41_TEST_DSPARK_GRAPH_BATCH") != "1" or sys.argv[1] != "decode":
        raise ValueError("the multibucket launcher is only for the explicit D fixture")
    execute = os.execvpe

    def exec_with_buckets(program, command, env):
        assert command[-1] == "rtp_llm.models_py.modules.dsv41.ppu.shared_server"
        assert env["ENABLE_CUDA_GRAPH"] == "1" and env["DECODE_CAPTURE_CONFIG"] == "1"
        return execute(program, command, {**env, "DECODE_CAPTURE_CONFIG": ",".join(map(str, BUCKETS))})

    with patch.object(os, "execvpe", exec_with_buckets):
        return main()


def exercise(output, address, process, deadline):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import access_records

    def request(index, kind, count, *, cancel=False, **sampling):
        trace_id = f"{output.name}-{kind}-{index}"
        query = make_query("hello " * 256, count, trace_id=trace_id, timeout=1200, stream=cancel, **sampling)
        if cancel:
            query["generate_config"]["ignore_eos"] = True
        result = run_request("http://" + address, query, output / "contract-http",
                             request=index, timeout=1200, expected_input_tokens=257,
                             cancel_after_tokens=2 if cancel else 0)
        return {**result, "case_kind": kind, "query": query}

    cases = [request(0, "serial", 8)]
    gate = Barrier(WAVE)
    profiles = (
        {"top_k": 1},
        {"top_k": 32, "top_p": 0.95, "temperature": 0.8, "seed": 17},
        {"top_k": 0, "top_p": 0.8, "temperature": 0.8, "seed": 29},
    )

    def concurrent(index):
        gate.wait(timeout=120)
        return request(index + 1, "cancel" if index == 0 else "concurrent",
                       256 if index == 0 else 64, cancel=index == 0,
                       **profiles[index % len(profiles)])

    with ThreadPoolExecutor(max_workers=WAVE) as pool:
        cases.extend(pool.map(concurrent, range(WAVE)))
    cancel = cases[1]

    def cancelled_access():
        return any(row["request"]["request_json"].get("generate_config", {}).get("trace_id") == cancel["trace_id"]
                   and row["response"].get("exception") is not None
                   for row in access_records(output / "logs"))

    wait_for(cancelled_access, process, deadline, "Graph cancel terminal access record")
    cases.append(request(WAVE + 1, "recovery", 8))
    write_json(output / "http-cases.json", {"cases": cases})
    return cases


def validate(output, peer, accepts, aborts, mode, stream_async, prepare_async, native_text):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import (
        access_records, assert_cancellation, assert_first_anchors, read_model_inputs,
    )

    assert f"-> useStreamAsync={int(stream_async)}" in native_text
    assert f"RTP_LLM_DEVICE_INPUT={int(stream_async)} -> enabled={int(stream_async)}" in native_text
    assert f"RTP_LLM_MTP_ASYNC_PREPARE={int(prepare_async)} -> enabled={int(prepare_async)}" in native_text
    records = [json.loads(path.read_text()) for path in (output / "forwards").glob("graph-rank*.json")]
    targets, proposals, commits = {}, defaultdict(deque), defaultdict(deque)
    for row in sorted(records, key=lambda item: item.get("observed_ns", 0)):
        assert row["bucket"][0] in BUCKETS
        if row["idle"]:
            if row["mode"] == "commit":
                assert row["native_writes_per_layer"] == [0] * 3
            continue
        prefix_key = row["rank"], tuple(row["prefixes"])
        if row["mode"] == "proposal":
            proposals[prefix_key].append(row)
        elif row["mode"] == "commit":
            commits[(*prefix_key, row["aux_sha256"])].append(row)
        else:
            assert row["mode"] in {"dense", "packed"}
            key = row["rank"], row["target_call"]
            assert key not in targets
            targets[key] = row
    cases = json.loads((peer / "http-cases.json").read_text())["cases"]
    assert len(cases) == REQUESTS
    assert Counter(case["case_kind"] for case in cases) == {
        "serial": 1, "concurrent": WAVE - 1, "cancel": 1, "recovery": 1,
    }
    cancellation = assert_cancellation(
        cases[1], cases[-1], access_records(peer / "logs"), native_text
    )
    events, rounds = [], []
    replay_buckets, actual_batches = Counter(), Counter()
    ragged = 0
    for outcome in sorted((*accepts, *aborts), key=lambda row: (row["rank"], row.get("call", 0))):
        if outcome.get("idle"):
            assert outcome["native_rows_published"] == 0
            continue
        rank, call = outcome["rank"], outcome["call"]
        verify = targets.pop((rank, call))
        requests, prefixes, widths = verify["request_ids"], verify["prefixes"], verify["lengths"]
        assert requests == outcome["request_ids"]
        assert len(requests) == len(prefixes) == len(widths)
        assert all(1 <= width <= mode.max_verified_rows for width in widths)
        batch, rows = len(widths), sum(widths)
        ragged += len(set(widths)) > 1
        assert verify["mode"] == ("dense" if rows == batch * mode.max_verified_rows else "packed")
        assert verify["bucket"][0] >= batch and verify["bucket"][0] * verify["bucket"][1] >= rows
        prefix_key = rank, tuple(prefixes)
        proposal = proposals[prefix_key].popleft()
        assert proposal["lengths"] == [mode.query_width] * batch
        assert proposal["bucket"][0] >= batch and proposal["bucket"][1] == mode.query_width
        candidates = commits[(*prefix_key, verify["aux_sha256"])]
        commit = candidates.popleft() if candidates else None
        if commit is not None:
            assert commit["bucket"] == verify["bucket"] and commit["lengths"] == widths
            assert commit["native_writes_per_layer"] == [rows] * 3
        if outcome["event"] == "abort":
            assert outcome["native_rows_published"] == 0
            counts = [0] * batch
        else:
            assert commit is not None and outcome["status"] == "ACCEPTED"
            assert outcome["native_aliases_preserved"] and outcome["prefixes"] == prefixes
            assert outcome["verified_rows"] == widths
            counts = outcome["accepted_input_rows"]
            assert len(counts) == batch
            assert outcome["accepted_feature_rows"] == sum(counts)
            assert outcome["accepted_ends"] == [prefix + count for prefix, count in zip(prefixes, counts)]
            assert [window is None for window in outcome["engram_committed_windows"]] == [count == 0 for count in counts]
            if not any(counts):
                assert outcome["native_rows_published"] == 0
                assert outcome["accepted_feature_sha256"] == hashlib.sha256(b"").hexdigest()
            cursor = 0
            for accepted, count, width in zip(outcome["accepted_input_token_ids"], counts, widths):
                assert accepted == verify["tokens"][cursor:cursor + count]
                cursor += width
        assert all(0 <= count <= width for count, width in zip(counts, widths))
        events.extend(target_round_events(rank, call, requests, prefixes, widths,
                                         verify["positions"], verify["tokens"], counts))
        replay_buckets[verify["bucket"][0]] += 1
        actual_batches[batch] += 1
        rounds.append({"owner": rank, "call": call, "widths": widths,
                       "proposal_bucket": proposal["bucket"], "verify_bucket": verify["bucket"],
                       "commit_bucket": commit["bucket"] if commit else None, "accepted_rows": counts})
    assert not targets and not any(proposals.values()) and not any(commits.values())
    assert 1 in replay_buckets and len(replay_buckets) > 1 and max(actual_batches) > 1
    assert ragged, "native workload did not actually replay unequal per-request widths"
    anchors = assert_first_anchors(
        read_model_inputs(output / "logs/model_inputs"), events, cases,
        terminal_prefetch_rows=mode.max_verified_rows if stream_async else 0,
    )
    return {
        "requests": len(anchors), "anchors": anchors, "cancellation": cancellation,
        "stream_async": stream_async, "device_input": stream_async, "mtp_async_prepare": prepare_async,
        "actual_batches": dict(actual_batches), "replayed_buckets": dict(replay_buckets),
        "ragged_verify_rounds": ragged, "rounds": rounds, "aux_feature_handoff": "EXACT_VERIFY_BYTES",
    }


if __name__ == "__main__":
    raise SystemExit(launch_multibucket())
