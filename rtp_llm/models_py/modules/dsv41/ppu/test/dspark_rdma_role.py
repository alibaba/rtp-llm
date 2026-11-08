"""One native RDMA gamma3/observe/dense/fixed/-1/eager DSpark request.

Lifecycle and narrow RDMA teardown qualification are the accepted N2 helpers.
This driver never invokes an old broad/TCP test main.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role import (
    AFFINITY, classify_rdma_failures, healthy, stop_shared_owned, wait_for, write_json,
)
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_dspark_modes import validate_decode_commit_layout
from rtp_llm.tools.v41.native_smoke import SSETokenChain, make_query


def issue_request(url, directory, *, max_new_tokens=8, trace_id="n3-shared-retained", **sampling):
    directory.mkdir(parents=True, exist_ok=False)
    query = make_query("hello " * 256, max_new_tokens, stream=True, trace_id=trace_id, timeout=1200, **sampling)
    write_json(directory / "query.json", query)
    post = urllib.request.Request(url.rstrip("/") + "/", data=json.dumps(query).encode(),
                                  headers={"Content-Type": "application/json", "Connection": "close"})
    chain = SSETokenChain(expected_input_tokens=257, max_new_tokens=max_new_tokens)
    events, tokens = [], []
    with urllib.request.urlopen(post, timeout=1200) as response:
        if response.status != 200:
            raise AssertionError("native HTTP request failed")
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            tokens = chain.append(event)
            events.append(event)
    assert events and events[-1]["finished"] is True and len(tokens) == max_new_tokens, events
    assert events[0]["aux_info"]["output_len"] == 1 and events[0]["aux_info"]["step_output_len"] == 1, events[0]
    write_json(directory / "events.json", events)
    write_json(directory / "summary.json", {
        "http_pd_generation_ok": True, "first_token_once": True,
        "contiguous_SSE_counters": True, "input_tokens": 257, "output_tokens": max_new_tokens,
        "output_ids": [tokens], "events": len(events),
    })


def validate_reports(role, reports, drafts, accepts):
    profile = {"mode": "compact", "keep_tokens": 140, "window": 128,
               "max_rollback_tokens": 4, "encoder_replay": False}
    assert reports and drafts, "actual target and draft evidence is required"
    for item in reports:
        assert item["profile"] == profile, item
        assert item["model_layers"] == 40 and item["yoco"]["executed_layers"] == list(range(40)), item
        assert item["geometry"] == {"hidden_size": 5120, "hc_mult": 4, "routed_experts": 384, "top_k": 6}, item
        assert item["native_config_cache_key_hash_seed"] == item["cache_protocol"]["cache_key_hash_seed"], item
        assert item["output_finite"] is True, item
        assert all(pool["native_alias"] for pool in item["native_pool_bindings"]), item
        if item["idle"]:
            assert not item["native_pool_bindings"] and not item["native_writes"], item
            assert len(item["routed_active_rows"]) == 40 and not any(item["routed_active_rows"].values()), item
    live = [x for x in reports if not x["idle"]]
    for item in live:
        assert item["normalization"]["attention_norm_calls"] == 40, item
        assert item["normalization"]["ffn_calls"] == 40, item
        assert item["normalization"]["ffn_norm_calls"] + item["normalization"]["mega_transitions"] == 40, item
    for item in drafts:
        assert item["profile"] == profile and item["native_aliases"] and item["vocab_framework_alias"], item
        assert len(item["target_vocab_aliases"]) == 2 and all(item["target_vocab_aliases"].values()), item
        assert item["model_layers"] == 3 and item["output_finite"], item
        assert item["geometry"] == {"hidden_size": 5120, "hc_mult": 4, "routed_experts": 128, "top_k": 3}, item
        assert item["native_config_cache_key_hash_seed"] == item["cache_protocol"]["cache_key_hash_seed"], item
        assert all(value == 1 for value in item["resident_conversions"].values()), item
        if item["idle"]:
            assert not item["native_writes"] and not any(item["routed_active_rows"].values()), item
    commits = [x for x in drafts if not x["idle"] and x["event"] == "commit" and x["feature_rows"]]
    assert commits and all(x["executed_layers"] == [0, 1, 2] and x["feature_finite"] and x["native_writes"] for x in commits), commits
    if role == "prefill":
        compact = [x for x in live if x["prefill"] and x["yoco"]["input_lengths"] == [257]]
        assert {x["rank"] for x in compact} == set(range(8)), compact
        positions = set()
        for target in compact:
            yoco = target["yoco"]
            assert yoco["logical_retained_rows"] == 140 and yoco["cp_padding_rows"] == 4, target
            assert yoco["compacted"] and yoco["restored"] and yoco["cp_size"] == 8, target
            positions.update(yoco["local_positions"])
            paired = [x for x in commits if x["rank"] == target["rank"]]
            assert any(x["input_feature_sha256"] == target["aux_feature_sha256"] for x in paired), (target, paired)
        assert positions == set(range(117, 257)), positions
        assert all(x["commit_only"] for x in drafts) and not any(x["event"] == "proposal" for x in drafts), drafts
        return {"target_layers": 40, "draft_commit_layers": 3, "CP_ranks": list(range(8)),
                "logical_rows": 140, "padding_rows": 4, "positions": [117, 256],
                "aux_feature_handoff": "EXACT_BYTES", "actual_max_backtracking": 4,
                "draft_prompt_fingerprints": [{"rank": x["rank"], "state": x["native_state"]} for x in commits]}
    assert not any(x["prefill"] for x in reports), reports
    assert {x["rank"] for x in reports} == set(range(8)), reports
    assert len({x["rank"] for x in live}) == 1, live
    verifies = [x for x in live if x["target_verify"]]
    assert verifies and all(x["verify_native_unchanged"] is True for x in verifies), verifies
    assert all(x["yoco"]["input_lengths"] == [4] for x in verifies), verifies
    proposals = [x for x in drafts if not x["idle"] and x["event"] == "proposal"]
    assert proposals and all(x["executed_layers"] == [0, 1, 2] and x["confidence_hidden"] and
                             x["proposal_native_unchanged"] and not x["native_writes"] for x in proposals), proposals
    for item in proposals:
        assert len(item["normalization"]) == 3 and all(
            x["attention_norm_calls"] == 1 and x["ffn_norm_calls"] == 1 and x["ffn_calls"] == 1
            for x in item["normalization"]), item
    accepted = sorted([x for x in accepts if not x.get("idle", False)], key=lambda x: x["call"])
    assert len(accepted) == len(verifies) and accepted, (accepted, verifies)
    validate_decode_commit_layout(verifies, commits)
    end = 257
    accepted_tokens = []
    for event in accepted:
        assert event["prefixes"] == [end] and event["verified_rows"] == [4], event
        assert event["maximum_backtracking"] == 4 and event["native_aliases_preserved"], event
        assert event["status"] == "ACCEPTED" and event["accepted_feature_rows"] == sum(event["accepted_input_rows"]), event
        accepted_tokens.extend(event["accepted_input_token_ids"][0])
        end = event["accepted_ends"][0]
    assert end > 257, accepted
    first = min(proposals, key=lambda x: x["call"])
    assert first["prefixes"] == [257], first
    return {"target_verify_cycles": len(verifies), "draft_proposals": len(proposals),
            "verify_feature_cache_commits": len(commits), "actual_max_backtracking": 4,
            "verify_input_width": 4, "accepted_end": end, "accepted_input_token_ids": accepted_tokens,
            "draft_initial_native_fingerprint": first["native_state"],
            "aux_feature_handoff": "EXACT_VERIFY_BYTES", "D_prefill": False}


def validate_graph_reports(directory, mode, accepts):
    records = [json.loads(path.read_text()) for path in directory.glob("graph-rank*.json")]
    ranks = set(range(8))
    phases = {item["mode"] for item in records}
    target_phases = {"dense", "packed"}
    assert {"proposal", "commit"} <= phases <= {"proposal", "commit"} | target_phases
    for allowed in ({"proposal"}, {"commit"}, target_phases):
        assert {item["rank"] for item in records if item["mode"] in allowed} == ranks
    if mode.budget_mode == "fixed":
        expected = "packed" if mode.verify == "packed" and mode.budget < mode.gamma else "dense"
        assert phases & target_phases == {expected}, phases

    def key(item):
        return item["rank"], tuple(item["prefixes"])

    live = {}
    for item in records:
        if item["idle"]:
            continue
        phase = "verify" if item["mode"] in target_phases else item["mode"]
        bucket = live.setdefault(key(item), {})
        assert phase not in bucket, item
        bucket[phase] = item
    accepted = sorted((item for item in accepts if not item.get("idle", False)), key=lambda item: item["call"])
    idle = [item for item in accepts if item.get("reason") == "graph_empty_owner"]
    assert idle and all(item["native_rows_published"] == 0 for item in idle)
    assert accepted and len(accepted) == len(live) and {key(item) for item in accepted} == live.keys()
    assert all(item["native_writes_per_layer"] == [0] * 3
               for item in records if item["mode"] == "commit" and item["idle"])
    rounds = []
    for accept in accepted:
        replay = live[key(accept)]
        assert set(replay) == {"proposal", "verify", "commit"}
        proposal, verify, commit = (replay[phase] for phase in ("proposal", "verify", "commit"))
        lengths = accept["verified_rows"]
        assert lengths and all(1 <= length <= mode.max_verified_rows for length in lengths)
        batch, rows = len(lengths), sum(lengths)
        verify_mode = "dense" if rows == batch * mode.max_verified_rows else "packed"
        assert verify["mode"] == verify_mode and verify["lengths"] == commit["lengths"] == lengths
        assert proposal["bucket"] == [batch, mode.query_width]
        assert verify["bucket"] == commit["bucket"] == [batch, rows // batch]
        assert rows % batch == 0
        assert verify["aux_sha256"] == commit["aux_sha256"]
        assert commit["native_writes_per_layer"] == [rows] * 3
        assert accept["status"] == "ACCEPTED" and accept["native_aliases_preserved"]
        assert accept["accepted_feature_rows"] == sum(accept["accepted_input_rows"])
        rounds.append({
            "rank": accept["rank"], "prefixes": accept["prefixes"], "lengths": lengths,
            "verify_mode": verify_mode, "draft_budget": rows - batch,
            **{phase + "_bucket": replay[phase]["bucket"] for phase in replay},
        })
    assert accepted[0]["prefixes"] == [257]
    return {"replay_modes": sorted(phases), "ranks_per_mode": 8, "business_rounds": len(accepted),
            "round_buckets": rounds,
            "accepted_input_rows": [item["accepted_input_rows"] for item in accepted],
            "aux_feature_handoff": "EXACT_VERIFY_BYTES", "D_prefill": False}


def wait_peer_evidence(control, deadline):
    while not (control / "peer-ready").exists():
        if (control / "abort").exists():
            raise RuntimeError("L0 closed this attempt before peer validation")
        if time.monotonic() >= deadline:
            raise TimeoutError("peer artifacts were not collected after stop")
        time.sleep(1)
    return control / "peer-evidence"


def main(*, mode=None):
    retained = mode is None
    if retained:
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_modes import SharedDSparkMode

        mode = SharedDSparkMode(3, "observe")
    role = os.environ["RTP_V41_SHARED_ROLE"]
    if role not in ("prefill", "decode"):
        raise ValueError("shared smoke role must be prefill or decode")
    graph = role == "decode" and os.environ.get("RTP_V41_TEST_DSPARK_GRAPH") == "1"
    matrix_sampling = os.environ.get("RTP_V41_TEST_DSPARK_MATRIX_SAMPLING") == "1"
    native31 = os.environ.get("RTP_V41_TEST_DSPARK_NATIVE31") == "1"
    probe_setting = os.environ.get("RTP_V41_TEST_NATIVE31_PROBE", "")
    native31_probe = tuple(map(int, probe_setting.split(","))) if probe_setting else ()
    native10 = os.environ.get("RTP_V41_TEST_DSPARK_NATIVE10") == "1"
    reuse_page_probe = os.environ.get("RTP_V41_TEST_NATIVE10_REUSE_PAGE_PROBE") == "1"
    if reuse_page_probe and not native10:
        raise ValueError("the CP reuse page probe requires the native10 profile")
    contract_suite = native31 or native10
    if native31_probe and (not native31 or native31_probe not in ((8192,), (8192, 32768))):
        raise ValueError("synchronous localization requires the original long native31 inputs")
    graph_batch = os.environ.get("RTP_V41_TEST_DSPARK_GRAPH_BATCH") == "1"
    graph_stream_async = os.environ.get("RTP_V41_TEST_DSPARK_GRAPH_STREAM_ASYNC") == "1"
    prepare_async = os.environ.get("RTP_LLM_MTP_ASYNC_PREPARE") == "1"
    if matrix_sampling and (graph or retained):
        raise ValueError("three-sampling smoke requires the selected eager shared role")
    if contract_suite:
        from rtp_llm.models_py.modules.dsv41.ppu.test import dspark_rdma_contract as contract
        from rtp_llm.models_py.modules.dsv41.ppu.test.dspark_workload import (
            configure_contract_logging, exercise_contract_cases,
        )
        expected = (3, "bounded", "packed", 1, "fixed", True) if native10 else (3, "observe", "dense", -1, "fixed", True)
        if graph or retained or matrix_sampling or native10 and native31 or (
            mode.gamma, mode.confidence, mode.verify, mode.budget, mode.budget_mode, mode.sample_from_anchor
        ) != expected:
            raise ValueError("native DSpARK contract requires its original eager workload mode")
        points = ((contract.CP_REUSE_PROBE_TOKENS,) if reuse_page_probe
                  else contract.REUSE_POINTS if native10 else contract.POINTS)
    if graph_batch:
        from rtp_llm.models_py.modules.dsv41.ppu.test import dspark_graph_batch as graph_contract
        from rtp_llm.models_py.modules.dsv41.ppu.test.dspark_workload import configure_contract_logging

        if retained or matrix_sampling or contract_suite or (role == "decode" and not graph):
            raise ValueError("multi-bucket Graph requires the isolated paired test profile")
        if (mode.gamma, mode.confidence, mode.verify) != (3, "bounded", "packed"):
            raise ValueError("multi-bucket Graph uses the bounded packed gamma-three profile")
    affinity = os.environ.get("ACCL_NIC_GPU_AFFINITY")
    if affinity is None or json.loads(affinity) != AFFINITY:
        raise ValueError("explicit qualified eight-rank RDMA affinity is required")
    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    output = Path(os.environ["RTP_V41_SHARED_OUTPUT"]).resolve()
    control = Path(os.environ["RTP_V41_SHARED_CONTROL"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    p_address = os.environ["RTP_V41_SHARED_PREFILL_ADDRESS"]
    d_address = os.environ["RTP_V41_SHARED_DECODE_ADDRESS"]
    address = p_address if role == "prefill" else d_address
    port = int(address.rsplit(":", 1)[1])
    env = dict(os.environ, PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
               LOG_PATH=str(output / "logs"), RTP_V41_SHARED_EVIDENCE_DIR=str(output / "forwards"),
               RTP_V41_PPU_YOCO_KEEP_TOKENS="140", RTP_V41_PPU_MAX_ROLLBACK_TOKENS=str(mode.max_verified_rows),
               RTP_V41_SHARED_VALIDATE_STATE="1",
               DSV41_CED="1", DSV41_SWA_BOUNDED_REPLAY="1", DSV41_ENCODER_SWA_REPLAY="0")
    if contract_suite:
        env.update(RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE="1", DISABLE_ACCESS_LOG="0")
        if native31:
            env.update(DSV41_CED="0", DSV41_SWA_BOUNDED_REPLAY="0")
        if native31_probe and role == "prefill":
            env.update(CUDA_LAUNCH_BLOCKING="1", HGGC_LAUNCH_BLOCKING="1")
        if role == "decode":
            retention = configure_contract_logging(env, "contracts", output, points,
                                                   contract.OUTPUT_TOKENS, mode.gamma)
            env.pop("RTP_V41_DSPARK_OBSERVE_DIR", None)
            write_json(output / "model-input-retention.json", retention)
    if graph_batch:
        env.update(RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE="1", DISABLE_ACCESS_LOG="0",
                   CONCURRENCY_LIMIT="4" if role == "decode" else "32")
        if role == "decode":
            retention = configure_contract_logging(env, "contracts", output, (257,), 64, mode.gamma)
            env.pop("RTP_V41_DSPARK_OBSERVE_DIR", None)
            write_json(output / "model-input-retention.json", retention)
    argv = [sys.executable, "-B", "-m", "rtp_llm.tools.v41.launch", role,
            "--checkpoint-path", os.environ["RTP_V41_TEST_CHECKPOINT"],
            "--prefill-address", p_address, "--decode-address", d_address,
            "--start-port", str(port), "--rdma",
            "--ced-mode", "off" if native31 else "bounded", "--yoco-keep-tokens", "140", "--max-rollback-tokens", str(mode.max_verified_rows), "--dspark-gamma", str(mode.gamma),
            "--dspark-confidence", mode.confidence, "--dspark-verify-mode", mode.verify,
            "--dspark-verify-budget", str(mode.budget), "--dspark-budget-mode", mode.budget_mode,
            "--token-map-cache", str(output / "token-map.json")]
    if contract_suite:
        argv.extend(("--max-seq-len", str(contract.SEQ_CAPACITY)))
    if native10:
        argv.append("--encoder-swa-replay")
    if graph_batch:
        argv.extend(("--max-seq-len", "1024"))
        if role == "decode":
            argv[3] = graph_contract.LAUNCHER
    if not retained:
        argv.append("--kv-cache-sharded")
    if graph:
        argv.append("--shared-grouped-graph")
        if graph_stream_async:
            argv.append("--shared-graph-async")
    if not mode.sample_from_anchor:
        argv.append("--no-dspark-sample-from-anchor")
    deadline = time.monotonic() + (28800 if native31 else 14400 if native10 else 6300)
    if (contract_suite or graph_batch) and (control / "abort").exists():
        raise RuntimeError("L0 closed this attempt before service launch")
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "launch.json", {"pid": process.pid, "argv": argv, "role": role, "snapshot": snapshot,
                                            "transport": "native RDMA CacheStore", "affinity": AFFINITY})
        stop_marker = None
        try:
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8 if role == "decode" else 1)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "native shared role HTTP readiness")
            if native31:
                log_path = output / "process.log"
                wait_for(lambda: contract.admission_capacity(log_path, role, mode.gamma),
                         process, min(deadline, time.monotonic() + 240), "live full-pool capacity")
                write_json(output / "admission-capacity.json", contract.admission_capacity(log_path, role, mode.gamma))
            write_json(output / "ready.json", {"role": role, "listeners": urls})
            if role == "prefill":
                host, base = d_address.rsplit(":", 1)
                peers = [f"http://{host}:{int(base) + rank * 9}" for rank in range(8)]
                wait_for(lambda: all(map(healthy, peers)), process, deadline, "native D owner readiness")
                if contract_suite:
                    if native31:
                        admit = control / "admit"
                        wait_for(admit.exists, process, deadline, "L0 paired capacity admission")
                        admission = json.loads(admit.read_text())
                        if admission["owner"] != "L0" or admission["snapshot_id"] != snapshot["snapshot_id"]:
                            raise ValueError("native31 admission does not match this immutable launch")
                        if admission["P"] != json.loads((output / "admission-capacity.json").read_text()):
                            raise ValueError("native31 admission lost actual P capacity evidence")
                        if admission["D"]["role"] != "decode" or admission["D"]["max_seq_len"] != contract.SEQ_CAPACITY:
                            raise ValueError("native31 admission lost actual D capacity evidence")
                    if native31_probe:
                        from rtp_llm.tools.v41.native_smoke import make_query, run_request

                        cases = []
                        for index, length in enumerate(native31_probe):
                            query = make_query("hello " * (length - 1), contract.OUTPUT_TOKENS,
                                               trace_id=f"{output.name}-{length}", timeout=1200)
                            cases.append(run_request("http://" + p_address, query, output / "contract-http",
                                                     request=index, timeout=1200, expected_input_tokens=length))
                    elif reuse_page_probe:
                        from rtp_llm.tools.v41.native_smoke import make_query, run_request

                        cases = []
                        length = contract.CP_REUSE_PROBE_TOKENS
                        for index in range(contract.CP_REUSE_PROBE_REQUESTS):
                            trace = f"{output.name}-cp-reuse-{index}"
                            query = make_query("hello " * (length - 1), contract.OUTPUT_TOKENS,
                                               trace_id=trace, timeout=1200, reuse_cache=True)
                            item = run_request("http://" + p_address, query, output / "contract-http",
                                               request=index, timeout=1200, expected_input_tokens=length)
                            cases.append({**item, "query": query, "case_kind": "warm" if index else "cold"})
                        write_json(output / "http-cases.json", {"cases": cases})
                    else:
                        cases = exercise_contract_cases(
                            output, p_address, points, contract.OUTPUT_TOKENS, mode.gamma,
                            os.environ["RTP_V41_TEST_CHECKPOINT"], native10, process, deadline,
                            request_timeout_seconds=7200,
                        )
                    write_json(output / "requests-done.json", {"requests": len(cases), "suite": "native10" if native10 else "native31"})
                elif graph_batch:
                    cases = graph_contract.exercise(output, p_address, process, deadline)
                    write_json(output / "requests-done.json", {"requests": len(cases), "suite": "multi-bucket-graph"})
                else:
                    max_new_tokens = int(os.environ.get("RTP_V41_TEST_DSPARK_OUTPUT_TOKENS", "8"))
                    sampling_cases = [("http-257", {})]
                    if matrix_sampling:
                        sampling_cases.extend([
                            ("http-257-topk32", dict(top_k=32, top_p=0.95, temperature=0.8, seed=17)),
                            ("http-257-topp08", dict(top_k=0, top_p=0.8, temperature=0.8, seed=29)),
                        ])
                    for name, sampling in sampling_cases:
                        issue_request("http://" + p_address, output / name, max_new_tokens=max_new_tokens,
                                      trace_id=f"{output.name}-{name}" if matrix_sampling else "n3-shared-retained",
                                      **sampling)
                    write_json(output / "requests-done.json", {"requests": len(sampling_cases), "input_tokens": 257,
                               "output_tokens": max_new_tokens,
                               "gamma": mode.gamma, "confidence": mode.confidence, "verify_mode": mode.verify,
                               "verify_budget": mode.budget, "budget_mode": mode.budget_mode,
                               "sample_from_anchor": mode.sample_from_anchor, "execution": "eager", "actual_B": mode.max_verified_rows})
            wait_for(lambda: (control / "stop").exists(), process, deadline, "L0 reception and coordinated stop")
            with (control / "stop").open() as marker:
                marker_payload = json.load(marker)
                marker_stat = os.fstat(marker.fileno())
            stop_marker = {
                "path": str(control / "stop"),
                "inode": marker_stat.st_ino,
                "payload": marker_payload,
                "observed_at": datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                ),
            }
        finally:
            primary_error = sys.exc_info()[1]
            lifecycle = {
                "launcher_pid": process.pid,
                "snapshot_id": snapshot["snapshot_id"],
                "stop_marker": stop_marker,
                "launcher_running_at_start": process.poll() is None,
                "started_at": datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                ),
                "primary_error": (
                    repr(primary_error) if primary_error is not None else None
                ),
                "cleanup_error": None,
            }
            write_json(output / "teardown-start.json", lifecycle)
            try:
                status = stop_shared_owned(process, output)
            except Exception as cleanup_error:
                lifecycle["cleanup_error"] = repr(cleanup_error)
                write_json(
                    output / "server-exit.json",
                    {
                        "exit_code": process.poll(),
                        "cleanup_error": repr(cleanup_error),
                        "primary_error": (
                            repr(primary_error) if primary_error is not None else None
                        ),
                    },
                )
                if primary_error is None:
                    raise
            else:
                write_json(output / "server-exit.json", {"exit_code": status})
            finally:
                lifecycle["finished_at"] = datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                )
                lifecycle["exit_code"] = process.poll()
                write_json(output / "teardown-finished.json", lifecycle)
    if status != 0:
        raise RuntimeError(f"native shared server failed: {status}")
    logs = [output / "process.log", *(output / "logs").rglob("*.log")]
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    actual_cpp = {}
    expected_cpp = {"gen_num_per_cycle": str(mode.gamma), "sp_dspark_confidence_mode": mode.confidence if role == "decode" else "off",
                    "sp_dspark_verify_budget": str(mode.budget) if role == "decode" else "-1",
                    "sp_dspark_verify_mode": mode.verify if role == "decode" else "dense",
                    "sp_dspark_verify_budget_mode": mode.budget_mode if role == "decode" else "fixed",
                    "enable_cuda_graph": str(int(graph))}
    if not retained:
        expected_cpp["sp_dspark_sample_from_anchor"] = str(int(mode.sample_from_anchor))
    for field, expected in expected_cpp.items():
        values = re.findall(r"\b" + field + r":\s*([\w-]+)", text)
        if not values or set(values) != {expected}:
            raise AssertionError(f"actual C++ configuration differs: {field}={values}")
        actual_cpp[field] = {"value": expected, "instances": len(values)}
    write_json(output / "actual-cpp-configuration.json", actual_cpp)
    if "tcp messager init success" in text.lower() or "TcpMessager.cpp" in text:
        raise AssertionError("TCP CacheStore data transport was instantiated")
    if "rdma messager init success" not in text.lower():
        raise AssertionError("missing actual RdmaMessager initialization")
    if role == "decode" and not re.search(r"rdma connection to .* connect success", text):
        raise AssertionError("no actual native RDMA connection for continuation")
    rdma_health = classify_rdma_failures(
        text,
        lifecycle,
        json.loads((output / "cleanup.json").read_text()),
        launcher_pid=process.pid,
        snapshot_id=snapshot["snapshot_id"],
        stop_path=str(control / "stop"),
        role=role,
        output_path=str(output),
    )
    write_json(output / "rdma-health.json", rdma_health)
    if not rdma_health["teardown_qualified"] or rdma_health["failures"]:
        raise AssertionError("native RDMA failed")
    if native31_probe:
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                             "transport": "native RDMA CacheStore", "evidence": {
                                                 "requests": len(native31_probe), "input_tokens": native31_probe,
                                             }})
        return
    reports = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("rank*-call*.json"))]
    drafts = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("draft-*.json"))]
    accepts = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("target-*-accept.json"))]
    if contract_suite:
        if role == "prefill":
            evidence = contract.validate_prefill(reports, drafts, cases, reuse=native10,
                                                 reuse_page_probe=reuse_page_probe, output=output)
        else:
            peer = wait_peer_evidence(control, deadline)
            aborts = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("target-*-abort.json"))]
            evidence = contract.validate_decode(output, peer, reports, drafts, accepts, aborts, mode, text,
                                                 reuse=native10, reuse_page_probe=reuse_page_probe)
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                             "transport": "native RDMA CacheStore", "evidence": evidence})
        return
    if graph:
        assert not any(not item["idle"] for item in [*reports, *drafts]), "unexpected eager D forward"
        if graph_batch:
            peer = wait_peer_evidence(control, deadline)
            aborts = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("target-*-abort.json"))]
            evidence = graph_contract.validate(output, peer, accepts, aborts, mode,
                                               graph_stream_async, prepare_async, text)
        else:
            evidence = validate_graph_reports(output / "forwards", mode, accepts)
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                             "transport": "native RDMA CacheStore", "evidence": evidence})
        return
    if retained:
        evidence = validate_reports(role, reports, drafts, accepts)
    else:
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_dspark_modes import validate_mode_reports

        evidence = validate_mode_reports(role, mode, reports, drafts, accepts,
                                         expected_requests=3 if matrix_sampling else 1)
        if graph_batch:
            evidence["requests"] = len(cases)
    protocols = {item["cache_protocol"]["sha256"] for item in reports}
    if len(protocols) != 1:
        raise AssertionError("runtime/cache profile identity changed during smoke")
    live = [item for item in reports if not item["idle"] and (item["native_state_after"] if role == "prefill" else item["native_state_before"])]
    if not live or not all(any(pool["nonzero_bytes"] > 0 for pool in (item["native_state_after"] if role == "prefill" else item["native_state_before"]).values()) for item in live):
        raise AssertionError("missing actual live native-cache byte evidence")
    write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                         "transport": "native RDMA CacheStore", "cache_protocol_sha256": next(iter(protocols)),
                                         "state_fingerprints": [{"rank": item["rank"], "call": item["call"], "prefix_lengths": item["yoco"]["prefix_lengths"],
                                                                 "state": item["native_state_after"] if role == "prefill" else item["native_state_before"]} for item in live],
                                         "evidence": evidence, "FULL_VALIDATION": "DEFERRED_TO_N10"})


if __name__ == "__main__":
    if sys.argv[1:] == ["--mode-from-env"]:
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_dspark_modes import selected_mode

        main(mode=selected_mode(os.environ))
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("expected no arguments or --mode-from-env")
