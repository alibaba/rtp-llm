"""Cold native RDMA timing with independent warmup and profile requests."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
from threading import Barrier
import time
from urllib.request import Request, urlopen

from rtp_llm.tools.v41.native_smoke import SSETokenChain, make_query, run_request


POINTS = ((257, 1), (1025, 8), (8192, 8), (32768, 1))
PACKED_POINTS = ((32, 1), (128, 8))
OUTPUT_TOKENS = 16
REPEATS = 2
TIMEOUT = 1800
PROFILE_STEPS = 3


def configure(argv, env, role, output, profile):
    ced, graph, dspark = profile.split(":")
    if ced not in ("off", "bounded") or graph not in ("off", "grouped", "megamoe") or dspark not in ("off", "adaptive"):
        raise ValueError(f"invalid performance profile {profile}")
    if dspark == "adaptive" and (graph != "grouped" or ced != "bounded"):
        raise ValueError("adaptive profile requires grouped Graph and compact CED")
    argv[argv.index("--ced-mode") + 1] = ced
    argv.extend(("--max-seq-len", "65536", "--no-fake-balance-expert"))
    if dspark == "adaptive":
        argv[argv.index("--max-rollback-tokens") + 1] = "4"
        argv.extend(("--dspark-gamma", "3", "--dspark-confidence", "bounded",
                     "--dspark-verify-mode", "packed", "--dspark-verify-budget", "1",
                     "--dspark-budget-mode", "adaptive"))
    if role == "decode" and graph != "off":
        argv.append("--shared-grouped-graph" if graph == "grouped" else "--shared-decode-graph")
        if dspark == "adaptive":
            argv.append("--shared-graph-async")
    for key in ("RTP_V41_SHARED_EVIDENCE_DIR", "RTP_V41_N9_GRAPH_REPORT_DIR",
                "RTP_V41_DSPARK_OBSERVE_DIR", "RTP_V41_PERF_EVIDENCE_DIR",
                "RTP_V41_TEST_DSPARK_GRAPH_BATCH"):
        env.pop(key, None)
    timeline = output / "timeline"
    timeline.mkdir()
    env.update(RTP_V41_SHARED_VALIDATE_STATE="0", ENABLE_MODEL_INPUTS_LOG="0",
               TORCH_CUDA_PROFILER_DIR=str(timeline), DISABLE_ACCESS_LOG="1",
               CONCURRENCY_LIMIT="1" if role == "decode" and graph != "off" else "16",
               MAX_CONTEXT_BATCH_SIZE="8",
               RTP_LLM_MTP_ASYNC_PREPARE=str(int(role == "decode" and dspark == "adaptive")))
    return {"model_path": "shared", "ced": ced, "graph": graph, "dspark": dspark,
            "backend": env["RTP_V41_PPU_MOE_BACKEND"], "precision": env["RTP_V41_PPU_MOE_PRECISION"],
            "model_input_logs": False, "state_validation": False, "fake_balance": False,
            "packed_attention": env["RTP_V41_PPU_PACKED_ATTENTION"],
            "fused_qk_rope": env["RTP_V41_PPU_FUSED_QK_ROPE"],
            "qkv_streams": env["RTP_V41_PPU_QKV_STREAMS"]}


def measure_request(address, length, trace, gate):
    query = make_query("hello " * (length - 1), OUTPUT_TOKENS, stream=True, trace_id=trace, timeout=TIMEOUT)
    config = query["generate_config"]
    config.update(ignore_eos=True, return_input_ids=False, return_cum_log_probs=False)
    request = Request("http://" + address + "/", data=json.dumps(query).encode(),
                      headers={"Content-Type": "application/json", "Connection": "close",
                               "trace-id": trace, "x-dashscope-request-id": trace})
    chain = SSETokenChain(expected_input_tokens=length, max_new_tokens=OUTPUT_TOKENS)
    arrivals, tokens = [], []
    gate.wait(timeout=60)
    started = time.monotonic_ns()
    with urlopen(request, timeout=TIMEOUT) as response:
        for line in response:
            received = time.monotonic_ns()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            tokens = chain.append(event)
            arrivals.append({"received_ns": received, "output_len": len(tokens)})
            if event["finished"]:
                break
    if not arrivals or not event["finished"] or len(tokens) != OUTPUT_TOKENS:
        raise RuntimeError(f"incomplete clean SSE response for {trace}")
    first, last = (arrivals[i]["received_ns"] for i in (0, -1))
    continuation = len(tokens) - arrivals[0]["output_len"]
    if continuation <= 0 or last <= first:
        raise RuntimeError(f"HTTP did not deliver timed continuation tokens for {trace}")
    return {"trace_id": trace, "started_ns": started, "arrivals": arrivals, "output_ids": tokens,
            "ttft_ms": (first - started) / 1e6, "tpot_ms": (last - first) / continuation / 1e6,
            "aux_info": event["aux_info"]}


def wave(address, length, concurrency, name):
    gate = Barrier(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(
            lambda i: measure_request(address, length, f"{name}-{i}", gate), range(concurrency)
        ))
    start = min(row["started_ns"] for row in records)
    first = min(row["arrivals"][0]["received_ns"] for row in records)
    end = max(row["arrivals"][-1]["received_ns"] for row in records)
    tokens = sum(len(row["output_ids"]) for row in records)
    continuation = tokens - sum(row["arrivals"][0]["output_len"] for row in records)
    return {"input_tokens": length, "concurrency": concurrency, "profiled": False,
            "seconds": (end - start) / 1e9, "requests": records,
            "output_tokens_per_s": tokens * 1e9 / (end - start),
            "continuation_tokens_per_s_including_overlap_with_prefill": continuation * 1e9 / (end - first),
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in records),
            "median_tpot_ms": statistics.median(row["tpot_ms"] for row in records)}


def exercise(output, address):
    from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import write_json

    receipts, requests = [], 0
    packed_sweep = os.environ.get("RTP_V41_TEST_PERF_SHORT_PACKED") == "1"
    for length, concurrency in PACKED_POINTS if packed_sweep else POINTS:
        point = f"{output.name}-L{length}-C{concurrency}"
        rows = []
        for repeat in range(REPEATS + 1):
            label = "warmup" if repeat == 0 else f"sample-{repeat}"
            result = wave(address, length, concurrency, f"{point}-{label}")
            path = output / f"perf-L{length}-C{concurrency}-{label}.json"
            temporary = path.with_suffix(".tmp")
            write_json(temporary, result)
            temporary.replace(path)
            rows.append(result)
            requests += concurrency
        receipts.append({"input_tokens": length, "concurrency": concurrency, "unprofiled_waves": rows[1:]})
    trace = "n10-profile-" + hashlib.sha256(output.name.encode()).hexdigest()[:16]
    profiled_length = 32 if packed_sweep else 257
    query = make_query("hello " * (profiled_length - 1), OUTPUT_TOKENS, trace_id=trace, timeout=TIMEOUT)
    query["generate_config"].update(ignore_eos=True, gen_timeline=True,
                                    profile_step=PROFILE_STEPS, profile_trace_name=trace)
    profile = run_request("http://" + address, query, output / "profile-http",
                          timeout=TIMEOUT, expected_input_tokens=profiled_length)
    # Prefill contributes one step to the shared request profile. Finish
    # its window on the engine thread before stopping either native role.
    drains = []
    for index in range(PROFILE_STEPS - 1):
        query = make_query("hello " * (profiled_length - 1), 2,
                           trace_id=f"{trace}-drain-{index}", timeout=TIMEOUT)
        query["generate_config"]["ignore_eos"] = True
        drains.append(run_request("http://" + address, query, output / "profile-drain-http",
                                  request=index, timeout=TIMEOUT, expected_input_tokens=profiled_length))
    result = {"requests": requests + 1 + len(drains), "timing": receipts, "profile_http": profile,
              "profile_drain_http": drains,
              "warmup_waves_per_point": 1, "unprofiled_waves_per_point": REPEATS}
    write_json(output / "performance.json", result)
    return result


def summarize(role, output, selected, text, performance):
    assert not (output / "forwards").exists(), "debug tensor reports contaminated clean timing"
    if role == "prefill":
        assert performance and all(len(row["unprofiled_waves"]) == REPEATS for row in performance["timing"])
        return {**selected, "requests": performance["requests"], "timed_points": len(performance["timing"])}
    traces = sorted((output / "timeline").glob("*.json"))
    assert traces, "native D profiler did not export a timeline"
    if selected["graph"] != "off":
        assert "Capture for " in text and " 1 begin." in text, "D did not capture the requested Graph"
    return {**selected, "timeline_files": [path.name for path in traces]}


def gemm_reference(output, checkpoint):
    import torch

    from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import write_json
    from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry

    config_bytes = (Path(checkpoint) / "config.json").read_bytes()
    config = json.loads(config_bytes)
    token_map_bytes = (output / "token-map.json").read_bytes()
    text_config = config.get("text_config", config)
    g = V41Geometry.from_config(
        {**text_config, "engram_token_map": json.loads(token_map_bytes)}
    )
    hidden, query, head = g.hidden_size, g.q_lora_rank, g.n_heads * g.head_dim
    macs = {
        "target_attention_projections": g.num_layers * (
            hidden * query + query * head + hidden * g.head_dim + head * g.o_lora_rank
            + hidden * g.o_groups * g.o_lora_rank
        ),
        "target_moe_experts": g.num_layers * 3 * hidden * g.moe_inter_dim * (
            g.n_activated_experts + g.n_shared_experts
        ),
        "target_head": hidden * g.vocab_size,
    }
    results = []
    assert torch.cuda.device_count() == 8
    for rank in range(8):
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        points = []
        for m, n, k in ((4096, 4096, 4096), (8192, 8192, 4096)):
            left = torch.empty((m, k), device=device, dtype=torch.bfloat16).normal_()
            right = torch.empty((n, k), device=device, dtype=torch.bfloat16).normal_().t()
            result = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            for _ in range(3):
                torch.mm(left, right, out=result)
            torch.cuda.synchronize()
            times = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    torch.mm(left, right, out=result)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / 8)
            assert min(times) > 0 and bool(torch.isfinite(result[0, :8]).all())
            points.append({"mnk": [m, n, k], "ms": times,
                           "median_tflops": 2 * m * n * k / (statistics.median(times) * 1e9)})
        results.append({"rank": rank, "device": torch.cuda.get_device_name(rank),
                        "points": points, "best_median_tflops": max(row["median_tflops"] for row in points)})
    write_json(output / "gemm-reference.json", {"checkpoint_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                                               "token_map_sha256": hashlib.sha256(token_map_bytes).hexdigest(),
                                               "model_mac_terms": macs, "devices": results})


if __name__ == "__main__":
    import sys

    gemm_reference(Path(sys.argv[1]), sys.argv[2])
