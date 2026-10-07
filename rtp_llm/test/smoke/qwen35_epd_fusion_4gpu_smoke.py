"""Manual Qwen3.5 four-GPU PD / EPD fusion and E2PD4 smoke; default invocation is a dry run."""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

TARGET = "//rtp_llm/test/smoke:qwen35_epd_fusion_4gpu_smoke"
TEXT_TARGET = "//rtp_llm/test/smoke:qwen35_pd_fusion_4gpu_smoke"
SINGLE_ENCODER_TARGET = "//rtp_llm/test/smoke:qwen35_e1pd4_smoke"
ENCODER_TARGET = "//rtp_llm/test/smoke:qwen35_e2pd4_smoke"
DEFAULT_MODEL = "/ssd/7/tuoyu.ty/workspace/model/Qwen3.5-397B-A17B-FP8"
DEFAULT_CACHE = "/root/.cache/bazel_cuda13_epd-fusion_cache"


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def gpu_ids(value, count=4):
    ids = [int(x) for x in value.split(",")]
    if (
        (count is not None and len(ids) != count)
        or len(set(ids)) != len(ids)
        or min(ids) < 0
    ):
        raise ValueError(f"expected {count or 'distinct'} non-negative GPU IDs")
    return ids


def validate_fixture(data):
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["assets_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError("fixture checksum mismatch: " + name)
    return manifest


def payload_for(data):
    messages = json.loads((data / "messages.json").read_text())
    for msg in messages:
        for part in msg.get("content", []):
            if part.get("type") == "video_url":
                part["video_url"]["url"] = str(data / "video.mp4")
                part["preprocess_config"] = {
                    "fps": 6,
                    "min_frames": 4,
                    "max_frames": 180,
                }
    return {
        "model": "qwen35",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "stream": True,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }


def server_config(
    gpus,
    profile="baseline",
    moe_strategy="fp8_per_block_ep_normal",
    decode_graph=0,
    fp8_kv_cache=0,
    native_fp8_attn=0,
    seq_size_per_block=0,
):
    gpu_ids(gpus)
    if profile not in ("baseline", "fused", "flashinfer"):
        raise ValueError("unknown optimization profile: " + profile)
    if moe_strategy not in ("fp8_per_block_ep_normal", "mega_moe_fp8"):
        raise ValueError("unsupported MoE strategy: " + moe_strategy)
    if decode_graph not in (0, 1) or (decode_graph and moe_strategy != "mega_moe_fp8"):
        raise ValueError("decode graph requires mega_moe_fp8")
    if fp8_kv_cache not in (0, 1) or native_fp8_attn not in (0, 1):
        raise ValueError("FP8 switches must be 0 or 1")
    if native_fp8_attn and not fp8_kv_cache:
        raise ValueError("native FP8 attention requires FP8 KV cache")
    # Qwen3.5-397B TP1 GDN state is 2 MiB + 72 KiB per layer. The legacy
    # shared pool requires the full-attention block to be at least that large.
    seq_size_per_block = seq_size_per_block or (4096 if fp8_kv_cache else 2048)
    if seq_size_per_block not in (2048, 4096) or (
        fp8_kv_cache and seq_size_per_block < 4096
    ):
        raise ValueError("Qwen3.5 FP8 KV shared pool requires seq_size_per_block=4096")
    env = {
        "CUDA_VISIBLE_DEVICES": gpus,
        "WORLD_SIZE": "4",
        "TP_SIZE": "1",
        "DP_SIZE": "4",
        "EP_SIZE": "4",
        "ROLE_TYPE": "PDFUSION",
        "VIT_SEPARATION": "0",
        "MOE_STRATEGY": "fp8_per_block_ep_normal",
        "QWEN35_VIDEO_BACKEND": "nvdec",
        "QWEN35_VIT_ATTN_BACKEND": "auto",
        "VIT_GPU_MAX_BATCH_SIZE": "1",
        "QWEN35_VIT_CUDA_GRAPH": "0",
        "MM_VIDEO_TOTAL_MIN_PIXELS": "2500000",
        "MM_VIDEO_TOTAL_MAX_PIXELS": "73728000",
        "REUSE_CACHE": "0",
        "MM_CACHE_ITEM_NUM": "0",
        "URL_CACHE_ITEM_NUM": "0",
        "RTP_QWEN35_FUSED_CONV_QKV_NORM": "0",
        "RTP_QWEN35_FUSED_GATED_RMSNORM_FP8": "0",
        "OMP_NUM_THREADS": "8",
        "FP8_KV_CACHE": str(fp8_kv_cache),
        "RTP_QWEN35_NATIVE_FP8_ATTN": str(native_fp8_attn),
        "RTP_FP8_ATTN_EXECUTION_LOG": "1" if native_fp8_attn else "0",
    }
    args = (
        "--use_local 1 --role_type PDFUSION --vit_separation 0 "
        "--world_size 4 --tp_size 1 --dp_size 4 --ep_size 4 "
        "--act_type BF16 --warm_up 0 --enable_cuda_graph 0 "
        "--moe_strategy fp8_per_block_ep_normal --use_deepep_moe 1 "
        "--use_deepep_low_latency 0 --use_deepep_internode 0 --use_all_gather 0 "
        "--max_seq_len 32768 --max_context_batch_size 1 "
        "--max_batch_tokens_size 32768 --max_batch_tokens_without_cache 32768 "
        "--concurrency_limit 4 --kv_cache_mem_mb 16384 --reserver_runtime_mem_mb 24576 "
        "--seq_size_per_block 2048 --kernel_seq_size_per_block 64 "
        "--fp8_kv_cache 0 --reuse_cache 0 --mm_cache_item_num 0 --url_cache_item_num 0"
    )
    args = args.replace("--fp8_kv_cache 0", f"--fp8_kv_cache {fp8_kv_cache}")
    args = args.replace(
        "--seq_size_per_block 2048", f"--seq_size_per_block {seq_size_per_block}"
    )
    env["RTP_QWEN35_DECODE_FUSION"] = "0" if profile == "baseline" else "1"
    env["RTP_QWEN35_GDN_DECODE_BACKEND"] = (
        "flashinfer" if profile == "flashinfer" else "native"
    )
    if profile != "baseline":
        env["RTP_QWEN35_FUSED_CONV_QKV_NORM"] = "1"
        env["RTP_QWEN35_FUSED_GATED_RMSNORM_FP8"] = "1"
    env["MOE_STRATEGY"] = moe_strategy
    env["RTP_CUDA_GRAPH_EXECUTION_LOG"] = "1"
    if moe_strategy == "mega_moe_fp8":
        env.update(
            DG_MEGA_MOE_FP8_IMPL="optimized",
            USE_DEEPEP_MOE="0",
            MEGA_MOE_PRE_KERNEL_BARRIER="0",
        )
        args = args.replace(
            "--moe_strategy fp8_per_block_ep_normal", "--moe_strategy mega_moe_fp8"
        )
        args = args.replace("--use_deepep_moe 1", "--use_deepep_moe 0")
    if decode_graph:
        args = args.replace("--enable_cuda_graph 0", "--enable_cuda_graph 1")
        args += " --decode_capture_config 1,2,4"
    return env, args


def throughput_config(
    env,
    args,
    policy,
    ratio="0",
    schedule_trace=0,
    trace_run_id="unset",
    coord_mode="off",
):
    if policy not in ("fifo", "prefill-first"):
        raise ValueError("unknown scheduler policy")
    if policy == "prefill-first":
        if not re.fullmatch(r"(?:0|[1-9][0-9]*|1/[1-9][0-9]*)", ratio):
            raise ValueError("invalid decode/prefill ratio")
        args += f" --pdfusion_scheduler_mode ratio --decode_prefill_ratio {ratio}"
    elif ratio != "0":
        raise ValueError("ratio requires the ratio scheduler")
    if schedule_trace:
        if (
            policy != "prefill-first"
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", trace_run_id)
            or trace_run_id == "unset"
        ):
            raise ValueError(
                "schedule trace requires ratio scheduler and a safe, explicit run ID"
            )
        args += f" --pdfusion_schedule_trace 1 --pdfusion_trace_run_id {trace_run_id}"
    if coord_mode not in ("off", "cadence"):
        raise ValueError("unknown coordinator mode")
    if coord_mode == "cadence":
        if (
            policy != "prefill-first"
            or not schedule_trace
            or len(trace_run_id) > 63
            or not ratio.isdigit()
            or not 1 <= int(ratio) <= 1000000
        ):
            raise ValueError(
                "global cadence requires trace, a unique <=63-character run ID and integer ratio >=1"
            )
    args += f" --pdfusion_coord_mode {coord_mode}"
    # concurrency_limit=4 also sets max_generate_batch_size in engine_config.
    return env, args


def positive_sizes(value):
    values = [int(x) for x in value.split(",") if x.strip()]
    if not values or min(values) < 1 or values != sorted(set(values)):
        raise ValueError("batch sizes must be positive, unique and increasing")
    return values


def capacity_config(
    env,
    args,
    rank_concurrency,
    kv_cache_mb,
    graph_batches,
    decode_graph,
    runtime_reserve_mb=24576,
):
    if not 1 <= rank_concurrency <= 256 or (kv_cache_mb != 0 and kv_cache_mb < 1024):
        raise ValueError("invalid capacity experiment limits")
    if runtime_reserve_mb < 1024:
        raise ValueError("runtime reserve must be at least 1024 MiB")
    args = args.replace(
        "--reserver_runtime_mem_mb 24576",
        f"--reserver_runtime_mem_mb {runtime_reserve_mb}",
    )
    args = args.replace(
        "--concurrency_limit 4", f"--concurrency_limit {rank_concurrency}"
    )
    args = args.replace("--kv_cache_mem_mb 16384", f"--kv_cache_mem_mb {kv_cache_mb}")
    if graph_batches:
        sizes = positive_sizes(graph_batches)
        if not decode_graph or sizes[-1] < rank_concurrency:
            raise ValueError(
                "capture batches must cover the configured rank concurrency"
            )
        args = args.replace(
            "--decode_capture_config 1,2,4", "--decode_capture_config " + graph_batches
        )
    elif decode_graph and rank_concurrency > 4:
        raise ValueError("larger concurrency requires explicit capture batches")
    return env, args


def capacity_plan(value, repeats):
    sizes = positive_sizes(value)
    return [("warmup_capacity", 16, 16)] + [
        (f"capacity_b{batch}_r{repeat+1}", batch * 4, batch * 4)
        for batch in sizes
        for repeat in range(repeats)
    ]


def capacity_refinement(summary, requested, repeats, rank_limit):
    reached = min(summary["max_observed_decode_batch_by_rank"].values())
    if reached < 1:
        return []
    return [
        (f"capacity_b{batch}_refine_r{repeat+1}", batch * 4, batch * 4)
        for batch in (reached, reached + 1)
        if batch <= rank_limit and batch not in requested
        for repeat in range(repeats)
    ]


def capacity_summary(report, evidence):
    summary = {}
    for phase in report["batches"]:
        if not phase["phase"].startswith("capacity_"):
            continue
        batch = phase["concurrency"] // 4
        rows = [r for r in report["requests"] if r["phase"] == phase["phase"]]
        entry = summary.setdefault(str(batch), dict(rounds=[], all_valid=True))
        entry["rounds"].append(phase)
        entry["all_valid"] &= all(r["ok"] for r in rows)
    for batch, entry in summary.items():
        # First occurrence of every real execution shape is always logged.
        # This is run-wide evidence, not an exact per-round batch histogram.
        entry["actual_batch_observed_on_ranks"] = sorted(
            {
                r["rank"]
                for r in evidence["records"]
                if not r["prefill"] and r["batch"] == int(batch)
            }
        )
        entry["all_ranks_reached_requested_batch"] = entry[
            "actual_batch_observed_on_ranks"
        ] == [0, 1, 2, 3]
        entry["output_tokens_per_s"] = sum(
            p["output_tokens_per_s"] * p["wall_s"] for p in entry["rounds"]
        ) / sum(p["wall_s"] for p in entry["rounds"])
    return dict(
        levels=summary,
        max_observed_decode_batch_by_rank={
            str(rank): max(
                (
                    r["batch"]
                    for r in evidence["records"]
                    if r["rank"] == rank and not r["prefill"]
                ),
                default=0,
            )
            for rank in range(4)
        },
    )


def steady_window_summary(rows, start, end):
    """Completion-time cohorts: exclude warmup/drain, include boundary carry-in."""
    completed = [r for r in rows if start <= r["completed_s"] < end]
    valid = [r for r in completed if r["ok"]]
    duration = end - start
    result = dict(
        start_s=start,
        end_s=end,
        duration_s=duration,
        completed=len(completed),
        successful=len(valid),
        errors=len(completed) - len(valid),
        qps=len(valid) / duration,
        completed_output_tokens_per_s=sum(
            r.get("usage", {}).get("completion_tokens", 0) for r in valid
        )
        / duration,
        starts=sum(start <= r["started_s"] < end for r in rows),
        in_flight_at_start=sum(
            r["started_s"] <= start < r["completed_s"] for r in rows
        ),
        in_flight_at_end=sum(r["started_s"] <= end < r["completed_s"] for r in rows),
        completion_lengths=sorted(
            {r.get("usage", {}).get("completion_tokens", 0) for r in valid}
        ),
        completed_by_rank={
            str(rank): sum(r["client_rank"] == rank for r in valid) for rank in range(4)
        },
    )
    for key in ("ttft_s", "e2e_s", "tpot_ms"):
        values = sorted(r[key] for r in valid if key in r)
        if values:
            result[key] = dict(
                median=statistics.median(values),
                p95=values[min(len(values) - 1, int(0.95 * len(values)))],
                min=values[0],
                max=values[-1],
            )
    return result


def steady_summary(rows, warmup_s, window_s, windows):
    end = warmup_s + window_s * windows
    result = dict(
        warmup_s=warmup_s,
        window_s=window_s,
        windows=windows,
        total_requests=len(rows),
        all_valid=all(r["ok"] for r in rows),
        measurement=steady_window_summary(rows, warmup_s, end),
        rounds=[
            steady_window_summary(
                rows, warmup_s + i * window_s, warmup_s + (i + 1) * window_s
            )
            for i in range(windows)
        ],
        minute_bins=[
            steady_window_summary(rows, begin, min(begin + 60, end))
            for begin in range(warmup_s, end, 60)
        ],
        warmup_completions=sum(r["completed_s"] < warmup_s for r in rows),
        drain_completions=sum(r["completed_s"] >= end for r in rows),
    )
    qps = [r["qps"] for r in result["rounds"]]
    result["round_relative_spread"] = (
        (max(qps) - min(qps)) / statistics.mean(qps) if any(qps) else None
    )
    result["rounds_within_10_percent"] = (
        bool(any(qps)) and result["round_relative_spread"] <= 0.10
    )
    return result


def run_steady_load_legacy(
    manager, payload, expected, concurrency, warmup_s, window_s, windows, out, phase
):
    """Fixed closed-loop lanes refill immediately; all outstanding requests drain."""
    from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

    lock = threading.Lock()
    abort = threading.Event()
    rows = []
    origin = {}
    state = dict(started=0, completed=0, in_flight=0, peak_in_flight=0, errors=0)
    duration = warmup_s + window_s * windows

    def start_clock():
        origin.update(monotonic=time.monotonic(), wall=time.time())

    barrier = threading.Barrier(concurrency, action=start_clock)

    def elapsed():
        return time.monotonic() - origin["monotonic"]

    with (out / (phase + "-requests.jsonl")).open("w") as journal:

        def lane(worker):
            rank = worker % 4
            url = f"http://127.0.0.1:{manager.port + rank*MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"
            barrier.wait(timeout=60)
            while not abort.is_set():
                with lock:
                    start = elapsed()
                    if start >= duration:
                        break
                    index = state["started"]
                    state["started"] += 1
                    state["in_flight"] += 1
                    state["peak_in_flight"] = max(
                        state["peak_in_flight"], state["in_flight"]
                    )
                try:
                    row = request(url, payload, index, expected)
                except BaseException:
                    abort.set()
                    raise
                row.update(
                    started_s=start,
                    completed_s=elapsed(),
                    client_rank=rank,
                    client_lane=worker,
                    phase=phase,
                )
                tokens = row.get("usage", {}).get("completion_tokens", 0)
                if tokens > 1 and "ttft_s" in row:
                    row["tpot_ms"] = (
                        1000 * (row["e2e_s"] - row["ttft_s"]) / (tokens - 1)
                    )
                with lock:
                    rows.append(row)
                    state["completed"] += 1
                    state["in_flight"] -= 1
                    if not row["ok"]:
                        state["errors"] += 1
                        abort.set()
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(lane, worker) for worker in range(concurrency)]
            while not all(f.done() for f in futures):
                for future in futures:
                    if future.done() and future.exception():
                        abort.set()
                if origin:
                    with lock:
                        now = elapsed()
                        live = dict(
                            state,
                            phase=phase,
                            elapsed_s=now,
                            target_concurrency=concurrency,
                            load_end_s=duration,
                            warmup_s=warmup_s,
                            recent_60s_successes=sum(
                                r["ok"] and now - 60 <= r["completed_s"] <= now
                                for r in rows
                            ),
                        )
                    # Replace atomically so readers never see a partial JSON document.
                    temporary = out / "steady-live.tmp"
                    save(temporary, live)
                    temporary.replace(out / "steady-live.json")
                time.sleep(5)
            for future in futures:
                future.result()
    result = steady_summary(rows, warmup_s, window_s, windows)
    result.update(
        concurrency=concurrency,
        started_at=origin["wall"],
        total_wall_s=elapsed(),
        peak_in_flight=state["peak_in_flight"],
        aborted=abort.is_set(),
    )
    save(out / (phase + "-summary.json"), result)
    return rows, result


# Explicit measurement version: old reports remain reproducible with legacy mode.
CLIENT_OPTIONS = dict(mode="legacy", chunk_size=1, profile_requests=0)


def fixed_output_payload(payload, expected, tokens):
    if not tokens:
        return payload, expected
    if payload.get("n", 1) != 1:
        raise ValueError("fixed performance counters require n=1")
    if tokens < 2 or tokens + expected["prompt_tokens"] > 32768:
        raise ValueError("fixed output must fit the unchanged context limit")
    payload = dict(payload, max_tokens=tokens)
    payload["extra_configs"] = dict(
        payload.get("extra_configs", {}),
        min_new_tokens=tokens,
        max_new_tokens=tokens,
        ignore_eos=True,
    )
    return payload, dict(expected, fixed_output_tokens=tokens, quality_test=False)


def _run_steady_load_single(
    manager, payload, expected, concurrency, warmup_s, window_s, windows, out, phase
):
    if CLIENT_OPTIONS["mode"] == "legacy":
        return run_steady_load_legacy(
            manager,
            payload,
            expected,
            concurrency,
            warmup_s,
            window_s,
            windows,
            out,
            phase,
        )
    import queue
    import resource

    MIN_WORKER_INFO_PORT_NUM = CLIENT_OPTIONS.get("worker_stride")
    if MIN_WORKER_INFO_PORT_NUM is None:
        from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
    lock = threading.Lock()
    abort = CLIENT_ABORT_EVENT if CLIENT_ABORT_EVENT is not None else threading.Event()
    rows, writer_errors = [], []
    pending = queue.SimpleQueue()
    origin = {}
    state = dict(started=0, completed=0, in_flight=0, peak_in_flight=0, errors=0)
    duration = warmup_s + window_s * windows

    def start_clock():
        if CLIENT_PROCESS_BARRIER is not None:
            CLIENT_PROCESS_BARRIER.wait(timeout=120)
            origin.update(
                monotonic=CLIENT_PROCESS_ORIGIN[0], wall=CLIENT_PROCESS_ORIGIN[1]
            )
        else:
            origin.update(monotonic=time.monotonic(), wall=time.time())

    barrier = threading.Barrier(concurrency, action=start_clock)

    def elapsed():
        return time.monotonic() - origin["monotonic"]

    # One writer owns the journal; no JSON encoding or disk I/O inside lane lock.
    def writer():
        try:
            with (out / (phase + "-requests.jsonl")).open("x") as journal:
                while True:
                    row = pending.get()
                    if row is None:
                        break
                    begin = time.monotonic()
                    row["journal_queue_s"] = begin - row.pop(
                        "journal_enqueued_monotonic"
                    )
                    data = json.dumps(row, ensure_ascii=False) + "\n"
                    journal.write(data)
                    journal.flush()
        except BaseException as error:
            writer_errors.append(error)
            abort.set()

    writer_thread = threading.Thread(target=writer, name="measurement-journal")
    writer_thread.start()

    def lane(worker):
        rank = (worker + CLIENT_OPTIONS.get("lane_offset", 0)) % 4
        url = f"http://127.0.0.1:{manager.port+rank*MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"
        barrier.wait(timeout=60)
        last_completed = None
        while not abort.is_set():
            lock_begin = time.monotonic()
            with lock:
                start = elapsed()
                if start >= duration:
                    break
                start_lock_s = time.monotonic() - lock_begin
                index = state["started"] * CLIENT_OPTIONS.get(
                    "index_stride", 1
                ) + CLIENT_OPTIONS.get("index_offset", 0)
                state["started"] += 1
                state["in_flight"] += 1
                state["peak_in_flight"] = max(
                    state["peak_in_flight"], state["in_flight"]
                )
            try:
                row = request(url, payload, index, expected)
            except BaseException:
                abort.set()
                raise
            completed = elapsed()
            row.update(
                started_s=start,
                completed_s=completed,
                client_rank=rank,
                client_lane=worker + CLIENT_OPTIONS.get("lane_offset", 0),
                phase=phase,
                start_lock_wait_s=start_lock_s,
                refill_gap_s=None if last_completed is None else start - last_completed,
                journal_enqueued_monotonic=time.monotonic(),
            )
            n = row.get("usage", {}).get("completion_tokens", 0)
            if n > 1 and "ttft_s" in row:
                row["tpot_ms"] = 1000 * (row["e2e_s"] - row["ttft_s"]) / (n - 1)
            lock_begin = time.monotonic()
            with lock:
                row["completion_lock_wait_s"] = time.monotonic() - lock_begin
                rows.append(row)
                state["completed"] += 1
                state["in_flight"] -= 1
                if not row["ok"]:
                    state["errors"] += 1
                    abort.set()
            row["journal_enqueued_monotonic"] = time.monotonic()
            pending.put(row)
            last_completed = completed

    try:
        with (out / (phase + "-client-live.jsonl")).open("x") as live_log:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(lane, worker) for worker in range(concurrency)]
                while not all(f.done() for f in futures):
                    if any(f.done() and f.exception() for f in futures):
                        abort.set()
                    if origin:
                        with lock:
                            live = dict(
                                state,
                                elapsed_s=elapsed(),
                                wall=time.time(),
                                journal_pending=pending.qsize(),
                                phase=phase,
                            )
                        usage = resource.getrusage(resource.RUSAGE_SELF)
                        live.update(
                            process_user_s=usage.ru_utime,
                            process_system_s=usage.ru_stime,
                            voluntary_switches=usage.ru_nvcsw,
                            involuntary_switches=usage.ru_nivcsw,
                        )
                        stat = Path("/proc/stat")
                        if (
                            CLIENT_OPTIONS.get("record_host_cpu", True)
                            and stat.exists()
                        ):
                            live["cpu_ticks"] = [
                                line
                                for line in stat.read_text().splitlines()
                                if line.startswith("cpu")
                            ]
                        live_log.write(json.dumps(live) + "\n")
                        live_log.flush()
                        save(out / "steady-live.tmp", live)
                        (out / "steady-live.tmp").replace(out / "steady-live.json")
                    time.sleep(1)
                for future in futures:
                    future.result()
    finally:
        pending.put(None)
        writer_thread.join()
    if writer_errors:
        raise RuntimeError("measurement journal failed") from writer_errors[0]
    result = steady_summary(rows, warmup_s, window_s, windows)
    result.update(
        concurrency=concurrency,
        started_at=origin["wall"],
        total_wall_s=elapsed(),
        peak_in_flight=state["peak_in_flight"],
        aborted=abort.is_set(),
        measurement_version=2,
        client_options=dict(CLIENT_OPTIONS),
    )
    save(out / (phase + "-summary.json"), result)
    return rows, result


CLIENT_PROCESS_BARRIER = None
CLIENT_PROCESS_ORIGIN = None
CLIENT_ABORT_EVENT = None


def _set_client_clock(origin):
    origin[0], origin[1] = time.monotonic(), time.time()


def _client_process_entry(
    port,
    options,
    payload,
    expected,
    concurrency,
    warmup_s,
    window_s,
    windows,
    out,
    phase,
    barrier,
    origin,
    abort,
):
    global CLIENT_PROCESS_BARRIER, CLIENT_PROCESS_ORIGIN, CLIENT_ABORT_EVENT
    from types import SimpleNamespace

    CLIENT_OPTIONS.clear()
    CLIENT_OPTIONS.update(options)
    CLIENT_PROCESS_BARRIER, CLIENT_PROCESS_ORIGIN, CLIENT_ABORT_EVENT = (
        barrier,
        origin,
        abort,
    )
    try:
        _run_steady_load_single(
            SimpleNamespace(port=port),
            payload,
            expected,
            concurrency,
            warmup_s,
            window_s,
            windows,
            Path(out),
            phase,
        )
    except BaseException:
        abort.set()
        raise


def run_steady_load(
    manager, payload, expected, concurrency, warmup_s, window_s, windows, out, phase
):
    processes = CLIENT_OPTIONS.get("processes", 1)
    if processes == 1:
        return _run_steady_load_single(
            manager,
            payload,
            expected,
            concurrency,
            warmup_s,
            window_s,
            windows,
            out,
            phase,
        )
    if (
        CLIENT_OPTIONS["mode"] != "measured"
        or not 1 < processes <= concurrency
        or concurrency % processes
    ):
        raise ValueError(
            "multiple client processes require measured mode and evenly divisible concurrency"
        )
    import functools
    import multiprocessing
    import resource

    stride = CLIENT_OPTIONS.get("worker_stride")
    if stride is None:
        from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

        stride = MIN_WORKER_INFO_PORT_NUM
    ctx = multiprocessing.get_context("spawn")
    origin = ctx.Array("d", [0, 0])
    abort = ctx.Event()
    barrier = ctx.Barrier(
        processes, action=functools.partial(_set_client_clock, origin)
    )
    children, directories = [], []
    each = concurrency // processes
    process_root = out / (phase + "-processes")
    process_root.mkdir(exist_ok=False)
    snapshots = {}
    try:
        for index in range(processes):
            directory = process_root / str(index)
            directory.mkdir()
            (directory / "client-profiles").mkdir()
            options = dict(
                CLIENT_OPTIONS,
                processes=1,
                worker_stride=stride,
                lane_offset=index * each,
                index_stride=processes,
                index_offset=index,
                profile_dir=str(directory / "client-profiles"),
                record_host_cpu=False,
            )
            process = ctx.Process(
                target=_client_process_entry,
                args=(
                    manager.port,
                    options,
                    payload,
                    expected,
                    each,
                    warmup_s,
                    window_s,
                    windows,
                    str(directory),
                    phase,
                    barrier,
                    origin,
                    abort,
                ),
                name=f"load-client-{index}",
            )
            process.start()
            children.append(process)
            directories.append(directory)
        save(
            process_root / "processes.json",
            dict(pids=[p.pid for p in children], start_method="spawn"),
        )
        with (out / (phase + "-client-live.jsonl")).open("x") as journal:
            while any(p.is_alive() for p in children):
                if any(p.exitcode not in (None, 0) for p in children):
                    abort.set()
                    barrier.abort()
                for index, directory in enumerate(directories):
                    try:
                        snapshots[index] = json.loads(
                            (directory / "steady-live.json").read_text()
                        )
                    except (OSError, ValueError):
                        pass
                if len(snapshots) == processes and origin[0]:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    live = dict(
                        phase=phase,
                        elapsed_s=time.monotonic() - origin[0],
                        wall=time.time(),
                        client_processes=processes,
                        target_concurrency=concurrency,
                    )
                    for key in (
                        "started",
                        "completed",
                        "in_flight",
                        "errors",
                        "journal_pending",
                        "process_user_s",
                        "process_system_s",
                        "voluntary_switches",
                        "involuntary_switches",
                    ):
                        live[key] = sum(v[key] for v in snapshots.values())
                    live["process_user_s"] += usage.ru_utime
                    live["process_system_s"] += usage.ru_stime
                    live["snapshot_lag_s"] = max(
                        live["wall"] - v["wall"] for v in snapshots.values()
                    )
                    live["peak_in_flight"] = sum(
                        v["peak_in_flight"] for v in snapshots.values()
                    )
                    stat = Path("/proc/stat")
                    if stat.exists():
                        live["cpu_ticks"] = [
                            line
                            for line in stat.read_text().splitlines()
                            if line.startswith("cpu")
                        ]
                    journal.write(json.dumps(live) + "\n")
                    journal.flush()
                    save(out / "steady-live.tmp", live)
                    (out / "steady-live.tmp").replace(out / "steady-live.json")
                time.sleep(1)
        for process in children:
            process.join()
        failures = [(p.pid, p.exitcode) for p in children if p.exitcode != 0]
        if failures:
            raise RuntimeError(
                f"load client process failed: {failures}; raw per-process journals retained"
            )
    finally:
        abort.set()
        # Only these explicitly spawned load clients are eligible for cleanup.
        for process in children:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    rows, summaries = [], []
    for directory in directories:
        with (directory / (phase + "-requests.jsonl")).open() as stream:
            rows.extend(json.loads(line) for line in stream)
        summaries.append(
            json.loads((directory / (phase + "-summary.json")).read_text())
        )
    rows.sort(key=lambda r: (r["completed_s"], r["index"]))
    assert len({r["index"] for r in rows}) == len(rows)
    assert all(s["started_at"] == origin[1] for s in summaries)
    with (out / (phase + "-requests.jsonl")).open("x") as journal:
        for row in rows:
            journal.write(json.dumps(row, ensure_ascii=False) + "\n")
    peak = in_flight = 0
    events = sorted(
        [(r["started_s"], 1) for r in rows] + [(r["completed_s"], -1) for r in rows]
    )
    for _, delta in events:
        in_flight += delta
        peak = max(peak, in_flight)
    assert in_flight == 0
    result = steady_summary(rows, warmup_s, window_s, windows)
    result.update(
        concurrency=concurrency,
        started_at=origin[1],
        total_wall_s=time.monotonic() - origin[0],
        peak_in_flight=peak,
        aborted=any(s["aborted"] for s in summaries),
        measurement_version=3,
        client_options=dict(CLIENT_OPTIONS),
        client_processes=processes,
        client_pids=[p.pid for p in children],
    )
    save(out / (phase + "-summary.json"), result)
    save(
        out / "steady-live.json",
        dict(
            phase=phase,
            elapsed_s=result["total_wall_s"],
            wall=time.time(),
            started=len(rows),
            completed=len(rows),
            in_flight=0,
            peak_in_flight=peak,
            errors=sum(not r["ok"] for r in rows),
            journal_pending=0,
            client_processes=processes,
        ),
    )
    return rows, result


def throughput_plan(repeats):
    # Reverse the second sweep to expose warmup/order effects.
    plan = [("warmup_throughput", 16, 16)]
    for repeat in range(max(repeats, 2)):
        for concurrency in (4, 8, 16) if repeat % 2 == 0 else (16, 8, 4):
            plan.append((f"throughput_c{concurrency}_r{repeat + 1}", concurrency, 32))
    return plan


def run_throughput_batch(manager, payload, expected, concurrency, count):
    from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

    barrier = threading.Barrier(concurrency)

    def lane(worker):
        rank = worker % 4
        target = f"http://127.0.0.1:{manager.port + rank * MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"
        barrier.wait(timeout=60)
        rows = []
        for i in range(worker, count, concurrency):
            row = request(target, payload, i, expected)
            row.update(client_rank=rank, client_lane=worker)
            rows.append(row)
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return [row for rows in pool.map(lane, range(concurrency)) for row in rows]


def encoder_gpu_ids(value):
    ids = gpu_ids(value, count=None)
    if len(ids) not in (1, 2):
        raise ValueError("expected one or two distinct Encoder GPUs")
    return ids


def selected_gpus(pd_gpus, encoder_gpus=""):
    pd = gpu_ids(pd_gpus)
    enc = encoder_gpu_ids(encoder_gpus) if encoder_gpus else []
    if set(pd) & set(enc):
        raise ValueError("Encoder and PDFUSION GPUs must not overlap")
    return ",".join(map(str, enc + pd))


def separated_service_configs(
    pd_gpus,
    encoder_gpus,
    profile,
    ports,
    worker_stride,
    moe_strategy="fp8_per_block_ep_normal",
    decode_graph=0,
    fp8_kv_cache=0,
    native_fp8_attn=0,
    seq_size_per_block=0,
):
    """One or two independent one-GPU encoders; one TP1/DP4/EP4 service."""
    selected_gpus(pd_gpus, encoder_gpus)
    if not encoder_gpus:
        raise ValueError("one or two encoder GPUs are required")

    def endpoint(address):
        return dict(type="Vipserver", address=address, protocol="http", path="/")

    route = dict(
        service_id=f"qwen35_e{len(encoder_gpu_ids(encoder_gpus))}pd4",
        use_local=True,
        role_endpoints=[
            dict(
                group="default",
                vit_endpoint=endpoint(
                    ",".join(
                        f"127.0.0.1:{ports[f'encoder_{i}']}"
                        for i in range(len(encoder_gpu_ids(encoder_gpus)))
                    )
                ),
                pd_fusion_endpoint=endpoint(
                    ",".join(
                        f"127.0.0.1:{ports['fusion'] + i * worker_stride}"
                        for i in range(4)
                    )
                ),
            )
        ],
    )
    env, args = server_config(
        pd_gpus,
        profile,
        moe_strategy,
        decode_graph,
        fp8_kv_cache,
        native_fp8_attn,
        seq_size_per_block,
    )
    env.update(
        VIT_SEPARATION="2", GPU_COUNT="4", MODEL_SERVICE_CONFIG=json.dumps(route)
    )
    args = args.replace("--vit_separation 0", "--vit_separation 2")
    configs = []
    for i, gpu in enumerate(encoder_gpu_ids(encoder_gpus)):
        e = dict(
            env,
            CUDA_VISIBLE_DEVICES=str(gpu),
            WORLD_SIZE="1",
            TP_SIZE="1",
            DP_SIZE="1",
            EP_SIZE="1",
            GPU_COUNT="1",
            ROLE_TYPE="VIT",
            VIT_SEPARATION="1",
            VIT_SERVER_COUNT="1",
            MOE_STRATEGY="auto",
        )
        eargs = (
            "--use_local 1 --role_type VIT --vit_separation 1 --vit_server_count 1 "
            "--world_size 1 --tp_size 1 --dp_size 1 --ep_size 1 --act_type BF16 "
            "--warm_up 0 --enable_cuda_graph 0 --concurrency_limit 4 "
            "--reuse_cache 0 --mm_cache_item_num 0 --url_cache_item_num 0"
        )
        name = f"encoder_{i}"
        configs.append(dict(name=name, gpus=[gpu], env=e, args=eargs, port=ports[name]))
    configs.append(
        dict(
            name="fusion",
            gpus=gpu_ids(pd_gpus),
            env=env,
            args=args,
            port=ports["fusion"],
        )
    )
    return configs


ENCODER_PROFILES = {
    "candidate-a": (64, 512, 8, 4),
    "candidate-b": (64, 512, 8, 8),
    "original": (256, 128, 32, 64),
}


def dp2_service_configs(
    pd_gpus,
    encoder_gpus,
    pd_env,
    pd_args,
    ports,
    worker_stride,
    encoder_profile,
    encoder_rdma_pool_bytes=17179869184,
):
    """One ViT proxy and one or two independent single-GPU workers."""
    selected_gpus(pd_gpus, encoder_gpus)
    if encoder_rdma_pool_bytes <= 0:
        raise ValueError("Encoder RDMA pool bytes must be positive")
    worker_count = len(encoder_gpu_ids(encoder_gpus))
    concurrency, queue_size, workers, batch = ENCODER_PROFILES[encoder_profile]

    def endpoint(address):
        return dict(type="Vipserver", address=address, protocol="http", path="/")

    route = dict(
        service_id=f"qwen35_encoder_dp{worker_count}_pd4",
        use_local=True,
        role_endpoints=[
            dict(
                group="default",
                vit_endpoint=endpoint(f"127.0.0.1:{ports['encoder_0']}"),
                pd_fusion_endpoint=endpoint(
                    ",".join(
                        f"127.0.0.1:{ports['fusion']+i*worker_stride}" for i in range(4)
                    )
                ),
            )
        ],
    )
    shared = dict(
        MODEL_SERVICE_CONFIG=json.dumps(route),
        MM_TRANSPORT_MODE="rdma",
        MM_CACHE_CPU_MAX_BYTES="0",
        MM_CACHE_GPU_MAX_BYTES="0",
        MM_HASH_KEY_CACHE_MAX_BYTES="1073741824",
    )
    env = dict(pd_env, **shared)
    env.update(VIT_SEPARATION="2", GPU_COUNT="4")
    args = pd_args.replace("--vit_separation 0", "--vit_separation 2")
    args += " --mm_transport_mode rdma --mm_cache_cpu_max_bytes 0 --mm_cache_gpu_max_bytes 0"
    e = dict(
        env,
        CUDA_VISIBLE_DEVICES=encoder_gpus,
        WORLD_SIZE="1",
        TP_SIZE="1",
        DP_SIZE="1",
        EP_SIZE="1",
        GPU_COUNT=str(worker_count),
        ROLE_TYPE="VIT",
        VIT_SEPARATION="1",
        VIT_SERVER_COUNT=str(worker_count),
        MOE_STRATEGY="auto",
        VIT_PROXY_LOAD_BALANCE_STRATEGY="round_robin",
        VIT_CONCURRENCY=str(concurrency),
        VIT_MAX_QUEUE_SIZE=str(queue_size),
        MM_PREPROCESS_MAX_WORKERS=str(workers),
        VIT_GPU_MAX_BATCH_SIZE=str(batch),
        MM_MAX_QUEUE_SIZE="1024",
        VIT_GPU_BATCH_WAIT_MS="10",
        ARPC_RDMA_MEMPOOL_GPU_MAX_TOTAL_BYTES=str(encoder_rdma_pool_bytes),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        CUDA_ENABLE_COREDUMP_ON_EXCEPTION="1",
        CUDA_COREDUMP_SHOW_PROGRESS="1",
        CUDA_COREDUMP_GENERATION_FLAGS="skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
        CUDA_COREDUMP_FILE=str(Path.cwd() / "cuda_coredump_%h.%p.%t"),
    )
    if worker_count == 1:
        e["RTP_VIT_SINGLE_WORKER_PROXY"] = "1"
    eargs = (
        f"--use_local 1 --role_type VIT --vit_separation 1 --vit_server_count {worker_count} "
        "--world_size 1 --tp_size 1 --dp_size 1 --ep_size 1 --act_type BF16 "
        "--warm_up 0 --enable_cuda_graph 0 --reuse_cache 0 "
        "--mm_cache_item_num 0 --url_cache_item_num 0 --mm_transport_mode rdma "
        f"--vit_concurrency {concurrency} --vit_max_queue_size {queue_size} "
        f"--mm_preprocess_max_workers {workers} --gpu_max_batch_size {batch} "
        "--mm_max_queue_size 1024 --gpu_batch_wait_ms 10 "
        "--mm_cache_cpu_max_bytes 0 --mm_cache_gpu_max_bytes 0 "
        "--mm_hash_key_cache_max_bytes 1073741824"
    )
    return [
        dict(
            name=f"encoder_dp{worker_count}",
            gpus=encoder_gpu_ids(encoder_gpus),
            env=e,
            args=eargs,
            port=ports["encoder_0"],
        ),
        dict(
            name="fusion",
            gpus=gpu_ids(pd_gpus),
            env=env,
            args=args,
            port=ports["fusion"],
        ),
    ]


def encoder_log_evidence(out, worker_count=2):
    if worker_count not in (1, 2):
        raise ValueError("Encoder worker count must be one or two")
    evidence = {}
    for name in (f"encoder_{i}" for i in range(worker_count)):
        records = []
        paths = list((out / (name + "_logs")).glob("mm_access*.log*"))
        if (out / f"encoder_dp{worker_count}_logs").exists():
            worker = name.rsplit("_", 1)[1]
            paths = list(
                (out / f"encoder_dp{worker_count}_logs").glob(
                    f"mm_access*_s{worker}.log*"
                )
            )
        for path in paths:
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("exception") is None and "MMEmbeddingRes" in str(
                    row.get("response", "")
                ):
                    records.append(
                        {
                            "request_id": row.get("request_id"),
                            "response": row["response"],
                        }
                    )
        evidence[name] = records
    return evidence


def check_gpu_idle(gpus):
    selected = set(gpu_ids(gpus, count=None))
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpus,
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=60,
    )
    rows = [
        tuple(int(x.strip()) for x in line.split(","))
        for line in raw.splitlines()
        if line.strip()
    ]
    if {r[0] for r in rows} != selected:
        raise RuntimeError("missing GPU telemetry: " + raw)
    if any(r[1] > 1024 or r[2] != 0 for r in rows):
        raise RuntimeError("selected GPUs are busy: " + raw)
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpus,
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=60,
    )
    if processes.strip():
        raise RuntimeError("selected GPUs have compute processes: " + processes)
    return {"memory": raw, "processes": processes}


def expected_tokens(model, data, payload):
    """Compute from tokenizer + video metadata, without loading model weights."""
    from types import SimpleNamespace

    import av
    import torch
    from transformers import AutoProcessor

    from rtp_llm.multimodal.qwen3_vl_video import (
        resolve_video_size,
        sample_frame_indices,
        video_resize_shape,
        video_timestamp_tokens,
    )

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    tokenizer = processor.tokenizer
    with av.open(str(data / "video.mp4")) as container:
        stream = container.streams.video[0]
        frames, fps, height, width = (
            stream.frames,
            float(stream.average_rate),
            stream.height,
            stream.width,
        )
        if not frames:
            frames = sum(1 for _ in container.decode(video=0))
    config = SimpleNamespace(fps=6, min_frames=4, max_frames=180)
    vp = processor.video_processor
    indices = sample_frame_indices(frames, fps, config)
    size = resolve_video_size(vp, 2500000, 73728000)
    h, w = video_resize_shape(config, len(indices), height, width, vp, size)
    t = (len(indices) + vp.temporal_patch_size - 1) // vp.temporal_patch_size
    grid = torch.tensor([[t, h // vp.patch_size, w // vp.patch_size]])
    timestamps = video_timestamp_tokens(grid, indices, fps, processor)
    visual = int(grid.prod()) // vp.merge_size**2
    messages = copy.deepcopy(payload["messages"])
    for msg in messages:
        for i, part in enumerate(msg.get("content", [])):
            if part["type"] == "video_url":
                msg["content"][i] = {"type": "video", "video": part["video_url"]["url"]}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
        tools=[],
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt)
    video_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if ids.count(video_id) != 1:
        raise ValueError("expected exactly one video placeholder")
    # The model replaces the video placeholder with timestamp + delimiter +
    # visual embeddings for each temporal patch. Outer delimiters remain.
    expanded = visual + sum(len(x) + 2 for x in timestamps)
    result = {
        "prompt_tokens": len(ids) - 1 + expanded,
        "video_tokens": expanded,
        "visual_tokens": visual,
        "video_text_tokens": expanded - visual,
        "frames": len(indices),
        "grid": grid.tolist(),
        "template_tokens": len(ids),
        "timestamp_tokens": sum(map(len, timestamps)),
        "method": "current HF chat template plus sampled video grid/timestamp expansion",
    }
    if visual != 20240:
        raise ValueError("unexpected fixture video token count: " + str(result))
    return result


def text_payload_and_tokens(model, target_tokens=24601):
    """Deterministic long-text workload; no video processing or model weights."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    instruction = (
        "Read the operational records below. Write an English report of about 500 words "
        "with five sections: workload, scheduling, memory, reliability, and recommendations. "
        "Base your report only on these synthetic records; do not invent measurements.\n\n"
    )
    records = "\n".join(
        f"Record {i:04d}: The service completed {100+i%37} requests. "
        f"The input queue held {i%9} requests and the memory pool used {60+i%20} percent. "
        "Prefill and decode shared one worker. No failures were reported. "
        "The operator recorded the queue and memory at the end of the interval."
        for i in range(800)
    )

    def messages(text):
        return [
            {"role": "user", "content": [{"type": "text", "text": instruction + text}]}
        ]

    def render(msgs):
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
            tools=[],
            enable_thinking=False,
        )

    budget = target_tokens - len(tokenizer.encode(render(messages(""))))
    text = tokenizer.decode(
        tokenizer.encode(records, add_special_tokens=False)[:budget]
    )
    msgs = messages(text)
    ids = tokenizer.encode(render(msgs))
    if abs(len(ids) - target_tokens) > 4:
        raise ValueError("unexpected text input length: " + str(len(ids)))
    payload = dict(
        model="qwen35",
        messages=msgs,
        max_tokens=4096,
        temperature=0,
        top_p=1,
        top_k=1,
        stream=True,
        enable_thinking=False,
        chat_template_kwargs={"enable_thinking": False},
        stream_options={"include_usage": True},
    )
    expected = dict(
        prompt_tokens=len(ids),
        video_tokens=0,
        visual_tokens=0,
        workload="text",
        method="current HF tokenizer and chat template on saved request",
        prompt_sha256=hashlib.sha256(render(msgs).encode()).hexdigest(),
    )
    return payload, expected


def validate_response(result, expected):
    errors = []
    if result.get("http_status") != 200:
        errors.append("HTTP status")
    allowed_finish = (
        ("stop", "length") if expected.get("fixed_output_tokens") else ("stop",)
    )
    if expected.get("fixed_output_tokens"):
        if (
            result.get("usage", {}).get("completion_tokens")
            != expected["fixed_output_tokens"]
        ):
            errors.append("fixed output token mismatch")
        if (
            result.get("aux_info", {}).get("output_len")
            != expected["fixed_output_tokens"]
        ):
            errors.append("fixed backend output token mismatch")
    if not result.get("content") or result.get("finish_reason") not in allowed_finish:
        errors.append("incomplete, empty, or truncated generation")
    if "choices" in result:
        choices = result["choices"]
        if set(map(int, choices)) != set(range(result.get("expected_choices", 1))):
            errors.append("missing or unexpected choice index")
        if any(
            not c.get("content") or c.get("finish_reason") not in allowed_finish
            for c in choices.values()
        ):
            errors.append("incomplete, empty, or truncated choice")
    usage = result.get("usage", {})
    aux = result.get("aux_info", {})
    if usage.get("prompt_tokens") != expected["prompt_tokens"]:
        errors.append("prompt token mismatch")
    video_tokens = (usage.get("prompt_tokens_details") or {}).get("video_tokens")
    if (expected["video_tokens"] and video_tokens != expected["video_tokens"]) or (
        not expected["video_tokens"] and video_tokens not in (None, 0)
    ):
        errors.append("video token mismatch")
    if aux.get("pd_sep") is not False:
        errors.append("missing/invalid PDFUSION evidence")
    if aux.get("reuse_len") != 0:
        errors.append("missing/invalid cache evidence")
    if result.get("error"):
        errors.append(result["error"])
    return errors


def request(url, payload, index, expected):
    import requests

    begin = time.monotonic()
    cpu_begin = time.thread_time()
    profiler = None
    if index < CLIENT_OPTIONS["profile_requests"] and CLIENT_OPTIONS.get("profile_dir"):
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    result = {
        "index": index,
        "request_id": str(uuid.uuid4()),
        "content": "",
        "usage": {},
        "aux_info": {},
        "choices": {},
        "expected_choices": payload.get("n", 1),
    }
    result.update(
        client_begin_wall=time.time(),
        sse_events=0,
        sse_line_bytes=0,
        parse_cpu_s=0.0,
        parse_wall_s=0.0,
        read_cpu_s=0.0,
        read_wall_s=0.0,
    )
    try:
        with requests.post(
            url,
            json=payload,
            headers={"X-Request-ID": result["request_id"]},
            stream=True,
            timeout=(10, 1200),
        ) as response:
            result["http_status"] = response.status_code
            response.raise_for_status()
            result["headers_s"] = time.monotonic() - begin
            read_begin, read_cpu = time.monotonic(), time.thread_time()
            for line in response.iter_lines(chunk_size=CLIENT_OPTIONS["chunk_size"]):
                received = time.monotonic()
                result["read_wall_s"] += received - read_begin
                result["read_cpu_s"] += time.thread_time() - read_cpu
                result["sse_line_bytes"] += len(line)
                read_begin, read_cpu = time.monotonic(), time.thread_time()
                if time.monotonic() - begin > 1200:
                    raise TimeoutError("request exceeded 1200 seconds")
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    result["done"] = True
                    break
                parse_begin, parse_cpu = time.monotonic(), time.thread_time()
                obj = json.loads(body)
                result["sse_events"] += 1
                if isinstance(obj.get("debug_info"), dict):
                    result["generation_debug"] = {
                        key: value
                        for key, value in obj["debug_info"].items()
                        if key
                        in (
                            "generate_config",
                            "eos_token_id",
                            "stop_word_ids_list",
                            "stop_words_list",
                        )
                    }
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                if obj.get("usage"):
                    result["usage"] = obj["usage"]
                    result["last_usage_received_s"] = received - begin
                if obj.get("aux_info"):
                    result["aux_info"].update(obj["aux_info"])
                for choice in obj.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content and "ttft_s" not in result:
                        result["ttft_s"] = time.monotonic() - begin
                    result["content"] += content
                    choice_result = result["choices"].setdefault(
                        str(choice.get("index", 0)), {"content": ""}
                    )
                    choice_result["content"] += content
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
                        choice_result["finish_reason"] = choice["finish_reason"]
                        result["finish_received_s"] = received - begin
                result["parse_cpu_s"] += time.thread_time() - parse_cpu
                result["parse_wall_s"] += time.monotonic() - parse_begin
                read_begin, read_cpu = time.monotonic(), time.thread_time()
    except Exception as error:
        result["error"] = repr(error)
    result["e2e_s"] = time.monotonic() - begin
    result["client_thread_cpu_s"] = time.thread_time() - cpu_begin
    result["client_end_wall"] = time.time()
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(
            str(
                Path(CLIENT_OPTIONS["profile_dir"]) / (result["request_id"] + ".pstats")
            )
        )
    result["validation_errors"] = validate_response(result, expected)
    result["ok"] = not result["validation_errors"]
    return result


def performance_summary(report):
    """Exclude warmup and report observed ranges, not small-sample tail claims."""
    summary = {}
    for name in (
        "single_video",
        "concurrent_video",
        "single_text",
        "concurrent_text",
        "throughput_c4_",
        "throughput_c8_",
        "throughput_c16_",
    ):
        rows = [r for r in report["requests"] if r.get("phase", "").startswith(name)]
        batches = [b for b in report["batches"] if b["phase"].startswith(name)]
        if not rows:
            continue
        group = {"requests": len(rows), "all_valid": all(r["ok"] for r in rows)}
        for metric in ("ttft_s", "e2e_s", "tpot_ms"):
            values = [r[metric] for r in rows if metric in r]
            if values:
                group[metric] = dict(
                    median=statistics.median(values), min=min(values), max=max(values)
                )
        group["output_tokens_per_s"] = sum(
            r.get("usage", {}).get("completion_tokens", 0) for r in rows
        ) / sum(b["wall_s"] for b in batches)
        group["completion_tokens"] = [
            r.get("usage", {}).get("completion_tokens", 0) for r in rows
        ]
        summary[name] = group
    return summary


def execution_evidence(out):
    ranks = {
        str(i): {
            "capture": False,
            "prefill": False,
            "decode": False,
            "replay": False,
            "padding3": False,
            "mega_moe": False,
        }
        for i in range(4)
    }
    records = []
    mega = False
    rank_by_pid = {}
    mega_pids = set()
    fp8_calls = []
    fallbacks = []
    # Python uses the role directory; the C++ alog configuration uses logs/.
    paths = list((out / "fusion_logs").rglob("*")) + list(
        (out / "logs").glob("engine.log*")
    )
    for path in paths:
        if not path.is_file() or path.suffix in (".gz", ".zip"):
            continue
        with path.open(errors="replace") as stream:
            for line in stream:
                pid = re.search(r"\[process-(\d+)\]", line)
                rank_match = re.search(
                    r"\[rank: (\d+)\] initialize process_group", line
                )
                if pid and rank_match:
                    rank_by_pid[pid[1]] = rank_match[1]
                if "MegaMoE FP8 weights prepared during model construction" in line:
                    mega = True
                    if pid:
                        mega_pids.add(pid[1])
                fp8 = re.search(
                    r"FP8_ATTN_EXECUTION phase=(prefill|decode) native=1 q=torch.float8_e4m3fn kv=torch.float8_e4m3fn out=torch.bfloat16",
                    line,
                )
                if pid and fp8:
                    fp8_calls.append((pid[1], fp8[1]))
                lowered = line.lower()
                if (
                    "no metric named" not in lowered
                    and ("fallback" in lowered or "cannot run" in lowered)
                    and ("graph" in lowered or "combo_position" in lowered)
                ):
                    fallbacks.append(line.strip()[-1500:])
                match = re.search(r"GRAPH_CAPTURE_READY rank=(\d+)", line)
                if match and match[1] in ranks:
                    ranks[match[1]]["capture"] = True
                match = re.search(
                    r"MODEL_EXECUTION rank=(\d+) mode=(\w+) prefill=(\d+) batch=(\d+) graph_bs=(\d+) count=(\d+)",
                    line,
                )
                if match and match[1] in ranks:
                    rank, mode, prefill, batch, graph_bs, count = match.groups()
                    row = dict(
                        rank=int(rank),
                        mode=mode,
                        prefill=int(prefill),
                        batch=int(batch),
                        graph_bs=int(graph_bs),
                        count=int(count),
                    )
                    records.append(row)
                    ranks[rank]["prefill" if prefill == "1" else "decode"] = True
                    if mode == "graph" and prefill == "0":
                        ranks[rank]["replay"] = True
                        if batch == "3" and graph_bs == "4":
                            ranks[rank]["padding3"] = True
    for pid in mega_pids:
        rank = rank_by_pid.get(pid)
        if rank in ranks:
            ranks[rank]["mega_moe"] = True
    native_ranks = {str(i): {"prefill": False, "decode": False} for i in range(4)}
    for pid, phase in fp8_calls:
        if rank_by_pid.get(pid) in native_ranks:
            native_ranks[rank_by_pid[pid]][phase] = True
    return dict(
        ranks=ranks,
        mega_moe=mega,
        records=records,
        fallbacks=fallbacks[-100:],
        native_fp8=native_ranks,
    )


def validate_execution_evidence(evidence, decode_graph, edge_cases):
    errors = []
    if not evidence["mega_moe"]:
        errors.append("missing MegaMoE weight initialization evidence")
    for rank, values in evidence["ranks"].items():
        for field in (
            ["mega_moe", "prefill", "decode", "capture", "replay"]
            if decode_graph
            else ["mega_moe", "prefill", "decode"]
        ):
            if not values[field]:
                errors.append(f"rank {rank}: missing {field} evidence")
    if any(r["mode"] == "graph" and r["prefill"] for r in evidence["records"]):
        errors.append("Prefill unexpectedly used graph execution")
    if decode_graph and any(
        r["mode"] == "eager" and not r["prefill"] for r in evidence["records"]
    ):
        errors.append("Decode silently used eager execution in graph validation")
    if (
        decode_graph
        and edge_cases
        and not any(r["padding3"] for r in evidence["ranks"].values())
    ):
        errors.append("missing actual batch-3 to graph-4 replay")
    return errors


def validate_coordinator_boundary(records, batch, decode_graph):
    # The frontend port is a routing entry point, not a pinned execution rank.
    matches = [r for r in records if not r["prefill"] and r["batch"] == batch]
    if not matches:
        raise RuntimeError(f"missing actual decode batch evidence: {batch}")
    expected_graph = 64 if batch <= 64 else 96 if batch <= 96 else 128
    if any(
        r["mode"] != ("graph" if decode_graph else "eager")
        or (decode_graph and r["graph_bs"] != expected_graph)
        for r in matches
    ):
        raise RuntimeError("unexpected graph boundary execution")
    return matches


def run_coordinator_checks(manager, model, out, decode_graph, fault="none"):
    """Correctness-only requests; never include in throughput measurements."""
    import requests
    from transformers import AutoTokenizer

    from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    saved = dict(requests=[], cancellations=[], idle_checks=[], graph_boundaries=[])
    log = out / "fusion_logs/process.log"

    def persist():
        save(out / "coordinator-checks.json", saved)

    def url(rank):
        return f"http://127.0.0.1:{manager.port + rank * MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"

    def short_payload(n):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List the integers 1 through 20 separated by commas. Output only that list.",
                    }
                ],
            }
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
            tools=[],
            enable_thinking=False,
        )
        payload = dict(
            model="qwen35",
            messages=messages,
            n=n,
            max_tokens=256,
            temperature=0,
            top_p=1,
            top_k=1,
            stream=True,
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False},
            stream_options={"include_usage": True},
        )
        return payload, dict(
            prompt_tokens=len(tokenizer.encode(rendered)), video_tokens=0
        )

    def step_ids():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 2 * 1024 * 1024))
            lines = f.read().decode(errors="replace").splitlines()
        ids = {}
        for line in lines:
            if "SCHEDULE_TRACE " in line and "event=end " in line:
                row = dict(
                    re.findall(r"(\w+)=([^\s]+)", line.split("SCHEDULE_TRACE ", 1)[1])
                )
                ids[int(row["rank"])] = int(row["model_step_id"])
        return ids

    def wait_idle(label):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            first = step_ids()
            time.sleep(0.5)
            second = step_ids()
            if len(first) == 4 and first == second:
                time.sleep(1)
                third = step_ids()
                if second == third:
                    saved["idle_checks"].append(
                        dict(label=label, model_step_ids=third, no_model_for_s=1.5)
                    )
                    persist()
                    return
        raise RuntimeError("coordinator did not become globally model-idle: " + label)

    # Different input lengths and staggered arrivals exercise reservations and fake peers.
    varied = [text_payload_and_tokens(model, n) for n in (256, 1024, 4096, 8192)]

    def send_varied(rank):
        time.sleep(rank * 0.15)
        payload, expected = varied[rank]
        return request(url(rank), payload, rank, expected)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(send_varied, range(4)))
    for row in rows:
        row["coord_case"] = "variable_staggered"
    saved["requests"].extend(rows)
    persist()
    if not all(r["ok"] for r in rows):
        raise RuntimeError("coordinator variable-length request failed")
    wait_idle("after_variable_staggered")
    if fault == "peer-exit":
        saved["fault_ready"] = dict(kind=fault, time=time.time(), global_idle=True)
        persist()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            alive = []
            for rank in range(4):
                try:
                    response = requests.get(
                        url(rank).replace("/v1/chat/completions", "/health"),
                        timeout=0.5,
                    )
                    alive.append(response.status_code == 200)
                except requests.RequestException:
                    alive.append(False)
            if not any(alive):
                saved["fault_observed"] = dict(
                    all_rank_health_unavailable=True, time=time.time()
                )
                persist()
                raise RuntimeError(
                    "EXPECTED_COORD_PEER_EXIT: all four rank endpoints stopped"
                )
            time.sleep(0.2)
        raise RuntimeError(
            "coordinator peer-exit test did not stop all ranks within 60 seconds"
        )

    # n-way branching reaches real Decode graph edges without filling a long-context rank.
    for i, n in enumerate((63, 64, 65, 95, 96, 97)):
        payload, expected = short_payload(n)
        rank = i % 4
        if n * expected["prompt_tokens"] >= 32768:
            raise RuntimeError("boundary request exceeds prefill token budget")
        row = request(url(rank), payload, 100 + i, expected)
        row["coord_case"] = f"branch_{n}_rank_{rank}"
        saved["requests"].append(row)
        persist()
        if not row["ok"]:
            raise RuntimeError(
                "coordinator graph-boundary request failed: " + row["coord_case"]
            )
        records = execution_evidence(out)["records"]
        matches = validate_coordinator_boundary(records, n, decode_graph)
        saved["graph_boundaries"].append(
            dict(
                frontend_rank=rank,
                execution_ranks=sorted({r["rank"] for r in matches}),
                batch=n,
                records=matches,
            )
        )
        persist()
    wait_idle("after_graph_boundaries")
    payload, expected = short_payload(1)
    payload, expected = fixed_output_payload(payload, expected, 4096)
    request_id = str(uuid.uuid4())
    started = time.time()
    seen_content = False
    with requests.post(
        url(0),
        json=payload,
        headers={"X-Request-ID": request_id},
        stream=True,
        timeout=(10, 60),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            event = json.loads(body)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                seen_content = True
                break
    saved["cancellations"].append(
        dict(
            request_id=request_id,
            rank=0,
            closed_after_first_content=seen_content,
            started_at=started,
            closed_at=time.time(),
            forced_output_tokens=4096,
        )
    )
    persist()
    if not seen_content:
        raise RuntimeError("cancellation was not exercised during generation")
    wait_idle("after_client_disconnect")
    payload, expected = short_payload(1)
    row = request(url(3), payload, 200, expected)
    row["coord_case"] = "resume_after_cancel"
    saved["requests"].append(row)
    persist()
    if not row["ok"]:
        raise RuntimeError("post-cancellation request failed")
    wait_idle("after_resume")
    saved["status"] = "PASSED"
    persist()
    return saved


def run_quality_probe(manager, model, out, label):
    """Small natural-stop sanity suite; separate from fixed-output performance."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = [
        ("multiplication", "Return only the decimal integer result of 17 * 19.", "323"),
        ("sum", "Return only the sum of these integers: 2, 5, 11, 17.", "35"),
        (
            "sorting",
            "Sort 9, 2, 5 in ascending order. Return only comma-separated integers without spaces.",
            "2,5,9",
        ),
        (
            "deduction",
            "All blue boxes contain keys. Box X is blue. Does box X contain keys? Answer only yes or no.",
            "yes",
        ),
    ]
    saved = dict(
        scope="Natural EOS/stop sanity checks; not a broad quality benchmark.",
        cases=[],
        requests=[],
    )
    for name, prompt, answer in cases:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
            tools=[],
            enable_thinking=False,
        )
        expected = dict(prompt_tokens=len(tokenizer.encode(rendered)), video_tokens=0)
        payload = dict(
            model="qwen35",
            messages=messages,
            max_tokens=128,
            temperature=0,
            top_p=1,
            top_k=1,
            stream=True,
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False},
            stream_options={"include_usage": True},
        )
        saved["cases"].append(
            dict(name=name, payload=payload, expected=expected, answer=answer)
        )
        rows = run_throughput_batch(manager, payload, expected, 4, 4)
        for row in rows:
            normalized = re.sub(r"\s+", "", row.get("content", "")).lower().strip(".。")
            row.update(
                quality_case=name,
                quality_pass=normalized == answer,
                quality_phase=label,
            )
        saved["requests"].extend(rows)
        save(out / f"quality-{label}.json", saved)
        if not all(r["ok"] and r["quality_pass"] for r in rows):
            raise RuntimeError(
                f"natural-stop quality sanity check failed: {label}/{name}"
            )
    saved["all_passed"] = True
    save(out / f"quality-{label}.json", saved)
    return dict(all_passed=True, requests=len(saved["requests"]), scope=saved["scope"])


def run_worker(a):
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    CLIENT_OPTIONS.update(
        mode=getattr(a, "client_mode", "legacy"),
        processes=getattr(a, "client_processes", 1),
        chunk_size=getattr(a, "sse_chunk_size", 1),
        profile_requests=getattr(a, "client_profile_requests", 0),
        profile_dir=str(out / "client-profiles"),
    )
    (out / "client-profiles").mkdir(exist_ok=True)
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = str(out)
    # gpu_lock owns the selection; never expand its assigned visible device set.
    gpus = a.gpus
    encoder_gpus = getattr(a, "encoder_gpus", "")
    selected = selected_gpus(gpus, encoder_gpus)
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES", selected)
    if set(gpu_ids(assigned, count=None)) != set(gpu_ids(selected, count=None)):
        raise RuntimeError("gpu_lock allocation differs from requested devices")
    profile = getattr(a, "profile", "baseline")
    repeats = getattr(a, "perf_repeats", 0)
    workload = getattr(a, "workload", "video")
    if workload == "text" and encoder_gpus:
        raise ValueError("text-only PD must not allocate Encoder GPUs")
    moe_strategy = getattr(a, "moe_strategy", "fp8_per_block_ep_normal")
    decode_graph = getattr(a, "decode_graph", 0)
    fp8_kv_cache = getattr(a, "fp8_kv_cache", 0)
    native_fp8_attn = getattr(a, "native_fp8_attn", 0)
    seq_size_per_block = getattr(a, "seq_size_per_block", 0) or (
        4096 if fp8_kv_cache else 2048
    )
    env, args = server_config(
        gpus,
        profile,
        moe_strategy,
        decode_graph,
        fp8_kv_cache,
        native_fp8_attn,
        seq_size_per_block,
    )
    policy = getattr(a, "scheduler_policy", "fifo")
    throughput = getattr(a, "throughput_sweep", 0)
    env, args = throughput_config(
        env,
        args,
        policy,
        getattr(a, "decode_prefill_ratio", "0"),
        getattr(a, "schedule_trace", 0),
        getattr(a, "trace_run_id", "unset"),
        getattr(a, "coord_mode", "off"),
    )
    capacity_batches = getattr(a, "capacity_batches", "")
    steady_concurrency = getattr(a, "steady_concurrency", "")
    env, args = capacity_config(
        env,
        args,
        getattr(a, "rank_concurrency", 4),
        getattr(a, "kv_cache_mb", 16384),
        getattr(a, "graph_batches", ""),
        decode_graph,
        getattr(a, "runtime_reserve_mb", 24576),
    )
    if workload == "text":
        env["VIT_SEPARATION"] = "2"
        args = args.replace("--vit_separation 0", "--vit_separation 2")
    report = {
        "status": "STARTING",
        "phase": "fixture",
        "requests": [],
        "env": env,
        "args": args,
    }
    report.update(
        profile=profile,
        workload=workload,
        perf_repeats=repeats,
        batches=[],
        moe_strategy=moe_strategy,
        decode_graph=decode_graph,
        fp8_kv_cache=fp8_kv_cache,
        native_fp8_attn=native_fp8_attn,
        seq_size_per_block=seq_size_per_block,
        scheduler_policy=policy,
        throughput_sweep=throughput,
        rank_concurrency=getattr(a, "rank_concurrency", 4),
        runtime_reserve_mb=getattr(a, "runtime_reserve_mb", 24576),
        kv_cache_mb=getattr(a, "kv_cache_mb", 16384),
        capacity_batches=capacity_batches,
        steady_concurrency=steady_concurrency,
        steady={},
    )
    manager = None
    managers = []
    stop = threading.Event()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def monitor():
        with (out / "gpu-memory.jsonl").open("a") as f:
            while not stop.is_set():
                try:
                    raw = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--id=" + selected,
                            "--query-gpu=index,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=30,
                    )
                    f.write(
                        json.dumps(
                            {"time": time.time(), "phase": report["phase"], "gpus": raw}
                        )
                        + "\n"
                    )
                    f.flush()
                except Exception as error:
                    f.write(json.dumps({"error": repr(error)}) + "\n")
                stop.wait(10)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        # Recheck after gpu_lock acquisition: a different container may not
        # share our lock namespace and can start during Bazel/lock setup.
        report["phase"] = "gpu_availability_after_lock"
        report["gpu_before_model"] = check_gpu_idle(selected)
        report["phase"] = "fixture"
        data = Path(a.data_dir).resolve()
        if workload == "video":
            validate_fixture(data)
            payload = payload_for(data)
            report["phase"] = "dependency_and_input_validation"
            expected = expected_tokens(a.model_dir, data, payload)
        else:
            report["phase"] = "dependency_and_input_validation"
            payload, expected = text_payload_and_tokens(a.model_dir)
        payload, expected = fixed_output_payload(
            payload, expected, getattr(a, "fixed_output_tokens", 0)
        )
        report["client_options"] = dict(CLIENT_OPTIONS)
        save(out / "request.json", payload)
        report["expected"] = expected
        save(out / "expected-input.json", expected)
        edge_payload, edge_expected = None, None
        if getattr(a, "graph_edge_cases", 0):
            # n=3 duplicates Prefill before branching Decode; keep its aggregate
            # token count within the existing MegaMoE per-rank token capacity.
            edge_payload, edge_expected = text_payload_and_tokens(
                a.model_dir, target_tokens=8192
            )
            edge_payload["n"] = 3
            save(out / "edge-request.json", edge_payload)
            save(out / "edge-expected-input.json", edge_expected)
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        for key in (
            "MODEL_SERVICE_CONFIG",
            "REMOTE_RPC_SERVER_IP",
            "REMOTE_SERVER_PORT",
        ):
            os.environ.pop(key, None)
        if encoder_gpus:
            from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

            ports = {
                name: int(MagaServerManager.get_free_port())
                for name in ("encoder_0", "encoder_1", "fusion")
            }
            if getattr(a, "encoder_dp2", 0):
                configs = dp2_service_configs(
                    gpus,
                    encoder_gpus,
                    env,
                    args,
                    ports,
                    MIN_WORKER_INFO_PORT_NUM,
                    getattr(a, "encoder_profile", "candidate-a"),
                    getattr(a, "encoder_rdma_pool_bytes", 17179869184),
                )
            else:
                configs = separated_service_configs(
                    gpus,
                    encoder_gpus,
                    profile,
                    ports,
                    MIN_WORKER_INFO_PORT_NUM,
                    moe_strategy,
                    decode_graph,
                    fp8_kv_cache,
                    native_fp8_attn,
                    seq_size_per_block,
                )
                configs[-1]["env"], configs[-1]["args"] = throughput_config(
                    configs[-1]["env"],
                    configs[-1]["args"],
                    policy,
                    getattr(a, "decode_prefill_ratio", "0"),
                    getattr(a, "schedule_trace", 0),
                    getattr(a, "trace_run_id", "unset"),
                    getattr(a, "coord_mode", "off"),
                )
                configs[-1]["env"], configs[-1]["args"] = capacity_config(
                    configs[-1]["env"],
                    configs[-1]["args"],
                    getattr(a, "rank_concurrency", 4),
                    getattr(a, "kv_cache_mb", 16384),
                    getattr(a, "graph_batches", ""),
                    decode_graph,
                    getattr(a, "runtime_reserve_mb", 24576),
                )
            for config in configs:
                if config["name"].startswith("encoder_dp"):
                    config["env"]["RTP_MM_METRICS_JSONL_DIR"] = str(
                        out / "encoder-metrics"
                    )
                    config["env"]["CUDA_COREDUMP_FILE"] = str(
                        out / "cuda_coredump_%h.%p.%t"
                    )
                m = MagaServerManager(
                    env_args=config["env"],
                    device_ids=config["gpus"],
                    port=config["port"],
                    role_name=config["name"],
                    smoke_args_str=config["args"],
                )
                managers.append((config["name"], m))
            manager = managers[-1][1]
            report.update(
                configs=configs,
                encoder_worker_count=len(encoder_gpu_ids(encoder_gpus)),
                topology=(
                    f"ENCODER_DP{len(encoder_gpu_ids(encoder_gpus))}+PDFUSION4"
                    if getattr(a, "encoder_dp2", 0)
                    else f"E{len(encoder_gpu_ids(encoder_gpus))}+PDFUSION4"
                ),
                env=configs[-1]["env"],
                args=configs[-1]["args"],
            )
        else:
            manager = MagaServerManager(
                env_args=env,
                device_ids=gpu_ids(gpus),
                role_name="fusion",
                smoke_args_str=args,
            )
            managers.append(("fusion", manager))
            report["topology"] = (
                "PDFUSION4_TEXT" if workload == "text" else "EPDFUSION4"
            )
        report.update(port=manager.port, phase="model_initialization")
        save(out / "result.json", report)

        # Each manager owns a distinct GPU subset, log directory and port range.
        def start_service(item):
            name, m = item
            return name, m.start_server(
                a.model_dir,
                "qwen35_moe",
                a.model_dir,
                log_to_file=True,
                timeout=getattr(a, "startup_timeout", 1600),
            )

        if len(managers) == 1:
            ready = dict(map(start_service, managers))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(managers)
            ) as pool:
                ready = dict(pool.map(start_service, managers))
        report["readiness"] = ready
        if not all(ready.values()):
            raise RuntimeError("service readiness failed: " + str(ready))
        report["pids"] = {name: m.server_pid for name, m in managers}
        report["pid"] = manager.server_pid
        if getattr(a, "coord_checks", 0):
            report["phase"] = "coordinator_checks"
            save(out / "result.json", report)
            report["coordinator_checks"] = run_coordinator_checks(
                manager,
                a.model_dir,
                out,
                decode_graph,
                getattr(a, "coord_fault", "none"),
            )
        url = f"http://127.0.0.1:{manager.port}/v1/chat/completions"
        phases = [(f"single_{workload}", 1), (f"concurrent_{workload}", 4)]
        if repeats:
            phases = [("warmup", 4)] + [
                (f"{name}_{i + 1}", count)
                for i in range(repeats)
                for name, count in (
                    (f"single_{workload}", 1),
                    (f"concurrent_{workload}", 4),
                )
            ]
        if getattr(a, "graph_edge_cases", 0):
            phases = (
                [("edge_batch3", 1)]
                + phases
                + [("edge_staggered", 4), ("edge_reuse", 1)]
            )
        if throughput:
            phases = (
                [("check_c4", 4, 4), ("check_c8", 8, 8), ("check_c16", 16, 16)]
                if throughput == 2
                else throughput_plan(repeats)
            )
        if capacity_batches:
            phases = capacity_plan(capacity_batches, a.capacity_repeats)
        if steady_concurrency:
            phases = [
                (f"steady_c{count}", count)
                for count in positive_sizes(steady_concurrency)
            ]
        if getattr(a, "quality_probe", 0):
            report["quality_before"] = run_quality_probe(
                manager, a.model_dir, out, "before"
            )
        if steady_concurrency and expected.get("fixed_output_tokens"):
            probe_payload = dict(payload, debug_info=True)
            probes = run_throughput_batch(manager, probe_payload, expected, 4, 4)
            save(out / "fixed-length-preflight.json", probes)
            if not all(row["ok"] for row in probes):
                raise RuntimeError(
                    "fixed-length preflight failed; no throughput measurement started"
                )
        if steady_concurrency and encoder_gpus:
            warmup = run_throughput_batch(manager, payload, expected, 4, 4)
            save(out / "encoder-preflight.json", warmup)
            if not all(row["ok"] for row in warmup):
                raise RuntimeError("encoder video preflight failed")
            if not all(
                encoder_log_evidence(out, len(encoder_gpu_ids(encoder_gpus))).values()
            ):
                raise RuntimeError(
                    "all configured Encoder workers must complete preflight"
                )
        capacity_refined = False
        encoder_warmup_batches = 0
        while phases:
            item = phases.pop(0)
            phase, concurrency = item[:2]
            count = item[2] if len(item) == 3 else concurrency
            report["phase"] = phase
            save(out / "result.json", report)
            wall_started = time.time()
            started = time.monotonic()
            if steady_concurrency:
                results, steady = run_steady_load(
                    manager,
                    payload,
                    expected,
                    concurrency,
                    a.steady_warmup_seconds,
                    a.steady_window_seconds,
                    a.steady_windows,
                    out,
                    phase,
                )
                count = len(results)
                report["steady"][phase] = steady
                save(out / "steady-summary.json", report["steady"])
            elif throughput or capacity_batches:
                results = run_throughput_batch(
                    manager, payload, expected, concurrency, count
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:

                    def send(i):
                        target = url
                        if phase.startswith("edge_"):
                            from rtp_llm.config.py_config_modules import (
                                MIN_WORKER_INFO_PORT_NUM,
                            )

                            rank = i if phase == "edge_staggered" else 0
                            if phase == "edge_staggered":
                                time.sleep(i * 2)
                            target = f"http://127.0.0.1:{manager.port + rank * MIN_WORKER_INFO_PORT_NUM}/v1/chat/completions"
                        is_batch_edge = phase in ("edge_batch3", "edge_reuse")
                        return request(
                            target,
                            edge_payload if is_batch_edge else payload,
                            i,
                            edge_expected if is_batch_edge else expected,
                        )

                    results = list(pool.map(send, range(count)))
            elapsed = time.monotonic() - started
            for result in results:
                result["phase"] = phase
                n = result.get("usage", {}).get("completion_tokens", 0)
                if n > 1 and "ttft_s" in result:
                    result["tpot_ms"] = (
                        1000 * (result["e2e_s"] - result["ttft_s"]) / (n - 1)
                    )
            report["requests"].extend(results)
            report["batches"].append(
                {
                    "phase": phase,
                    "count": count,
                    "concurrency": concurrency,
                    "wall_s": elapsed,
                    "started_at": wall_started,
                    "ended_at": time.time(),
                    "output_tokens_per_s": sum(
                        r.get("usage", {}).get("completion_tokens", 0) for r in results
                    )
                    / elapsed,
                }
            )
            save(out / "result.json", report)
            save(out / (phase + ".json"), results)
            if throughput or capacity_batches or steady_concurrency:
                observed = execution_evidence(out)
                save(out / (phase + "-execution.json"), observed)
                if capacity_batches:
                    report["capacity"] = capacity_summary(report, observed)
                    save(out / "capacity.json", report["capacity"])
                    if (
                        getattr(a, "capacity_refine", 0)
                        and not phases
                        and not capacity_refined
                    ):
                        phases.extend(
                            capacity_refinement(
                                report["capacity"],
                                positive_sizes(capacity_batches),
                                max(2, a.capacity_repeats),
                                a.rank_concurrency,
                            )
                        )
                        capacity_refined = True
            if not all(r["ok"] for r in results):
                raise RuntimeError(phase + " failed response validation")
            if encoder_gpus and phase.startswith("warmup"):
                encoder_warmup_batches += 1
                evidence = encoder_log_evidence(out, len(encoder_gpu_ids(encoder_gpus)))
                report["encoder_warmup_evidence"] = evidence
                if not all(evidence.values()):
                    if encoder_warmup_batches >= 4:
                        raise RuntimeError(
                            "all configured Encoders must complete warmup before performance measurement"
                        )
                    phases.insert(
                        0, (f"warmup_encoder_{encoder_warmup_batches + 1}", 4)
                    )
        if getattr(a, "quality_probe", 0):
            report["quality_after"] = run_quality_probe(
                manager, a.model_dir, out, "after"
            )
        if encoder_gpus:
            report["phase"] = "encoder_evidence"
            for _ in range(20):
                evidence = encoder_log_evidence(out, len(encoder_gpu_ids(encoder_gpus)))
                if all(evidence.values()):
                    break
                time.sleep(0.25)
            report["encoder_evidence"] = evidence
            save(out / "encoder-evidence.json", evidence)
            if not all(evidence.values()):
                raise RuntimeError(
                    "all configured Encoder services must process video successfully"
                )
        if moe_strategy == "mega_moe_fp8":
            evidence = execution_evidence(out)
            report["execution_evidence"] = evidence
            save(out / "execution-evidence.json", evidence)
            errors = validate_execution_evidence(
                evidence, decode_graph, getattr(a, "graph_edge_cases", 0)
            )
            if native_fp8_attn:
                for rank, phases in evidence["native_fp8"].items():
                    if not all(phases.values()):
                        errors.append(
                            f"rank {rank}: missing native FP8 Prefill/Decode invocation evidence"
                        )
            if errors:
                raise RuntimeError("; ".join(errors))
        report["performance"] = performance_summary(report)
        save(out / "performance.json", report["performance"])
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(
            status="FAILED", error=repr(error), traceback=traceback.format_exc()
        )
    finally:
        if moe_strategy == "mega_moe_fp8":
            try:
                report["execution_evidence"] = execution_evidence(out)
                save(out / "execution-evidence.json", report["execution_evidence"])
            except Exception as error:
                report["evidence_error"] = repr(error)
        if manager is not None:
            report["server_exit_code_before_cleanup"] = manager.exit_code
        report["service_exit_codes_before_cleanup"] = {
            name: m.exit_code for name, m in managers
        }
        for name, m in reversed(managers):
            try:
                m.stop_server()
            except Exception as error:
                report.setdefault("cleanup_errors", {})[name] = repr(error)
                report["status"] = "FAILED"
        stop.set()
        thread.join(timeout=35)
        save(out / "result.json", report)
    return 0 if report["status"] == "PASSED" else 1


def bazel_command(a):
    gpus = ",".join(map(str, gpu_ids(a.gpus)))
    encoder_gpus = getattr(a, "encoder_gpus", "")
    selected = selected_gpus(gpus, encoder_gpus)
    count = len(gpu_ids(selected, count=None))
    command = [
        "bazelisk",
        "--batch",
        "--output_user_root=" + str(Path(a.cache_root).resolve()),
        "test",
        (
            (
                SINGLE_ENCODER_TARGET
                if len(encoder_gpu_ids(encoder_gpus)) == 1
                else ENCODER_TARGET
            )
            if encoder_gpus
            else (TEXT_TARGET if getattr(a, "workload", "video") == "text" else TARGET)
        ),
        "--config=cuda13",
        "--config=sm10x",
        "--action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3",
        "--host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3",
        "--disk_cache=/root/.cache/disk_cache",
        "--config=daily_aone_bazel_cache",
        "--remote_header=x-aone-bazel-api-key=ai-infra-cicd",
        "--run_under=//rtp_llm/test/utils:gpu_lock",
        "--test_output=all",
        "--nocache_test_results",
        "--test_timeout=" + str(getattr(a, "test_timeout_seconds", 7200)),
        "--test_env=CUDA_VISIBLE_DEVICES=" + selected,
        "--test_env=WORLD_SIZE=" + str(count),
        "--test_env=GPU_COUNT=" + str(count),
        "--test_env=RTP_GPU_ACQUIRE_TIMEOUT=360",
        "--test_env=PYTHONNOUSERSITE=1",
        "--test_env=RTP_GPU_QUERY_TIMEOUT=60",
        "--test_env=RTP_GPU_PASSIVE_LOCK=1",
    ]
    command += a.bazel_option
    for name, value in (
        ("encoder-dp2", getattr(a, "encoder_dp2", 0)),
        ("encoder-profile", getattr(a, "encoder_profile", "candidate-a")),
        ("encoder-rdma-pool-bytes", getattr(a, "encoder_rdma_pool_bytes", 17179869184)),
        ("model-dir", a.model_dir),
        ("data-dir", a.data_dir),
        ("output", str(Path(a.output).resolve() / "smoke")),
        ("gpus", gpus),
        ("profile", a.profile),
        ("workload", getattr(a, "workload", "video")),
        ("perf-repeats", a.perf_repeats),
        ("encoder-gpus", encoder_gpus),
        ("moe-strategy", getattr(a, "moe_strategy", "fp8_per_block_ep_normal")),
        ("decode-graph", getattr(a, "decode_graph", 0)),
        ("fp8-kv-cache", getattr(a, "fp8_kv_cache", 0)),
        ("native-fp8-attn", getattr(a, "native_fp8_attn", 0)),
        ("seq-size-per-block", getattr(a, "seq_size_per_block", 0)),
        ("graph-edge-cases", getattr(a, "graph_edge_cases", 0)),
        ("scheduler-policy", getattr(a, "scheduler_policy", "fifo")),
        ("decode-prefill-ratio", getattr(a, "decode_prefill_ratio", "0")),
        ("schedule-trace", getattr(a, "schedule_trace", 0)),
        ("trace-run-id", getattr(a, "trace_run_id", "unset")),
        ("coord-mode", getattr(a, "coord_mode", "off")),
        ("coord-checks", getattr(a, "coord_checks", 0)),
        ("coord-fault", getattr(a, "coord_fault", "none")),
        ("throughput-sweep", getattr(a, "throughput_sweep", 0)),
        ("rank-concurrency", getattr(a, "rank_concurrency", 4)),
        ("kv-cache-mb", getattr(a, "kv_cache_mb", 16384)),
        ("runtime-reserve-mb", getattr(a, "runtime_reserve_mb", 24576)),
        ("graph-batches", getattr(a, "graph_batches", "")),
        ("capacity-batches", getattr(a, "capacity_batches", "")),
        ("capacity-repeats", getattr(a, "capacity_repeats", 2)),
        ("capacity-refine", getattr(a, "capacity_refine", 0)),
        ("steady-concurrency", getattr(a, "steady_concurrency", "")),
        ("steady-warmup-seconds", getattr(a, "steady_warmup_seconds", 240)),
        ("steady-window-seconds", getattr(a, "steady_window_seconds", 300)),
        ("steady-windows", getattr(a, "steady_windows", 2)),
        ("client-mode", getattr(a, "client_mode", "legacy")),
        ("client-processes", getattr(a, "client_processes", 1)),
        ("quality-probe", getattr(a, "quality_probe", 0)),
        ("sse-chunk-size", getattr(a, "sse_chunk_size", 1)),
        ("client-profile-requests", getattr(a, "client_profile_requests", 0)),
        ("fixed-output-tokens", getattr(a, "fixed_output_tokens", 0)),
        ("startup-timeout", getattr(a, "startup_timeout", 1600)),
        ("test-timeout-seconds", getattr(a, "test_timeout_seconds", 7200)),
    ):
        command.append("--test_arg=--" + name + "=" + str(value))
    return command


def run_launcher(a):
    repo = Path(__file__).resolve().parents[3]
    # Validate experiment arguments before acquiring GPUs or starting model workers.
    throughput_config(
        {},
        "",
        a.scheduler_policy,
        a.decode_prefill_ratio,
        a.schedule_trace,
        a.trace_run_id,
        a.coord_mode,
    )
    command = bazel_command(a)
    public = [
        "--remote_header=<redacted>" if x.startswith("--remote_header=") else x
        for x in command
    ]
    print(shlex.join(public), flush=True)
    if not a.execute:
        return 0
    if a.workload == "video":
        validate_fixture(Path(a.data_dir))
    if not Path(a.model_dir).is_dir():
        raise ValueError("model directory does not exist")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "command.json", public)
    snapshot = out / "source-snapshot"
    snapshot.mkdir()
    fingerprints = {}
    for relative in (
        "rtp_llm/start_server.py",
        "rtp_llm/test/smoke/qwen35_epd_fusion_4gpu_smoke.bzl",
        "rtp_llm/test/smoke/qwen35_epd_fusion_4gpu_smoke.py",
        "rtp_llm/test/smoke/qwen35_epd_fusion_4gpu_smoke_test.py",
        "rtp_llm/cpp/config/ConfigModules.h",
        "rtp_llm/cpp/config/ConfigModules.cc",
        "rtp_llm/cpp/pybind/ConfigInit.cc",
        "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.cc",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionCoordinatedScheduler.h",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionCoordinatedScheduler.cc",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h",
        "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.cc",
        "rtp_llm/cpp/normal_engine/NormalExecutor.h",
        "rtp_llm/test/smoke/pdfusion_schedule_trace.py",
        "rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py",
        "rtp_llm/cpp/normal_engine/NormalEngine.h",
        "rtp_llm/cpp/normal_engine/NormalEngine.cc",
        "rtp_llm/server/server_args/fifo_scheduler_group_args.py",
        "rtp_llm/cpp/models/PyWrappedModel.cc",
        "rtp_llm/cpp/models/PyWrappedModel.h",
        "rtp_llm/models_py/modules/factory/attention/cuda_impl/trtllm_gen.py",
    ):
        data = (repo / relative).read_bytes()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        fingerprints[relative] = hashlib.sha256(data).hexdigest()
    save(out / "source-sha256.json", fingerprints)
    save(
        out / "source.json",
        {
            label: subprocess.check_output(cmd, cwd=repo, text=True)
            for label, cmd in {
                "external_head": ["git", "rev-parse", "HEAD"],
                "external_status": ["git", "status", "--short"],
                "internal_head": ["git", "-C", str(repo.parent), "rev-parse", "HEAD"],
                "internal_status": ["git", "-C", str(repo.parent), "status", "--short"],
            }.items()
        },
    )
    rc = 1
    stage = "precheck"
    try:
        precheck = (
            repo / "internal_source/.cursor/skills/test-execution/pre_build_check.sh"
        )
        with (out / "precheck.log").open("w") as log:
            subprocess.run(
                [
                    "bash",
                    str(precheck),
                    "local",
                    str(repo.parent),
                    "--output-user-root=" + a.cache_root,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        stage = "gpu_availability"
        save(
            out / "gpu-precheck.json",
            check_gpu_idle(selected_gpus(a.gpus, a.encoder_gpus)),
        )
        stage = "bazel_test"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=selected_gpus(a.gpus, a.encoder_gpus),
            PYTHONNOUSERSITE="1",
        )
        with (out / "bazel.log").open("w") as log:
            rc = subprocess.run(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
    except BaseException as error:
        (out / "launcher-error.txt").write_text(traceback.format_exc())
        save(
            out / "result.json",
            {"status": "INFRA_ERROR", "phase": stage, "error": repr(error)},
        )
        raise
    finally:
        save(out / "exit.json", {"exit_code": rc})
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--startup-timeout", type=int, default=1600)
    p.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=7200,
        help="Explicit Bazel test budget, including steady phases and cleanup (3600–21600 seconds)",
    )
    p.add_argument("--client-mode", choices=("legacy", "measured"), default="legacy")
    p.add_argument("--client-processes", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--quality-probe", type=int, choices=(0, 1), default=0)
    p.add_argument("--sse-chunk-size", type=int, default=1)
    p.add_argument("--client-profile-requests", type=int, default=0)
    p.add_argument("--fixed-output-tokens", type=int, default=0)
    p.add_argument("--model-dir", default=DEFAULT_MODEL)
    p.add_argument("--gpus", default="0,1,2,3", help="four PDFUSION GPUs")
    p.add_argument("--workload", choices=("video", "text"), default="video")
    p.add_argument(
        "--encoder-dp2",
        "--encoder-proxy",
        dest="encoder_dp2",
        type=int,
        choices=(0, 1),
        default=0,
        help="one ViT proxy with one worker per Encoder GPU; legacy option retained",
    )
    p.add_argument(
        "--encoder-profile", choices=tuple(ENCODER_PROFILES), default="candidate-a"
    )
    p.add_argument(
        "--encoder-rdma-pool-bytes",
        type=int,
        default=17179869184,
        help="RDMA output pool cap in bytes per Encoder worker",
    )
    p.add_argument(
        "--encoder-gpus",
        default="",
        help="one or two separate Encoder GPUs; selects the matching five/six GPU target",
    )
    p.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "qwen35_e2p4d2_data"),
    )
    p.add_argument("--output", required=True)
    p.add_argument("--cache-root", default=DEFAULT_CACHE)
    p.add_argument("--bazel-option", action="append", default=[])
    p.add_argument(
        "--profile", choices=("baseline", "fused", "flashinfer"), default="baseline"
    )
    p.add_argument(
        "--perf-repeats",
        type=int,
        default=0,
        help="warm up with four requests, then repeat single/four-concurrent measurements",
    )
    p.add_argument(
        "--moe-strategy",
        choices=("fp8_per_block_ep_normal", "mega_moe_fp8"),
        default="fp8_per_block_ep_normal",
    )
    p.add_argument("--decode-graph", type=int, choices=(0, 1), default=0)
    p.add_argument("--fp8-kv-cache", type=int, choices=(0, 1), default=0)
    p.add_argument("--native-fp8-attn", type=int, choices=(0, 1), default=0)
    p.add_argument(
        "--seq-size-per-block",
        type=int,
        choices=(0, 2048, 4096),
        default=0,
        help="0 selects 4096 for FP8 KV, otherwise 2048; kernel page remains 64",
    )
    p.add_argument("--graph-edge-cases", type=int, choices=(0, 1), default=0)
    p.add_argument(
        "--scheduler-policy", choices=("fifo", "prefill-first"), default="fifo"
    )
    p.add_argument("--decode-prefill-ratio", default="0")
    p.add_argument("--schedule-trace", type=int, choices=(0, 1), default=0)
    p.add_argument("--trace-run-id", default="unset")
    p.add_argument("--coord-mode", choices=("off", "cadence"), default="off")
    p.add_argument("--coord-checks", type=int, choices=(0, 1), default=0)
    p.add_argument("--coord-fault", choices=("none", "peer-exit"), default="none")
    p.add_argument(
        "--throughput-sweep",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="1: warmed two-round throughput sweep; 2: correctness-only 4/8/16 request bursts",
    )
    p.add_argument("--rank-concurrency", type=int, default=4)
    p.add_argument(
        "--kv-cache-mb",
        type=int,
        default=16384,
        help="0 uses engine automatic KV sizing with the existing runtime reserve",
    )
    p.add_argument(
        "--runtime-reserve-mb",
        type=int,
        default=24576,
        help="PD runtime memory reserve in MiB; reduces automatically sized KV allocation",
    )
    p.add_argument("--graph-batches", default="")
    p.add_argument(
        "--capacity-batches",
        default="",
        help="per-rank batch sizes to probe; all four ranks receive that many requests",
    )
    p.add_argument("--capacity-repeats", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--capacity-refine", type=int, choices=(0, 1), default=0)
    p.add_argument(
        "--steady-concurrency",
        default="",
        help="total closed-loop client concurrency levels",
    )
    p.add_argument("--steady-warmup-seconds", type=int, default=240)
    p.add_argument("--steady-window-seconds", type=int, default=300)
    p.add_argument("--steady-windows", type=int, choices=(2, 3, 4), default=2)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not 3600 <= a.test_timeout_seconds <= 21600:
        p.error("test timeout must be between 3600 and 21600 seconds")
    if a.client_processes > 1 and (
        a.client_mode != "measured" or not a.steady_concurrency
    ):
        p.error("multiple client processes require measured steady load")
    if a.steady_concurrency and any(
        n % a.client_processes for n in positive_sizes(a.steady_concurrency)
    ):
        p.error("steady concurrency must divide evenly across client processes")
    for field in ("model_dir", "data_dir", "output", "cache_root"):
        setattr(a, field, str(Path(getattr(a, field)).resolve()))
    selected_gpus(a.gpus, a.encoder_gpus)
    env, args = server_config(
        a.gpus,
        a.profile,
        a.moe_strategy,
        a.decode_graph,
        a.fp8_kv_cache,
        a.native_fp8_attn,
        a.seq_size_per_block,
    )
    capacity_config(
        env,
        args,
        a.rank_concurrency,
        a.kv_cache_mb,
        a.graph_batches,
        a.decode_graph,
        a.runtime_reserve_mb,
    )
    if a.steady_concurrency:
        levels = positive_sizes(a.steady_concurrency)
        if any(n % 4 or n > 4 * a.rank_concurrency for n in levels):
            p.error(
                "steady concurrency must be divisible by four and fit configured client/rank limits"
            )
        if (
            not (
                a.workload == "text"
                and not a.encoder_gpus
                or a.workload == "video"
                and a.encoder_gpus
                and a.encoder_dp2
            )
            or a.graph_edge_cases
            or a.throughput_sweep
            or a.capacity_batches
            or a.perf_repeats
        ):
            p.error(
                "steady load requires text PDFUSION or video Encoder proxy without other suites"
            )
        if a.steady_warmup_seconds < 0 or a.steady_window_seconds < 60:
            p.error(
                "steady warmup must be nonnegative; measurement windows must be at least 60 seconds"
            )
        if (
            len(levels)
            * (
                a.steady_warmup_seconds
                + a.steady_window_seconds * a.steady_windows
                + 600
            )
            > a.test_timeout_seconds - 1800
        ):
            p.error(
                "steady load plus drain reserve exceeds the configured test budget (1800 seconds reserved for startup)"
            )
    if a.coord_fault != "none" and not a.coord_checks:
        raise ValueError(
            "fault injection requires explicit coordinator correctness checks"
        )
    if a.coord_checks and (
        a.coord_mode != "cadence"
        or a.steady_concurrency
        or a.capacity_batches
        or a.throughput_sweep
        or a.perf_repeats
    ):
        raise ValueError(
            "coordinator checks require cadence correctness run without performance measurement"
        )
    if a.capacity_batches:
        if (
            max(positive_sizes(a.capacity_batches)) > a.rank_concurrency
            or a.rank_concurrency < 4
        ):
            p.error(
                "capacity batches must fit the configured rank concurrency (at least 4)"
            )
        if (
            a.workload != "text"
            or a.encoder_gpus
            or a.graph_edge_cases
            or a.throughput_sweep
        ):
            p.error(
                "capacity sweep requires standalone text PDFUSION without other suites"
            )
    if a.workload == "text" and a.encoder_gpus:
        p.error("text-only PD must not allocate Encoder GPUs")
    if a.throughput_sweep and (
        a.encoder_gpus or a.workload != "text" or a.graph_edge_cases
    ):
        p.error("throughput experiment requires text PDFUSION without graph edge suite")
    if a.encoder_dp2 and (not a.encoder_gpus or a.workload != "video"):
        p.error("Encoder proxy requires one or two encoder GPUs and video workload")
    if not 0 <= a.perf_repeats <= 4:
        p.error("--perf-repeats must be between 0 and 4 (7200-second test budget)")
    return run_worker(a) if a.worker else run_launcher(a)


if __name__ == "__main__":
    raise SystemExit(main())
