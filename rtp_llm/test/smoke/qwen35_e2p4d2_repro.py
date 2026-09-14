"""E2/P4/D2 expansion benchmark, E on GPU5 and GPU0."""

import argparse
import concurrent.futures
import copy
import json
import logging
import os
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)


def save(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, default=str))


def request(url, p, index):
    start = time.perf_counter()
    r = {
        "index": index,
        "ok": False,
        "content": "",
        "usage": {},
        "request_id": str(uuid.uuid4()),
    }
    try:
        with requests.post(
            url,
            json=p,
            headers={"X-Request-ID": r["request_id"]},
            stream=True,
            timeout=(10, 1200),
        ) as response:
            r["http_status"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError(response.text[:4000])
            for line in response.iter_lines(chunk_size=1):
                if not line or not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    r["done"] = True
                    break
                obj = json.loads(body)
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                if obj.get("usage"):
                    r["usage"] = obj["usage"]
                if obj.get("aux_info"):
                    r["aux_info"] = obj["aux_info"]
                for c in obj.get("choices", []):
                    text = c.get("delta", {}).get("content") or ""
                    if text and "ttft_s" not in r:
                        r["ttft_s"] = time.perf_counter() - start
                    r["content"] += text
                    if c.get("finish_reason"):
                        r["finish_reason"] = c["finish_reason"]
        r["truncated"] = r.get("finish_reason") == "length"
        r["ok"] = (
            bool(r["content"])
            and bool(r.get("done") or r.get("finish_reason"))
            and (not r["truncated"])
        )
        if r["ok"] and (
            r["usage"].get("prompt_tokens") != 24422
            or r["usage"].get("prompt_tokens_details", {}).get("video_tokens") != 20240
            or r.get("aux_info", {}).get("reuse_len") != 0
            or (not r.get("aux_info", {}).get("pd_sep"))
        ):
            r["ok"] = False
            r["validation_error"] = "Input, video, reuse, or PD evidence mismatch"
    except Exception as e:
        r["error"] = repr(e)
    r["e2e_s"] = time.perf_counter() - start
    return r


def main():
    global get_gpu_ids, MIN_WORKER_INFO_PORT_NUM, MagaServerManager, EndPoint, GroupEndPoint, ServiceRoute
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--concurrencies", default="96")
    parser.add_argument("--min-seconds", type=float, default=180)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--reserve-mb", type=int, default=24576)
    parser.add_argument("--p-token-budget", type=int, default=20000)
    parser.add_argument("--d-seq-limit", type=int, default=96)
    parser.add_argument("--e-batch", type=int, default=1)
    parser.add_argument("--fixed-tasks", type=int, default=0)
    parser.add_argument("--d-kv-mb", type=int, default=49152)
    parser.add_argument("--d-reserve-mb", type=int, default=8192)
    parser.add_argument("--long-repeats", type=int, default=3)
    a = parser.parse_args()
    a.graph_diagnostic = False
    a.runtime_diagnostic = False
    from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
    from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
    from rtp_llm.test.utils.device_resource import get_gpu_ids
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager

    MODEL = str(Path(a.model_dir).resolve())
    data = Path(a.data_dir).resolve()
    import hashlib

    manifest = json.loads((data / "manifest.json").read_text())
    for filename, expected in manifest["assets_sha256"].items():
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Fixture checksum mismatch: " + filename)
    if not Path(MODEL).is_dir():
        raise ValueError("Model directory does not exist")
    O = Path(a.output)
    O.mkdir(parents=True, exist_ok=False)
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = str(O)
    os.environ["OMP_NUM_THREADS"] = "8"
    assert sorted(get_gpu_ids()) == [0, 1, 2, 3, 4, 5, 6, 7], get_gpu_ids()
    max_seq = (30720, max_seq)
    roles = [
        ("vit0", [5], "VIT"),
        ("vit1", [0], "VIT"),
        ("prefill", [1, 2, 3, 4], "PREFILL"),
        ("decode", [6, 7], "DECODE"),
    ]
    ports = {name: int(MagaServerManager.get_free_port()) for (name, _, _) in roles}
    assert len(set(ports.values())) == 4

    def ep(name):
        count = {"vit0": 1, "vit1": 1, "prefill": 4, "decode": 2}[name]
        return EndPoint(
            type="Vipserver",
            address=",".join(
                (
                    "127.0.0.1:" + str(ports[name] + i * MIN_WORKER_INFO_PORT_NUM)
                    for i in range(count)
                )
            ),
            protocol="http",
            path="/",
        )

    route = ServiceRoute(
        service_id="qwen35_offline_native_video",
        use_local=True,
        role_endpoints=[
            GroupEndPoint(
                group="default",
                vit_endpoint=EndPoint(
                    type="Vipserver",
                    address="127.0.0.1:"
                    + str(ports["vit0"])
                    + ",127.0.0.1:"
                    + str(ports["vit1"]),
                    protocol="http",
                    path="/",
                ),
                prefill_endpoint=ep("prefill"),
                decode_endpoint=ep("decode"),
            )
        ],
    ).model_dump_json()
    common = f"--use_local 1 --tp_size 1 --act_type BF16 --seq_size_per_block 2048 --kernel_seq_size_per_block 64 --max_seq_len {max_seq} --warm_up 0 --concurrency_limit 96 --reserver_runtime_mem_mb {a.reserve_mb} --reuse_cache 0 --mm_cache_item_num 0 --url_cache_item_num 0 --use_deepep_moe 1 --use_all_gather 0 --use_deepep_internode 0 --fp8_kv_cache 0 --load_cache_timeout_ms 60000"
    managers = []
    configs = []
    for name, gpus, role in roles:
        n = len(gpus)
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "WORLD_SIZE": str(n),
            "TP_SIZE": "1",
            "DP_SIZE": str(n),
            "EP_SIZE": str(n),
            "ROLE_TYPE": role,
            "VIT_SEPARATION": "1" if role == "VIT" else "2",
            "MODEL_SERVICE_CONFIG": route,
            "QWEN35_BENCH_NATIVE_VIDEO": "1",
            "QWEN35_BENCH_VIDEO_AUDIT": str(O / (name + "-video-audit.jsonl")),
            "MM_CACHE_ITEM_NUM": "0",
            "URL_CACHE_ITEM_NUM": "0",
            "REUSE_CACHE": "0",
            "OMP_NUM_THREADS": "8",
        }
        if role == "DECODE" and a.graph_diagnostic:
            env["LOG_LEVEL"] = "DEBUG"
        if role != "VIT":
            env.update(
                REMOTE_RPC_SERVER_IP="127.0.0.1",
                REMOTE_SERVER_PORT=str(
                    ports["decode" if role == "PREFILL" else "prefill"]
                ),
            )
        if role == "VIT":
            args = (
                "--use_local 1 --role_type VIT --vit_separation 1 --vit_server_count 1 --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 --act_type BF16 --warm_up 0 --concurrency_limit 96 --mm_cache_item_num 0 --url_cache_item_num 0 --gpu_max_batch_size "
                + str(a.e_batch)
            )
        else:
            args = (
                common
                + f" --role_type {role} --dp_size {n} --ep_size {n} --world_size {n} --vit_separation 2"
            )
            args += (
                " --moe_strategy fp8_per_block_ep_low_latency --use_deepep_low_latency 1 --enable_cuda_graph 1 --decode_capture_config 1,2,4,8,16,32,48,64,96"
                if role == "DECODE"
                else " --moe_strategy fp8_per_block_ep_normal --use_deepep_low_latency 0 --enable_cuda_graph 0 --max_batch_tokens_size 40000 --max_batch_tokens_without_cache 40000"
            )
        if role == "PREFILL":
            args = args.replace(
                "--max_batch_tokens_size 40000",
                "--max_batch_tokens_size " + str(a.p_token_budget),
            ).replace(
                "--max_batch_tokens_without_cache 40000",
                "--max_batch_tokens_without_cache " + str(a.p_token_budget),
            )
        if role == "DECODE":
            if a.d_kv_mb is not None:
                args += " --kv_cache_mem_mb " + str(a.d_kv_mb)
            if a.d_reserve_mb is not None:
                args = args.replace(
                    "--reserver_runtime_mem_mb " + str(a.reserve_mb),
                    "--reserver_runtime_mem_mb " + str(a.d_reserve_mb),
                )
            args = args.replace(
                "--concurrency_limit 96", "--concurrency_limit " + str(a.d_seq_limit)
            )
            if a.d_seq_limit > 96:
                args = args.replace(
                    "--decode_capture_config 1,2,4,8,16,32,48,64,96",
                    "--decode_capture_config 1,2,4,8,16,32,48,64,96,128,192",
                )
        m = MagaServerManager(
            env_args=env, port=ports[name], role_name=name, smoke_args_str=args
        )
        managers.append(m)
        configs.append(
            {"role": name, "gpus": gpus, "env": env, "args": args, "port": ports[name]}
        )
    messages = json.loads((data / "messages.json").read_text())
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") == "video_url":
                item["video_url"]["url"] = str(data / "video.mp4")
    payload = {
        "model": "qwen35",
        "messages": messages,
        "max_tokens": 32 if a.graph_diagnostic else 4096,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "stream": True,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    save(O / "request.json", payload)
    report = {
        "status": "STARTING",
        "configs": configs,
        "model": MODEL,
        "max_seq_len": max_seq,
        "requests": [],
        "throughput_not_run": True,
        "graph_diagnostic": a.graph_diagnostic,
    }
    save(O / "result.json", report)
    stop = threading.Event()

    def monitor():
        with (O / "resource-samples.jsonl").open("w") as f:
            while not stop.is_set():
                try:
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.used,utilization.gpu",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    )
                    f.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "status": report["status"],
                                "gpu": gpu,
                                "load": os.getloadavg(),
                                "meminfo": Path("/proc/meminfo").read_text(),
                            }
                        )
                        + "\n"
                    )
                    f.flush()
                except Exception:
                    logging.exception("Resource monitor")
                stop.wait(5)

    def status_monitor():
        with (O / "worker-status-samples.jsonl").open("w") as f:
            while not stop.is_set():
                sample = {
                    "time": time.time(),
                    "status": report["status"],
                    "workers": {},
                }
                for role, count in [("prefill", 1), ("decode", 1)]:
                    for rank in range(count):
                        try:
                            sample["workers"][role + "/" + str(rank)] = requests.get(
                                "http://127.0.0.1:"
                                + str(ports[role] + rank * MIN_WORKER_INFO_PORT_NUM)
                                + "/worker_status",
                                timeout=1,
                            ).json()
                        except Exception as error:
                            sample["workers"][role + "/" + str(rank)] = {
                                "sampling_error": str(error)
                            }
                f.write(json.dumps(sample, default=str) + chr(10))
                f.flush()
                stop.wait(5)

    status_watcher = threading.Thread(target=status_monitor, daemon=True)
    status_watcher.start()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        begin = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            states = list(
                pool.map(
                    lambda m: m.start_server(
                        MODEL, "qwen35_moe", MODEL, log_to_file=True, timeout=900
                    ),
                    managers,
                )
            )
        report.update(
            startup_s=time.perf_counter() - begin,
            startup_states=states,
            pids=[m.server_pid for m in managers],
        )
        if not all(states):
            report["status"] = "STARTUP_FAILED"
            raise RuntimeError("One or more roles failed startup")
        report["status"] = "SINGLE_REQUEST"
        save(O / "result.json", report)
        url = "http://127.0.0.1:" + str(ports["prefill"]) + "/v1/chat/completions"
        first = request(url, payload, 0)
        report["requests"].append(first)
        save(O / "result.json", report)
        if not first["ok"] and (
            not (
                a.graph_diagnostic
                and first.get("finish_reason") == "length"
                and (not first.get("error"))
            )
        ):
            report["status"] = "SINGLE_REQUEST_FAILED"
            raise RuntimeError("Single request did not complete")
        report["status"] = "CONCURRENCY_4"
        save(O / "result.json", report)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(lambda i: request(url, payload, i), range(1, 5)):
                report["requests"].append(r)
                save(O / "result.json", report)
        report["status"] = (
            (
                "GRAPH_DIAGNOSTIC_COMPLETED"
                if all(
                    (
                        r["ok"]
                        or (r.get("finish_reason") == "length" and (not r.get("error")))
                        for r in report["requests"]
                    )
                )
                else "GRAPH_DIAGNOSTIC_FAILED"
            )
            if a.graph_diagnostic
            else (
                "FUNCTIONAL_PASSED_RUNTIME_EVIDENCE_PENDING"
                if all((r["ok"] for r in report["requests"]))
                else "FUNCTIONAL_FAILED"
            )
        )

        def e_counts():
            return {
                name: sum(
                    (
                        len(f.read_text().splitlines())
                        for f in (O / (name + "_logs")).glob("mm_access_r0_s0.log*")
                    )
                )
                for name in ["vit0", "vit1"]
            }

        if report["status"] == "FUNCTIONAL_PASSED_RUNTIME_EVIDENCE_PENDING":
            for extra in range(4):
                if min(e_counts().values()) > 0:
                    break
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    report["requests"].extend(
                        pool.map(
                            lambda i: request(
                                url, payload, "e-routing-" + str(extra) + "-" + str(i)
                            ),
                            range(4),
                        )
                    )
            report["e_route_acceptance"] = e_counts()
            save(O / "result.json", report)
            assert min(report["e_route_acceptance"].values()) > 0, report[
                "e_route_acceptance"
            ]
            assert all((r["ok"] for r in report["requests"]))
        if (
            a.benchmark
            and report["status"] == "FUNCTIONAL_PASSED_RUNTIME_EVIDENCE_PENDING"
        ):
            report["throughput_not_run"] = False
            report["phases"] = []
            concurrency_points = [int(x) for x in a.concurrencies.split(",")]
            plateau_count = 0
            for phase_index, concurrency in enumerate(concurrency_points):
                label = (
                    "c"
                    + str(concurrency)
                    + (
                        "_repeat" + str(phase_index + 1)
                        if len(concurrency_points) != len(set(concurrency_points))
                        else ""
                    )
                )
                report["status"] = "BENCHMARK_C" + str(concurrency)
                save(O / "result.json", report)
                phase = run_benchmark_phase(
                    url,
                    payload,
                    concurrency,
                    O,
                    label,
                    report,
                    a.min_seconds,
                    a.max_seconds,
                    a.min_samples,
                )
                report["phases"].append(phase)
                save(O / "result.json", report)
                if phase["cohort"]["errors"] or phase["cohort"]["truncated"]:
                    report["status"] = "BENCHMARK_FAILED"
                    break
                if phase_index and concurrency > report["phases"][-2]["concurrency"]:
                    prior = report["phases"][-2]["steady"]["total_tps"]
                    gain = phase["steady"]["total_tps"] / prior - 1 if prior > 0 else 1
                    plateau_count = plateau_count + 1 if gain < 0.03 else 0
                    phase["steady_total_tps_gain"] = gain
                    if plateau_count >= 2:
                        report["concurrency_stop_reason"] = (
                            "two_successive_gains_below_3_percent"
                        )
                        report["status"] = "BENCHMARK_COMPLETED"
                        break
                else:
                    plateau_count = 0
            else:
                report["status"] = "BENCHMARK_COMPLETED"
            if report["status"] == "BENCHMARK_COMPLETED":
                valid = [
                    p
                    for p in report["phases"]
                    if p.get("steady")
                    and (not p["sample_insufficient"])
                    and (not p["cohort"]["errors"])
                    and (not p["cohort"]["truncated"])
                ]
                if not valid:
                    raise RuntimeError("No valid measurement window")
                peak = max((p["steady"]["total_tps"] for p in valid))
                concurrency = min(
                    (
                        p["concurrency"]
                        for p in valid
                        if p["steady"]["total_tps"] >= peak * 0.97
                    )
                )
                report["selected_concurrency"] = concurrency
                report["long_phases"] = []
                for repeat in range(1, a.long_repeats + 1):
                    report["status"] = (
                        "LONG_C" + str(concurrency) + "_REPEAT" + str(repeat)
                    )
                    save(O / "result.json", report)
                    phase = run_benchmark_phase(
                        url,
                        payload,
                        concurrency,
                        O,
                        "long_c" + str(concurrency) + "_repeat" + str(repeat),
                        report,
                        630,
                        900,
                        32,
                    )
                    report["long_phases"].append(phase)
                    save(O / "result.json", report)
                    if (
                        phase["cohort"]["errors"]
                        or phase["cohort"]["truncated"]
                        or (not phase["steady"])
                        or (phase["steady"]["elapsed_s"] < 600)
                    ):
                        report["status"] = "BENCHMARK_FAILED"
                        break
                else:
                    report["status"] = "BENCHMARK_COMPLETED"
            if report["status"] == "BENCHMARK_COMPLETED" and a.fixed_tasks:
                report["status"] = "FIXED_BATCH"
                save(O / "result.json", report)
                fixed = run_benchmark_phase(
                    url,
                    payload,
                    concurrency,
                    O,
                    "fixed_batch",
                    report,
                    0,
                    a.max_seconds,
                    a.fixed_tasks,
                    a.fixed_tasks,
                )
                report["fixed_batch"] = fixed
                report["status"] = (
                    "BENCHMARK_COMPLETED"
                    if fixed["fixed_task_count_met"]
                    and (not fixed["cohort"]["errors"])
                    and (not fixed["cohort"]["truncated"])
                    else "BENCHMARK_FAILED"
                )
    except Exception:
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save(O / "result.json", report)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as cleanup_pool:
            list(cleanup_pool.map(lambda m: m.stop_server(), managers))
        stop.set()
        watcher.join(timeout=10)
        status_watcher.join(timeout=10)
        report["cleaned_up"] = True
        save(O / "result.json", report)
        print("NATIVE_VIDEO_GATE", report["status"], flush=True)
    if report["status"] not in [
        "FUNCTIONAL_PASSED_RUNTIME_EVIDENCE_PENDING",
        "GRAPH_DIAGNOSTIC_COMPLETED",
        "BENCHMARK_COMPLETED",
    ]:
        raise SystemExit(1)


def run_benchmark_phase(
    url,
    payload,
    concurrency,
    output,
    label,
    report,
    min_s=180,
    max_s=900,
    min_samples=32,
    fixed_tasks=None,
):
    """Closed-loop cohort measurement; no retry; preserve drain and failure time."""
    lock = threading.Lock()
    stop_dispatch = threading.Event()
    rows = []
    begin = time.perf_counter()
    begin_wall = time.time()
    stop_at = [None]
    stop_reason = [None]
    progress_at = [begin]
    submitted = [0]
    raw_path = output / (label + ".requests.jsonl")

    def stop(reason):
        if stop_dispatch.is_set() and reason == "request_failure_or_truncation":
            stop_reason[0] = reason
        if not stop_dispatch.is_set():
            stop_at[0] = time.perf_counter()
            stop_reason[0] = reason
            stop_dispatch.set()

    with raw_path.open("w") as raw:

        def worker(worker_id):
            sequence = 0
            while not stop_dispatch.is_set():
                if fixed_tasks is not None:
                    with lock:
                        if submitted[0] >= fixed_tasks:
                            return
                        submitted[0] += 1
                        if submitted[0] == fixed_tasks:
                            stop("fixed_task_count")
                row = request(url, payload, f"{label}-{worker_id}-{sequence}")
                sequence += 1
                row["phase"] = label
                row["client_worker"] = worker_id
                row["ended_phase_s"] = time.perf_counter() - begin
                row["started_phase_s"] = row["ended_phase_s"] - row["e2e_s"]
                usage = row.get("usage", {})
                aux = row.get("aux_info", {})
                if row.get("ok") and (
                    usage.get("prompt_tokens") != 24422
                    or usage.get("prompt_tokens_details", {}).get("video_tokens")
                    != 20240
                    or aux.get("reuse_len") != 0
                    or (not aux.get("pd_sep"))
                ):
                    row["ok"] = False
                    row["validation_error"] = (
                        "Input, video, reuse, or PD evidence mismatch"
                    )
                with lock:
                    rows.append(row)
                    raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw.flush()
                    now = time.perf_counter()
                    if not row.get("ok"):
                        stop("request_failure_or_truncation")
                    elif (
                        fixed_tasks is None
                        and now - begin >= min_s
                        and (sum((x.get("ok", False) for x in rows)) >= min_samples)
                    ):
                        stop("measurement_target_reached")
                    if now - progress_at[0] >= 15:
                        save(
                            output / "phase-progress.json",
                            {
                                "phase": label,
                                "concurrency": concurrency,
                                "elapsed_s": now - begin,
                                "completed": len(rows),
                                "complete_success": sum(
                                    (x.get("ok", False) for x in rows)
                                ),
                                "dispatch_stopped": stop_dispatch.is_set(),
                            },
                        )
                        progress_at[0] = now

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker, i) for i in range(concurrency)]
            while not all((f.done() for f in futures)):
                with lock:
                    if time.perf_counter() - begin >= max_s:
                        stop("maximum_measurement_time")
                time.sleep(0.2)
            for future in futures:
                future.result()
    end = time.perf_counter()
    dispatch_end = (stop_at[0] or end) - begin

    def metrics(selected, seconds):
        good = [x for x in selected if x.get("ok")]
        inp = sum((x["usage"]["prompt_tokens"] for x in good))
        out = sum((x["usage"]["completion_tokens"] for x in good))
        truncated = [x for x in selected if x.get("truncated")]
        return {
            "elapsed_s": seconds,
            "complete_success": len(good),
            "requests": len(selected),
            "task_per_s": len(good) / seconds,
            "input_tps": inp / seconds,
            "output_tps": out / seconds,
            "total_tps": (inp + out) / seconds,
            "input_tokens": inp,
            "output_tokens": out,
            "success_rate": len(good) / len(selected) if selected else None,
            "truncated": len(truncated),
            "truncation_rate": len(truncated) / len(selected) if selected else None,
            "truncated_input_tokens": sum(
                (x.get("usage", {}).get("prompt_tokens", 0) for x in truncated)
            ),
            "truncated_output_tokens": sum(
                (x.get("usage", {}).get("completion_tokens", 0) for x in truncated)
            ),
            "errors": sum(
                (not x.get("ok") and (not x.get("truncated")) for x in selected)
            ),
        }

    result = {
        "label": label,
        "concurrency": concurrency,
        "started_wall": begin_wall,
        "fixed_tasks": fixed_tasks,
        "fixed_task_count_met": (
            len(rows) == fixed_tasks if fixed_tasks is not None else None
        ),
        "dispatch_s": dispatch_end,
        "drain_s": end - begin - dispatch_end,
        "stop_reason": stop_reason[0],
        "cohort": metrics(rows, end - begin),
        "steady_definition": "Full logical requests completing in [30s, dispatch stop]; failure time retained.",
        "steady": (
            metrics(
                [x for x in rows if 30 <= x["ended_phase_s"] <= dispatch_end],
                dispatch_end - 30,
            )
            if dispatch_end > 30
            else None
        ),
        "sample_insufficient": sum((x.get("ok", False) for x in rows)) < min_samples,
        "raw_requests": str(raw_path),
    }
    save(output / (label + ".summary.json"), result)
    return result


if __name__ == "__main__":
    main()
