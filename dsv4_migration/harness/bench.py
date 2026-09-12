"""Real-prefill fixed-cohort decode benchmark with explicit failure accounting."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import psutil
import pynvml
import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scheme", choices=("vanilla", "offload"))
    parser.add_argument("--corpus", default="/tmp/dsv4-long-corpus.json")
    parser.add_argument("--batches", default="1,8,16,24,32")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--slo-ms", type=float, default=45)
    parser.add_argument("--port", type=int, default=18640)
    parser.add_argument("--kv-mib", type=int, default=12288)
    parser.add_argument("--gpu-cache-mib", type=int, default=6144)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=14400)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--wait-for-model", action="store_true")
    args = parser.parse_args()
    root = Path(args.result_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]
    assert min(batches) > 0 and max(batches) <= args.max_batch
    if args.wait_for_model:
        model_path = Path("/home/admin/model/DeepSeek-V4-Pro")
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            index_path = model_path / "model.safetensors.index.json"
            if index_path.exists():
                index = json.loads(index_path.read_text())
                missing = [
                    name
                    for name in set(index["weight_map"].values())
                    if not (model_path / name).exists()
                ]
                if not missing and (model_path / "tokenizer.json").exists():
                    break
                print(f"waiting_for_model missing_shards={len(missing)}", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("model weight download")
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/admin/model/DeepSeek-V4-Pro", local_files_only=True
    )
    source = json.loads(Path(args.corpus).read_text())
    samples = []
    for row in source["samples"][: max(batches)]:
        ids = row["input_ids"][: args.context]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
        encoded = tokenizer.encode(prompt)
        samples.append(
            {
                "prompt": prompt,
                "input_tokens": len(encoded),
                "source_file": row["source_file"],
                "source_row": row["source_row"],
                "original_input_tokens": row["original_input_tokens"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    assert len(samples) == max(batches)
    (root / "corpus-manifest.json").write_text(
        json.dumps(
            [{k: v for k, v in row.items() if k != "prompt"} for row in samples],
            indent=2,
        )
    )
    base = f"http://127.0.0.1:{args.port}"
    control = requests.Session()
    control.trust_env = False
    env = os.environ.copy()
    env.update(
        DSV4_RUN_ROOT=str(root / "server"),
        DSV4_PORT=str(args.port),
        DSV4_MAX_BATCH=str(args.max_batch),
        DSV4_CAPTURE=args.batches,
        DSV4_MAX_SEQ=str(
            max(s["input_tokens"] for s in samples) + args.tokens + args.warmup + 1024
        ),
        DSV4_KV_MIB=str(args.kv_mib),
        DSV4_CSA_GPU_CACHE_MIB=str(args.gpu_cache_mib),
    )
    repo = Path(__file__).resolve().parents[2]
    metadata = {
        "args": vars(args),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "tp": 4,
        "dp": 1,
        "ep": 4,
        "cp": 1,
        "cuda_graph": True,
        "preferred_numa_node": int(env.get("DSV4_NUMA_NODE", "1")),
        "fetch_ctas": int(env.get("DSV4_CSA_FETCH_CTAS", "256")),
        "real_prefill": True,
        "csa_byte_validation": env.get("DSV4_CSA_VALIDATE_BYTES", "0") == "1",
        "sequential_prefill_fixed_cohort": True,
        "decode_goodput_definition": "successful measured output tokens from requests with TPOT <= SLO / decode wall time",
        "slo_failure_is_request_failure": False,
        "cpu_full_csa_copy": args.scheme == "offload",
        "hot_entries_per_request_per_csa_layer": (
            2048 if args.scheme == "offload" else 0
        ),
        "cpu_logical_blocks": int(env.get("DSV4_CSA_LOGICAL_BLOCKS", "65537")),
        "source_diff": subprocess.check_output(
            ["git", "diff", "--stat"], cwd=repo, text=True
        ),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]

    def stream(sample, row):
        client = requests.Session()
        client.trust_env = False
        count = args.tokens + args.warmup + 1
        payload = {
            "prompt": sample["prompt"],
            "yield_generator": True,
            "generate_config": {
                "min_new_tokens": count,
                "max_new_tokens": count,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 1.0,
                "random_seed": 42,
                "return_output_ids": True,
                "return_incremental": True,
                "timeout_ms": args.request_timeout * 1000,
            },
        }
        started = time.perf_counter()
        arrivals, ids, last = [], [], {}
        error = None
        graph_probe_active = False
        instrumentation_errors = []

        def set_probe_level(level):
            try:
                control.post(
                    base + "/set_log_level", json={"log_level": level}, timeout=5
                ).raise_for_status()
                return True
            except requests.RequestException as exc:
                instrumentation_errors.append(repr(exc))
                return False

        try:
            with client.post(
                base + "/",
                json=payload,
                stream=True,
                timeout=(30, args.request_timeout),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=None):
                    line = line.strip()
                    if not line or line.startswith(b":"):
                        continue
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    if line.lower() == b"[done]":
                        break
                    last = json.loads(line)
                    if "error" in last or "error_code" in last:
                        raise RuntimeError(last)
                    output = (last.get("output_ids") or [[]])[0]
                    if output:
                        ids.extend(output)
                        assert len(ids) == last["aux_info"]["output_len"], last[
                            "aux_info"
                        ]
                        arrivals.append(
                            {"time": time.perf_counter(), "count": len(ids)}
                        )
                        if row == 0 and len(ids) == 2 and args.warmup >= 8:
                            graph_probe_active = set_probe_level("DEBUG")
                        elif graph_probe_active and len(ids) >= 5:
                            if set_probe_level("INFO"):
                                graph_probe_active = False
                    if last.get("finished"):
                        break
            assert last.get("finished") and len(ids) == count, (len(ids), count, last)
        except Exception as exc:
            error = repr(exc)
        finally:
            if graph_probe_active:
                set_probe_level("INFO")
        result = {
            "row": row,
            "started": started,
            "finished_at": time.perf_counter(),
            "arrivals": arrivals,
            "output_ids": ids,
            "aux_info": last.get("aux_info", {}),
            "error": error,
            "instrumentation_errors": instrumentation_errors,
            "error_payload": last if error else None,
            "expected_input_tokens": sample["input_tokens"],
        }
        if not error:
            first = next(a for a in arrivals if a["count"] >= args.warmup + 1)
            end = arrivals[-1]
            result.update(
                measured_tokens=end["count"] - first["count"],
                decode_start=first["time"],
                decode_end=end["time"],
            )
            result["tpot_ms"] = (
                (end["time"] - first["time"]) * 1000 / result["measured_tokens"]
            )
        return result

    server = None
    log_path = root / "startup.log"
    try:
        if not args.attach:
            try:
                control.get(base + "/health", timeout=1)
            except requests.RequestException:
                pass
            else:
                raise RuntimeError("benchmark port already occupied")
            with log_path.open("w") as log:
                server = subprocess.Popen(
                    ["bash", str(Path(__file__).with_name("start.sh")), args.scheme],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(f"server_pid={server.pid}", flush=True)
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if server and server.poll() is not None:
                raise RuntimeError(f"server exited: {log_path}")
            try:
                if control.get(base + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("server startup")
        print("server_ready", flush=True)
        for batch in batches:
            engine_log = root / "server/logs/engine.log"
            offset = engine_log.stat().st_size if engine_log.exists() else 0
            control.post(
                base + "/update_scheduler_info",
                json={"batch_size": batch, "mode": "decode"},
                timeout=30,
            ).raise_for_status()
            stop = threading.Event()
            snapshots = []

            def monitor():
                while not stop.wait(0.5):
                    snapshots.append(
                        [
                            pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
                            for h in handles
                        ]
                    )

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            started = time.perf_counter()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                    rows = list(
                        pool.map(
                            lambda pair: stream(pair[1], pair[0]),
                            enumerate(samples[:batch]),
                        )
                    )
            finally:
                stop.set()
                thread.join()
            with engine_log.open() as handle:
                handle.seek(offset)
                segment = handle.read()
            actual = [
                int(b)
                for b in re.findall(
                    r"BENCH_DECODE step=\d+ actual_batch=(\d+)", segment
                )
            ]
            successful = [r for r in rows if r["error"] is None]
            failed = batch - len(successful)
            result = {
                "scheme": args.scheme,
                "batch": batch,
                "rows": rows,
                "context_tokens": [s["input_tokens"] for s in samples[:batch]],
                "requests_attempted": batch,
                "requests_completed": len(successful),
                "requests_failed": failed,
                "request_failure_rate": failed / batch,
                "fixed_cohort_capacity_rejected": "BENCH_CAPACITY_REJECT" in segment,
                "actual_decode_batches": actual,
                "graph_replay_evidence": [
                    line
                    for line in segment.splitlines()
                    if "using CUDA graph forward" in line
                ][:8],
                "prefill_calls": segment.count("BENCH_PREFILL request="),
                "elapsed_seconds_including_prefill": time.perf_counter() - started,
                "gpu_peak_mib": [
                    max((s[i] for s in snapshots), default=0) for i in range(4)
                ],
                "slo_ms": args.slo_ms,
                "capacity_evidence": [
                    line
                    for line in segment.splitlines()
                    if "CAPACITY_REJECT" in line
                    or "LACK MEM" in line
                    or "out of memory" in line
                ][-12:],
            }
            result["status"] = "completed" if not failed else "request_failure"
            if successful:
                elapsed = max(r["decode_end"] for r in successful) - min(
                    r["decode_start"] for r in successful
                )
                good = [r for r in successful if r["tpot_ms"] <= args.slo_ms]
                result.update(
                    tpot_ms=statistics.mean(r["tpot_ms"] for r in successful),
                    decode_tokens_per_second=sum(
                        r["measured_tokens"] for r in successful
                    )
                    / elapsed,
                    decode_goodput_tokens_per_second=sum(
                        r["measured_tokens"] for r in good
                    )
                    / elapsed,
                    slo_pass_requests=len(good),
                )
                if not failed and (not actual or any(b != batch for b in actual)):
                    result["status"] = "invalid_batch_measurement"
            (root / f"b{batch}.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k
                        not in (
                            "rows",
                            "actual_decode_batches",
                            "capacity_evidence",
                            "context_tokens",
                        )
                    }
                ),
                flush=True,
            )
            if result["status"] == "invalid_batch_measurement":
                raise RuntimeError("actual decode batch did not match requested cohort")
        (root / "completed.json").write_text(
            json.dumps({"scheme": args.scheme, "batches": batches})
        )
    except Exception as error:
        (root / "failure.json").write_text(json.dumps({"error": repr(error)}))
        raise
    finally:
        if server:
            try:
                processes = psutil.Process(server.pid).children(recursive=True) + [
                    psutil.Process(server.pid)
                ]
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(processes, timeout=10)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            server.wait(timeout=30)


if __name__ == "__main__":
    main()
