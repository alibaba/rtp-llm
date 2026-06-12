#!/usr/bin/env python3
"""Standalone RTP-LLM Qwen3.5 no-MTP debug run.

This is intentionally not a replacement for the PD smoke. It is a narrow
precision isolation tool: if standalone output is sane while PD output is not,
the next suspect is PD cache transfer / heterogenous prefill-decode topology.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import psutil
import requests

MODEL_PATH = "/home/zw193905/models/Qwen3.5-397B-A17B-FP8"
WORK_DIR = "/home/zw193905/RTP-LLM/github-opensource"
OUT_PATH = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/"
    "qwen35_rtp_no_mtp_standalone_debug.json"
)
LOG_PATH = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/"
    "qwen35_rtp_no_mtp_standalone_debug.server.log"
)

SMOKE_MESSAGES = {
    "capital_france": [{"role": "user", "content": "What is the capital of France?"}],
    "translate_hello": [
        {"role": "user", "content": "Translate to French: 'Hello, how are you today?'"}
    ],
}


def build_args(args):
    return [
        "--load_method",
        "scratch",
        "--force_cpu_load_weights",
        "1",
        "--load_cache_timeout_ms",
        "600000",
        "--fp8_kv_cache",
        "1",
        "--ssm_state_dtype",
        "fp32",
        "--act_type",
        "BF16",
        "--enable_fp32_lm_head",
        "0",
        "--reserver_runtime_mem_mb",
        str(args.reserve_mem_mb),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_batch_tokens_size",
        str(args.max_seq_len),
        "--max_context_batch_size",
        "1",
        "--seq_size_per_block",
        "4096",
        "--kernel_seq_size_per_block",
        "64",
        "--test_block_num",
        str(args.test_block_num),
        "--tp_size",
        str(args.tp),
        "--dp_size",
        "1",
        "--ep_size",
        str(args.ep),
        "--world_size",
        str(args.world_size),
        "--warm_up",
        "0",
        "--reuse_cache",
        "0",
        "--enable_device_cache",
        "0",
        "--concurrency_limit",
        "8",
        "--frontend_server_count",
        "1",
        "--quantization",
        "FP8_PER_BLOCK_NO_MOE",
        "--moe_strategy",
        "mega_moe",
        "--use_deepep_moe",
        "0",
        "--use_deepep_low_latency",
        "0",
        "--use_all_gather=0",
        "--enable_cuda_graph",
        "0",
        "--enable_xqa",
        "0",
    ]


def wait_server_ready(port, timeout):
    url = f"http://0.0.0.0:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"Server ready after {time.time() - start:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def query_chat(port, messages, max_tokens, timeout):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "top_k": 1,
        "top_p": 0,
        "temperature": 0,
        "debug_info": True,
    }
    r = requests.post(
        f"http://0.0.0.0:{port}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def stop_proc(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = list(parent.children(recursive=True))
        for child in children:
            child.terminate()
        psutil.wait_procs(children, timeout=10)
        for child in psutil.wait_procs(children, timeout=0)[1]:
            child.kill()
        parent.terminate()
        try:
            parent.wait(timeout=20)
        except psutil.TimeoutExpired:
            parent.kill()
    except Exception as exc:
        print(f"Cleanup warning: {exc}", flush=True)
        proc.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--log", default=LOG_PATH)
    parser.add_argument("--port", type=int, default=18239)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--test-block-num", type=int, default=20)
    parser.add_argument("--reserve-mem-mb", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--start-timeout", type=int, default=1800)
    parser.add_argument("--query-timeout", type=int, default=600)
    parser.add_argument("--moedbg", action="store_true")
    parser.add_argument("--moedbg-dir", default="/tmp/qwen35_rtp_no_mtp_moedbg")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["CHECKPOINT_PATH"] = args.model
    env["MODEL_TYPE"] = "qwen35_moe"
    env["TOKENIZER_PATH"] = args.model
    env["START_PORT"] = str(args.port)
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["NVCC_PREPEND_FLAGS"] = (
        "-ccbin=/home/zw193905/.conda_gcc/bin/x86_64-conda-linux-gnu-g++"
    )
    env["MOE_STRATEGY"] = "mega_moe"
    env["DETERMINISTIC_GEMM"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["RTP_LLM_LOG_WEIGHT_MEMORY_SUMMARY"] = "0"
    env["ENABLE_FP32_LM_HEAD"] = "0"
    env["DG_JIT_CACHE_DIR"] = os.path.join(
        os.environ.get("HOME", os.path.expanduser("~")), ".deep_gemm"
    )
    if args.moedbg:
        env["MOEDBG"] = "1"
        env["MOEDBG_DIR"] = args.moedbg_dir
        env["MOEDBG_CASE"] = "standalone"
        env["MOEDBG_FULL_THRESHOLD"] = str(16 * 1024 * 1024)
    else:
        env["MOEDBG"] = "0"

    cmd = ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"] + build_args(args)
    print("Starting RTP-LLM standalone debug server", flush=True)
    print(f"  GPUs: {args.gpus}", flush=True)
    print(f"  Log: {args.log}", flush=True)
    log_fh = open(args.log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh, cwd=WORK_DIR)

    try:
        if not wait_server_ready(args.port, args.start_timeout):
            raise RuntimeError(f"server did not become ready; see {args.log}")

        results = {
            "engine": "rtp_llm",
            "mode": "standalone_no_mtp_debug",
            "model": args.model,
            "gpus": args.gpus,
            "args": build_args(args),
            "tests": {},
        }
        for name, messages in SMOKE_MESSAGES.items():
            print(f"\n=== {name} ===", flush=True)
            resp = query_chat(args.port, messages, args.max_tokens, args.query_timeout)
            text = resp["choices"][0]["message"]["content"]
            print(f"output_text={text!r}", flush=True)
            dbg = resp.get("debug_info") or {}
            results["tests"][name] = {
                "messages": messages,
                "output_text": text,
                "debug_info": dbg,
                "raw_response": resp,
            }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {args.out}", flush=True)
    finally:
        stop_proc(proc)
        log_fh.close()


if __name__ == "__main__":
    main()
