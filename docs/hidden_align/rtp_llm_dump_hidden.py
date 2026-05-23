#!/usr/bin/env python3
"""RTP-LLM hidden-state dump for GLM-5-BF16-4layer.

Mirrors the FP8_PER_BLOCK_NO_MOE + mega_moe quant/strategy of the
``mla_mega_moe_fp8_attn_cp_pd`` smoke, but uses TP=1/EP=1/DP=1 so the
captured hidden state matches the single-rank vLLM dump 1:1.

Per prompt:
  1. Sets MOEDBG_CASE=<prompt_name> and starts a server
  2. Sends a single greedy request via HTTP
  3. Waits for the layer-by-layer dump file under MOEDBG_DIR
  4. Kills the server and moves the dump into vllm_dumps' sibling dir
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import psutil
import requests
import torch

MODEL_PATH = "/home/zw193905/models/GLM-5-BF16-4layer"
WORK_DIR = "/home/zw193905/RTP-LLM/github-opensource"
DUMP_BASE = "/tmp/rtp_llm_hidden_dumps"
OUT_DIR = "/home/zw193905/RTP-LLM/github-opensource/docs/hidden_align/rtp_llm_dumps"
PORT = 18235
GPU = "1"

SMOKE_ARGS = [
    "--warm_up",
    "0",
    "--seq_size_per_block",
    "64",
    "--act_type",
    "BF16",
    "--enable_cuda_graph",
    "0",
    "--tp_size",
    "1",
    "--ep_size",
    "1",
    "--dp_size",
    "1",
    "--world_size",
    "1",
    "--quantization",
    "FP8_PER_BLOCK_NO_MOE",
    "--moe_strategy",
    "mega_moe",
    "--reserver_runtime_mem_mb",
    "8192",
    "--force_cpu_load_weights",
    "1",
    "--fp8_kv_cache",
    "1",
    "--use_deepep_moe",
    "0",
    "--use_deepep_low_latency",
    "0",
    "--use_all_gather",
    "0",
]


def build_long_prompt(target_tokens=4096):
    base = (
        "Please provide a comprehensive analysis of the following topics. "
        "For each topic, discuss the historical background, current state, "
        "and future implications. Be thorough and detailed in your analysis.\n\n"
    )
    topics = [
        "Topic 1: The evolution of artificial intelligence from rule-based systems "
        "to modern deep learning architectures. Discuss key milestones including "
        "expert systems, neural networks, convolutional networks, transformers, "
        "and large language models. Analyze the role of compute scaling, data "
        "availability, and algorithmic innovations.\n\n",
        "Topic 2: The impact of climate change on global food security. Examine "
        "how rising temperatures, changing precipitation patterns, and extreme "
        "weather events affect agricultural production. Discuss adaptation "
        "strategies including drought-resistant crops, precision agriculture, "
        "and changes in farming practices.\n\n",
        "Topic 3: The transformation of global financial systems through "
        "digital currencies and blockchain technology. Analyze the rise of "
        "Bitcoin, Ethereum, and central bank digital currencies. Discuss "
        "regulatory challenges, environmental concerns, and the potential "
        "for financial inclusion.\n\n",
        "Topic 4: The role of space exploration in advancing human knowledge "
        "and technology. Discuss missions to Mars, the development of reusable "
        "rockets, space tourism, and the search for extraterrestrial life. "
        "Analyze the economic and scientific benefits.\n\n",
    ]
    prompt = base
    while len(prompt) < target_tokens * 3:
        for topic in topics:
            prompt += topic
            if len(prompt) >= target_tokens * 3:
                break
    return prompt


PROMPTS = {
    "short": ("The capital of France is", 20),
    "medium": (
        "Write a detailed essay about the future of quantum computing and its "
        "applications in drug discovery, cryptography, and materials science.",
        100,
    ),
    "long_4k": (build_long_prompt(4096), 50),
}


def wait_server_ready(port, timeout=300):
    url = f"http://0.0.0.0:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"  Server ready after {time.time() - start:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  Server did NOT come up after {timeout}s", flush=True)
    return False


def kill_server(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = list(parent.children(recursive=True))
        for child in children:
            child.terminate()
        psutil.wait_procs(children, timeout=5)
        for child in psutil.wait_procs(children, timeout=0)[1]:
            try:
                child.kill()
            except Exception:
                pass
        parent.terminate()
        try:
            parent.wait(timeout=10)
        except psutil.TimeoutExpired:
            parent.kill()
    except Exception as e:
        print(f"  Cleanup warning: {e}", flush=True)
        proc.terminate()


def query_server(port, prompt, max_tokens):
    url = f"http://0.0.0.0:{port}/"
    payload = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": max_tokens,
            "top_k": 1,
            "top_p": 0,
            "temperature": 0.0,
            "return_output_ids": True,
        },
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def run_one_prompt(name, prompt, max_tokens):
    print(f"\n=== Prompt {name!r} ===", flush=True)
    case_dir = os.path.join(DUMP_BASE, name)
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    # bazel builds generated pbs into bazel-bin; make them importable
    bazel_bin = os.path.join(WORK_DIR, "bazel-bin")
    extra_paths = [bazel_bin]
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths) + (
        ":" + existing_pp if existing_pp else ""
    )
    env["CHECKPOINT_PATH"] = MODEL_PATH
    env["MODEL_TYPE"] = "glm_5"
    env["TOKENIZER_PATH"] = MODEL_PATH
    env["START_PORT"] = str(PORT)
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["NVCC_PREPEND_FLAGS"] = (
        "-ccbin=/home/zw193905/.conda_gcc/bin/x86_64-conda-linux-gnu-g++"
    )
    env["MOE_STRATEGY"] = "mega_moe"
    env["DETERMINISTIC_GEMM"] = "1"
    env["MOEDBG"] = "1"
    env["MOEDBG_DIR"] = DUMP_BASE
    env["MOEDBG_CASE"] = name
    env["MOEDBG_FULL_THRESHOLD"] = str(
        16 * 1024 * 1024
    )  # 16M elements, enough for 2k×6k
    home = os.environ.get("HOME", os.path.expanduser("~"))
    env["DG_JIT_CACHE_DIR"] = os.path.join(home, ".deep_gemm")

    log_file = os.path.join(OUT_DIR, f"{name}.server.log")
    print(f"  Log: {log_file}", flush=True)
    cmd = ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"] + SMOKE_ARGS
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh, cwd=WORK_DIR)

    try:
        if not wait_server_ready(PORT, timeout=420):
            print(f"  FATAL: server failed to start; see {log_file}", flush=True)
            return None

        # Query
        print(f"  Querying ({len(prompt)} chars, max_tokens={max_tokens})", flush=True)
        t0 = time.time()
        resp = query_server(PORT, prompt, max_tokens)
        print(f"  Response in {time.time() - t0:.1f}s", flush=True)
        output_text = resp.get("response", "")
        output_ids = resp.get("output_ids", [[]])
        if (
            isinstance(output_ids, list)
            and output_ids
            and isinstance(output_ids[0], list)
        ):
            output_ids = output_ids[0]
        print(f"  output_text[:80]: {output_text[:80]!r}", flush=True)
        print(f"  output_token_ids[:10]: {output_ids[:10]}", flush=True)

        # Locate dump files
        # _rt writes one .pt per forward; the FIRST forward is prefill
        # (full sequence), subsequent forwards are decode (B,T=1)
        files = sorted(glob.glob(os.path.join(case_dir, "rank*_pid*_step*.pt")))
        print(f"  Dump files: {len(files)}", flush=True)
        if not files:
            return None
        prefill_file = files[0]
        dump = torch.load(prefill_file, map_location="cpu")
        print(f"  Prefill dump keys: {sorted(dump['tensors'].keys())}", flush=True)
        stats_summary = {
            k: {kk: vv for kk, vv in s.items() if kk != "md5"}
            for k, s in dump.get("stats", {}).items()
        }

        # Save consolidated dump under OUT_DIR
        out_payload = {
            "name": name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "output_text": output_text,
            "output_token_ids": output_ids,
            "tensors": dump.get("tensors", {}),
            "stats": dump.get("stats", {}),
            "extra": dump.get("extra", {}),
        }
        out_path = os.path.join(OUT_DIR, f"{name}.pt")
        torch.save(out_payload, out_path)
        print(f"  Saved {out_path}", flush=True)
        with open(os.path.join(OUT_DIR, f"{name}.stats.json"), "w") as f:
            json.dump(
                {
                    "name": name,
                    "output_text": output_text,
                    "output_token_ids": output_ids,
                    "stats": stats_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return out_payload
    finally:
        print("  Stopping server", flush=True)
        kill_server(proc)
        log_fh.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*", default=["short", "medium", "long_4k"])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DUMP_BASE, exist_ok=True)

    meta_path = os.path.join(OUT_DIR, "meta.json")
    meta = {"model": MODEL_PATH, "engine": "rtp_llm", "prompts": {}}

    for name in args.prompts:
        prompt, max_tokens = PROMPTS[name]
        result = run_one_prompt(name, prompt, max_tokens)
        if result is None:
            print(f"  {name}: FAILED", flush=True)
            continue
        meta["prompts"][name] = {
            "output_token_ids": result["output_token_ids"],
            "output_text": result["output_text"],
            "max_tokens": max_tokens,
        }
        # checkpoint meta after every prompt
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nRTP-LLM dump COMPLETE", flush=True)
    print(f"Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
