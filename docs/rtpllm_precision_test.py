#!/usr/bin/env python3
"""RTP-LLM precision test for GLM-5 4-layer model — compare with vLLM baseline."""
import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL_PATH = "/home/zw193905/models/GLM-5-BF16-4layer"
VLLM_BASELINE = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/vllm_precision_output.json"
)
OUTPUT_FILE = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/rtpllm_precision_output.json"
)
PORT = 18234
GPUS = "4,5,6,7"

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
    "4",
    "--world_size",
    "4",
    "--dp_size",
    "4",
    "--reserver_runtime_mem_mb",
    "8192",
    "--force_cpu_load_weights",
    "1",
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


def wait_server_ready(port, timeout=300):
    url = f"http://0.0.0.0:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"  Server ready after {time.time()-start:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  Server not ready after {timeout}s!", flush=True)
    return False


def query_server(port, prompt, max_tokens, endpoint="/"):
    url = f"http://0.0.0.0:{port}{endpoint}"
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
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    print("=== RTP-LLM Precision Test ===", flush=True)

    with open(VLLM_BASELINE) as f:
        vllm_data = json.load(f)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["CHECKPOINT_PATH"] = MODEL_PATH
    env["MODEL_TYPE"] = "glm_5"
    env["TOKENIZER_PATH"] = MODEL_PATH
    env["START_PORT"] = str(PORT)
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["NVCC_PREPEND_FLAGS"] = (
        "-ccbin=/home/zw193905/.conda_gcc/bin/x86_64-conda-linux-gnu-g++"
    )
    env["DETERMINISTIC_GEMM"] = "1"
    home = os.environ.get("HOME", os.path.expanduser("~"))
    env["DG_JIT_CACHE_DIR"] = os.path.join(home, ".deep_gemm")

    log_file = (
        "/home/zw193905/RTP-LLM/github-opensource/docs/rtpllm_precision_server.log"
    )
    print(f"Starting RTP-LLM server on GPUs {GPUS}...", flush=True)
    print(f"  Model: {MODEL_PATH}", flush=True)
    print(f"  Port: {PORT}", flush=True)
    print(f"  Log: {log_file}", flush=True)

    cmd = ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"] + SMOKE_ARGS

    log_fh = open(log_file, "w")
    server_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=log_fh,
        cwd="/home/zw193905/RTP-LLM/github-opensource",
    )

    try:
        if not wait_server_ready(PORT, timeout=300):
            print("FATAL: Server failed to start. Check log:", log_file, flush=True)
            server_proc.terminate()
            return

        results = {"model": MODEL_PATH, "engine": "rtp_llm", "tests": {}}

        # Test 1: Short prompt
        print("\n=== Test 1: Short prompt ===", flush=True)
        short_prompt = "The capital of France is"
        resp = query_server(PORT, short_prompt, 20)
        short_text = resp.get("response", "")
        short_ids = resp.get("output_ids", [[]])[0] if resp.get("output_ids") else []
        print(f"  Output: {short_text!r}", flush=True)
        print(f"  Token IDs: {short_ids}", flush=True)
        results["tests"]["short"] = {
            "prompt": short_prompt,
            "output_text": short_text,
            "output_token_ids": short_ids,
            "raw_response": resp,
        }

        # Test 2: Long prompt
        print("\n=== Test 2: Long prompt ===", flush=True)
        long_prompt = build_long_prompt(4096)
        resp = query_server(PORT, long_prompt, 50)
        long_text = resp.get("response", "")
        long_ids = resp.get("output_ids", [[]])[0] if resp.get("output_ids") else []
        print(f"  Output: {long_text!r}", flush=True)
        print(f"  Token IDs: {long_ids}", flush=True)
        results["tests"]["long_4k"] = {
            "prompt_prefix": long_prompt[:200] + "...",
            "output_text": long_text,
            "output_token_ids": long_ids,
        }

        # Test 3: Medium prompt
        print("\n=== Test 3: Medium prompt ===", flush=True)
        medium_prompt = "Write a detailed essay about the future of quantum computing and its applications in drug discovery, cryptography, and materials science."
        resp = query_server(PORT, medium_prompt, 100)
        medium_text = resp.get("response", "")
        medium_ids = resp.get("output_ids", [[]])[0] if resp.get("output_ids") else []
        print(f"  Output: {medium_text!r}", flush=True)
        print(f"  Token IDs: {medium_ids}", flush=True)
        results["tests"]["medium"] = {
            "prompt": medium_prompt,
            "output_text": medium_text,
            "output_token_ids": medium_ids,
        }

        # Save results
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {OUTPUT_FILE}", flush=True)

        # Compare with vLLM
        print("\n=== Precision Comparison ===", flush=True)
        comparison = {}
        for test_name in ["short", "long_4k", "medium"]:
            vllm_ids = vllm_data["tests"][test_name]["output_token_ids"]
            rtp_ids = results["tests"][test_name]["output_token_ids"]
            min_len = min(len(vllm_ids), len(rtp_ids))
            match_count = sum(1 for i in range(min_len) if vllm_ids[i] == rtp_ids[i])
            first_diff = -1
            for i in range(min_len):
                if vllm_ids[i] != rtp_ids[i]:
                    first_diff = i
                    break
            if first_diff == -1 and len(vllm_ids) == len(rtp_ids):
                first_diff = -1  # perfect match
            elif first_diff == -1:
                first_diff = min_len

            comparison[test_name] = {
                "vllm_len": len(vllm_ids),
                "rtp_len": len(rtp_ids),
                "match_count": match_count,
                "match_ratio": match_count / max(min_len, 1),
                "first_diff_pos": first_diff,
            }
            status = "MATCH" if first_diff == -1 else f"DIFF at pos {first_diff}"
            print(
                f"  {test_name}: {status} ({match_count}/{min_len} tokens match)",
                flush=True,
            )
            if first_diff >= 0 and first_diff < min_len:
                print(
                    f"    vLLM[{first_diff}]={vllm_ids[first_diff]}, RTP[{first_diff}]={rtp_ids[first_diff]}",
                    flush=True,
                )

        results["comparison"] = comparison
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nRTP-LLM precision test COMPLETE", flush=True)

    finally:
        print("Stopping server...", flush=True)
        try:
            import psutil

            parent = psutil.Process(server_proc.pid)
            children = list(parent.children(recursive=True))
            for child in children:
                child.terminate()
            psutil.wait_procs(children, timeout=5)
            for child in psutil.wait_procs(children, timeout=0)[1]:
                child.kill()
            parent.terminate()
            try:
                parent.wait(timeout=10)
            except psutil.TimeoutExpired:
                parent.kill()
        except Exception as e:
            print(f"  Cleanup warning: {e}", flush=True)
            server_proc.terminate()
        log_fh.close()


if __name__ == "__main__":
    main()
