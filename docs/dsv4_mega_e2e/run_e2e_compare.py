#!/usr/bin/env python3
"""DSV4 single-card e2e: normal vs mega (CSA+HCA) greedy compare.

Starts the RTP server twice — once with the mega switches off, once with
DSV4_MEGA_CSA=1 DSV4_MEGA_HCA=1 — replays the same greedy queries against
both, and diffs the outputs token-by-token.

Environment:
    E2E_CKPT    (required) checkpoint dir, e.g. a DeepSeek-V4-Flash checkout
    E2E_GPU     CUDA device index (default 0)
    E2E_PYTHON  python of a serving-capable venv (default: this interpreter)
    E2E_OUT     output dir for logs/results (default ./e2e_out)
    E2E_JIT_CACHE  base dir for the managed JIT caches (default ~/.cache/rtp_jit)
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VENV_PY = os.environ.get("E2E_PYTHON", sys.executable)
CKPT = os.environ.get("E2E_CKPT")
if not CKPT:
    sys.exit("E2E_CKPT must point at a DSV4 checkpoint directory")
PORT = int(os.environ.get("E2E_PORT", "18901"))
GPU = os.environ.get("E2E_GPU", "0")
OUT_DIR = Path(os.environ.get("E2E_OUT", "e2e_out"))
SERVER_ARGS = [
    "--start_port",
    str(PORT),
    "--load_method",
    "scratch",
    "--max_seq_len",
    "4096",
    "--enable_cuda_graph",
    "0",
    "--act_type",
    "BF16",
    "--tp_size",
    "1",
    "--dp_size",
    "1",
    "--ep_size",
    "1",
    "--world_size",
    "1",
    "--seq_size_per_block",
    "256",
    "--kv_cache_mem_mb",
    "8192",
    "--concurrency_limit",
    "1",
    "--max_context_batch_size",
    "1",
    "--reserver_runtime_mem_mb",
    "4096",
    "--fp8_kv_cache",
    "1",
]
QUERIES = [
    {"prompt": "What is the capital of France?", "max_new_tokens": 64},
    {"prompt": "2+2=", "max_new_tokens": 64},
    # Long generation: decode crosses the 128-token compression boundary, so
    # both the CSA and HCA boundary-compressor paths run inside mega decode.
    {
        "prompt": "Write a detailed step-by-step explanation of how paged "
        "attention works in LLM inference engines.",
        "max_new_tokens": 200,
    },
]


def start_server(tag: str, extra_env: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_TYPE": "deepseek_v4",
            "CHECKPOINT_PATH": CKPT,
            "TOKENIZER_PATH": CKPT,
            "START_PORT": str(PORT),
            "CUDA_VISIBLE_DEVICES": GPU,
            "DG_JIT_CPP_STANDARD": "20",
            "LOG_PATH": str(OUT_DIR / f"{tag}_logs"),
        }
    )
    # /tmp/rtp-llm belongs to another user in this container; preset every
    # managed JIT cache env so jit_cache_manager's setdefault keeps our dirs.
    jit_base = Path(
        os.environ.get("E2E_JIT_CACHE", str(Path.home() / ".cache/rtp_jit"))
    )
    for env_name, sub in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("DG_JIT_CACHE_DIR", "deep_gemm"),
        ("TRTLLM_DG_CACHE_DIR", "trtllm_deep_gemm"),
        ("TILELANG_CACHE_DIR", "tilelang"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("TVM_FFI_CACHE_DIR", "tvm_ffi"),
        ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        target = jit_base / sub
        target.mkdir(parents=True, exist_ok=True)
        env.setdefault(env_name, str(target))
    env.update(extra_env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / f"{tag}.server.log", "w")
    print(f"[{tag}] starting server (GPU {GPU}, port {PORT}) ...", flush=True)
    return subprocess.Popen(
        [VENV_PY, "-m", "rtp_llm.start_server"] + SERVER_ARGS,
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
        cwd=str(OUT_DIR),
    )


# Full V4-Flash (156GB) loads from NAS at ~4GB/min plus first-run JIT;
# 30 minutes is not enough.
def wait_ready(proc: subprocess.Popen, timeout: int = 5400) -> bool:
    sys.path.insert(0, str(Path(VENV_PY).parents[1] / "lib/python3.10/site-packages"))
    from rtp_llm.utils.util import wait_sever_done

    return wait_sever_done(proc, PORT, timeout)


def query_all(tag: str) -> list:
    results = []
    for item in QUERIES:
        body = json.dumps(
            {
                "prompt": item["prompt"],
                "generate_config": {
                    "max_new_tokens": item["max_new_tokens"],
                    "top_k": 1,
                    "top_p": 0,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        results.append(payload)
        text = payload.get("response", payload)
        print(
            f"[{tag}] Q: {item['prompt'][:40]!r}\n[{tag}] A: " f"{str(text)[:160]!r}",
            flush=True,
        )
    return results


def stop_server(proc: subprocess.Popen) -> None:
    # SIGTERM first so rtp_llm's process manager can tear down its children
    # and free device memory; SIGKILL only as a last resort (a killed CUDA
    # process can leave driver-held memory behind in this container).
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(2)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()
    # Give the driver a moment to reclaim device memory.
    time.sleep(20)


def run(tag: str, extra_env: dict) -> list:
    proc = start_server(tag, extra_env)
    try:
        if not wait_ready(proc):
            raise RuntimeError(
                f"[{tag}] server failed to become ready; see "
                f"{OUT_DIR / (tag + '.server.log')}"
            )
        print(f"[{tag}] server ready", flush=True)
        results = query_all(tag)
        (OUT_DIR / f"{tag}.results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        return results
    finally:
        stop_server(proc)


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    runs = {}
    if only in (None, "baseline"):
        runs["baseline"] = run(
            "baseline",
            {
                "DSV4_MEGA_CSA": "0",
                "DSV4_MEGA_HCA": "0",
            },
        )
    if only in (None, "mega"):
        runs["mega"] = run(
            "mega",
            {
                "DSV4_MEGA_CSA": "1",
                "DSV4_MEGA_HCA": "1",
            },
        )
    if len(runs) < 2:
        for tag in ("baseline", "mega"):
            path = OUT_DIR / f"{tag}.results.json"
            if tag not in runs and path.exists():
                runs[tag] = json.loads(path.read_text())
    if len(runs) == 2:
        print("\n========== COMPARISON ==========", flush=True)
        mismatches = 0
        for index, (base, mega) in enumerate(zip(runs["baseline"], runs["mega"])):
            base_text = base.get("response")
            mega_text = mega.get("response")
            same = base_text == mega_text
            mismatches += not same
            print(f"query {index}: {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                print(f"  baseline: {str(base_text)[:200]!r}")
                print(f"  mega    : {str(mega_text)[:200]!r}")
        print(
            f"\n{len(runs['baseline']) - mismatches}/"
            f"{len(runs['baseline'])} queries identical"
        )


if __name__ == "__main__":
    main()
