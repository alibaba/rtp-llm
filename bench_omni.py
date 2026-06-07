"""Comprehensive benchmark for Qwen2.5-Omni-7B thinker on rtp-llm."""
import json
import time
import requests
import concurrent.futures
import statistics
import subprocess

SERVER = "http://localhost:18080"
ENDPOINT = f"{SERVER}/v1/chat/completions"


def send_request(prompt, max_tokens=128, stream=False):
    payload = {
        "model": "qwen2_5_omni_thinker",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    start = time.time()
    if stream:
        first_token_time = None
        tokens = 0
        content = ""
        resp = requests.post(ENDPOINT, json=payload, timeout=120, stream=True)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    if delta.get("content"):
                        if first_token_time is None:
                            first_token_time = time.time()
                        tokens += 1
                        content += delta["content"]
                except Exception:
                    pass
        elapsed = time.time() - start
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        return {
            "elapsed_s": elapsed,
            "first_token_ms": ttft,
            "total_cost_ms": elapsed * 1000,
            "prompt_tokens": 0,
            "completion_tokens": tokens,
            "output_text": content[:80],
        }
    else:
        resp = requests.post(ENDPOINT, json=payload, timeout=120)
        elapsed = time.time() - start
        data = resp.json()
        aux = data.get("aux_info", {})
        usage = data.get("usage", {})
        return {
            "elapsed_s": elapsed,
            "first_token_ms": aux.get("first_token_cost_time", 0),
            "total_cost_ms": aux.get("cost_time", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "output_text": data["choices"][0]["message"]["content"][:80],
        }


def run_batch(prompts, max_tokens, concurrency):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, p, max_tokens) for p in prompts]
        results = [f.result() for f in futures]
    wall_time = time.time() - start
    return results, wall_time


print("=" * 70)
print("Qwen2.5-Omni-7B (Thinker) Benchmark on rtp-llm")
print("GPU: NVIDIA A10 (23GB) x1")
print("=" * 70)

# Warmup
print("\n[Warmup]")
send_request("Hello", max_tokens=5)
send_request("Hi there", max_tokens=5)
print("  Done\n")

# --- 1. Single Request Latency ---
print("=" * 70)
print("1. SINGLE REQUEST LATENCY")
print("=" * 70)

short_prompts = [
    "What is 1+1?",
    "Name the capital of Japan.",
    "What color is the sky?",
]
medium_prompts = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "What are the main causes of climate change?",
]
long_prompts = [
    "Write a detailed essay about the history of artificial intelligence, "
    "covering its origins, key milestones, major researchers, and future "
    "prospects. Include specific dates and names.",
]

for label, prompts, max_tok in [
    ("Short prompts (< 20 tokens input)", short_prompts, 32),
    ("Medium prompts (20-50 tokens input)", medium_prompts, 128),
    ("Long generation (128+ tokens output)", long_prompts, 256),
]:
    print(f"\n  {label}:")
    results = [send_request(p, max_tok) for p in prompts]
    for r in results:
        tps = r["completion_tokens"] / (r["total_cost_ms"] / 1000) if r["total_cost_ms"] > 0 else 0
        text_preview = r["output_text"][:50]
        print(
            f"    TTFT={r['first_token_ms']:7.1f}ms  "
            f"Total={r['total_cost_ms']:8.1f}ms  "
            f"In={r['prompt_tokens']:4d}  Out={r['completion_tokens']:4d}  "
            f"TPS={tps:6.1f}  "
            f'"{text_preview}..."'
        )

    ttfts = [r["first_token_ms"] for r in results]
    totals = [r["total_cost_ms"] for r in results]
    all_tps = [
        r["completion_tokens"] / (r["total_cost_ms"] / 1000)
        for r in results
        if r["total_cost_ms"] > 0
    ]
    print(
        f"    --- Avg TTFT={statistics.mean(ttfts):.1f}ms  "
        f"Avg Total={statistics.mean(totals):.1f}ms  "
        f"Avg TPS={statistics.mean(all_tps):.1f}"
    )

# --- 2. Throughput vs Concurrency ---
print("\n" + "=" * 70)
print("2. THROUGHPUT vs CONCURRENCY")
print("=" * 70)

bench_prompt = "Explain the benefits of renewable energy in detail."
for conc in [1, 2, 4, 8, 16]:
    prompts_list = [f"{bench_prompt} (variant {i})" for i in range(conc)]
    results, wall = run_batch(prompts_list, 64, conc)
    total_out = sum(r["completion_tokens"] for r in results)
    total_in = sum(r["prompt_tokens"] for r in results)
    avg_ttft = statistics.mean([r["first_token_ms"] for r in results])
    sorted_ttfts = sorted([r["first_token_ms"] for r in results])
    p99_ttft = sorted_ttfts[int(len(results) * 0.99)]
    print(
        f"  Concurrency={conc:2d}: "
        f"wall={wall:6.2f}s  "
        f"in_tok={total_in:5d}  out_tok={total_out:4d}  "
        f"throughput={total_out / wall:6.1f} tok/s  "
        f"avg_TTFT={avg_ttft:7.1f}ms  "
        f"p99_TTFT={p99_ttft:7.1f}ms"
    )

# --- 3. Input Length Scaling ---
print("\n" + "=" * 70)
print("3. INPUT LENGTH SCALING (TTFT vs input length)")
print("=" * 70)

base = "The quick brown fox jumps over the lazy dog. "
for repeat in [1, 4, 16, 64]:
    prompt = base * repeat
    r = send_request(prompt + " Summarize.", max_tokens=32)
    print(
        f"  Input ~{r['prompt_tokens']:5d} tokens: "
        f"TTFT={r['first_token_ms']:7.1f}ms  "
        f"Total={r['total_cost_ms']:8.1f}ms  "
        f"Out={r['completion_tokens']:3d}"
    )

# --- 4. Streaming vs Non-Streaming ---
print("\n" + "=" * 70)
print("4. STREAMING vs NON-STREAMING")
print("=" * 70)

prompt = "Write a short poem about the ocean."
for mode_label, stream in [("Non-streaming", False), ("Streaming", True)]:
    r = send_request(prompt, max_tokens=128, stream=stream)
    tps = r["completion_tokens"] / (r["total_cost_ms"] / 1000) if r["total_cost_ms"] > 0 else 0
    print(
        f"  {mode_label:15s}: TTFT={r['first_token_ms']:7.1f}ms  "
        f"Total={r['total_cost_ms']:8.1f}ms  "
        f"Tokens={r['completion_tokens']:3d}  TPS={tps:.1f}"
    )

# --- 5. GPU Memory ---
print("\n" + "=" * 70)
print("5. GPU MEMORY USAGE")
print("=" * 70)

result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
)
for line in result.stdout.strip().split("\n"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 4:
        idx, used, total, util = parts
        print(f"  GPU {idx}: {used} / {total} (util: {util})")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
