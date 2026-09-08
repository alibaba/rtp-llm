#!/usr/bin/env python3
"""TPOT + output probe for the A/B/C offload comparison.

The prompt is materialised once into --prompt-file and every later run reads the
same bytes, so A, B and C are guaranteed to be measuring the identical request;
the prompt digest and the server-reported input_len are recorded with each result
and mismatches are refused rather than silently averaged.

Generation is greedy (top_k=1 plus a fixed seed), and the produced text is stored
so the lossy scheme can be diffed against the baseline token for token.

Usage: bench_ab.py [--port 26100] [--ctx 63000] [--out 128] [--reps 2] [--tag A]
"""
import argparse
import hashlib
import json
import random
import statistics as st
import time
import urllib.request

HEAD = "请把下面的文本原样复述一遍并总结要点。"
FILLER = "深度学习模型推理性能分析。"

# A prompt of one repeated phrase makes nearly every token score alike, so
# top-2048 tie-breaking is unstable: any implementation that visits candidates in
# a different order picks a different but equally valid set, which surfaces as an
# occasional flipped token and looks exactly like a correctness bug. The varied
# filler separates the scores while staying byte-reproducible via a fixed seed.
_WORDS = (
    "模型 推理 性能 分析 显存 带宽 延迟 吞吐 算子 融合 调度 批次 缓存 命中 预取 "
    "量化 精度 稀疏 注意力 索引 块表 主机 设备 拷贝 流水 并行 张量 专家 路由 负载"
).split()


def make_corpus_prompt(nchars, root):
    """Build the prompt from real source files, in sorted order so it is fixed.

    Neither filler is a good stand-in for traffic: the repeated one makes almost
    every token score alike, and the word-salad one is uniformly unpredictable.
    Real code has both confident and genuinely uncertain positions, which is what
    a determinism measurement needs.
    """
    import os

    parts, n = [HEAD], len(HEAD)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.endswith((".cc", ".h", ".cu", ".py")):
                continue
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                    t = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            parts.append(t)
            n += len(t)
            if n >= nchars:
                return "".join(parts)[:nchars]
    raise SystemExit(f"corpus {root} too small: {n} < {nchars} chars")


def make_prompt(nchars, varied=False):
    if not varied:
        return HEAD + (FILLER * (nchars // len(FILLER) + 1))[:nchars]
    rng = random.Random(20260830)
    out = [HEAD]
    n = len(HEAD)
    while n < nchars:
        k = rng.randint(4, 12)
        s = "".join(rng.choice(_WORDS) for _ in range(k)) + "。"
        out.append(s)
        n += len(s)
    return "".join(out)[:nchars]


def post(port, prompt, max_new, min_new=0, timeout=2400):
    generate_config = {
        "max_new_tokens": max_new,
        "top_k": 1,
        "random_seed": 1234,
    }
    if min_new:
        generate_config["min_new_tokens"] = min_new
    body = json.dumps(
        {
            "prompt": prompt,
            "generate_config": generate_config,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def resolve_prompt(path, ctx, port, varied=False, corpus=None):
    """Read the pinned prompt, creating it on first use by calibrating chars/token."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(), False
    except FileNotFoundError:
        pass
    build = (
        (lambda n: make_corpus_prompt(n, corpus))
        if corpus
        else (lambda n: make_prompt(n, varied))
    )
    probe, _ = post(port, build(4000), 1)
    il = (probe.get("aux_info") or {}).get("input_len") or 0
    ratio = il / 4019 if il else 0.463
    prompt = build(int(ctx / ratio))
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"pinned prompt: calib ratio={ratio:.4f} chars={len(prompt)} -> {path}")
    return prompt, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=26100)
    ap.add_argument("--ctx", type=int, default=63000)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument(
        "--force-length",
        action="store_true",
        help="set min_new_tokens=out so attribution runs cannot stop early",
    )
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discarded reps; the first request on a fresh server pays mirror catch-up",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompt-file", default="/home/admin/rtp-hol/logs/ab_prompt.txt")
    ap.add_argument(
        "--corpus",
        default=None,
        help="build the prompt from real source files under this directory",
    )
    ap.add_argument(
        "--varied",
        action="store_true",
        help="use score-separating filler; the repeated one makes top-k ties unstable",
    )
    ap.add_argument("--jsonl", default="/home/admin/rtp-hol/logs/ab_bench.jsonl")
    a = ap.parse_args()

    prompt, fresh = resolve_prompt(a.prompt_file, a.ctx, a.port, a.varied, a.corpus)
    pdigest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    print(f"prompt sha256={pdigest} chars={len(prompt)} fresh={fresh}")

    rows, il_seen = [], set()
    for r in range(a.warmup + a.reps):
        out, wall = post(a.port, prompt, a.out, min_new=a.out if a.force_length else 0)
        aux = out.get("aux_info") or {}
        text = out.get("response") or out.get("generate_texts") or ""
        if isinstance(text, list):
            text = text[0] if text else ""
        ol = aux.get("output_len") or a.out
        cost = aux.get("cost_time")
        model_prefill = aux.get("prefill_time")
        first_token = aux.get("first_token_cost_time")
        pd = aux.get("pd_latency") or {}
        decode_service_ms = (pd.get("decode_service_us") or 0) / 1000.0
        if decode_service_ms and ol > 1:
            tpot = decode_service_ms / (ol - 1)
        else:
            steady_start = first_token or model_prefill
            tpot = (
                (cost - steady_start) / (ol - 1)
                if (cost and steady_start and ol > 1)
                else None
            )

        def phase_ms(name):
            value = pd.get(name)
            return round(value / 1000.0, 2) if value is not None else None

        il_seen.add(aux.get("input_len"))
        row = {
            "warmup": r < a.warmup,
            "il": aux.get("input_len"),
            "ol": ol,
            "wall_s": round(wall, 2),
            "prefill_ms": model_prefill and round(model_prefill, 1),
            "ttft_ms": first_token and round(first_token, 1),
            "cost_ms": cost and round(cost, 1),
            "steady_tpot_ms": tpot and round(tpot, 2),
            "tpot_ms": tpot and round(tpot, 2),
            "prefill_queue_ms": phase_ms("prefill_queue_us"),
            "prefill_compute_wall_ms": phase_ms("prefill_compute_wall_us"),
            "handoff_total_ms": phase_ms("handoff_total_us"),
            "handoff_blocking_tail_ms": phase_ms("handoff_blocking_tail_us"),
            "decode_kv_load_ms": phase_ms("decode_kv_load_us"),
            "admission_prepare_ms": phase_ms("admission_prepare_us"),
            "admission_prepare_wait_ms": phase_ms("admission_prepare_wait_us"),
            "decode_normal_load_ms": phase_ms("decode_normal_load_us"),
            "decode_ring_load_ms": phase_ms("decode_ring_load_us"),
            "decode_queue_ms": phase_ms("decode_queue_us"),
            "decode_first_token_ms": phase_ms("decode_first_token_us"),
            "decode_service_ms": phase_ms("decode_service_us"),
            "transport_path": pd.get("transport_path"),
            "iter": aux.get("iter_count"),
            "out_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "out_text": text,
        }
        rows.append(row)
        print("  ", {k: v for k, v in row.items() if k != "out_text"})

    if len(il_seen) > 1:
        raise SystemExit(f"input_len drifted across reps: {il_seen} - not comparable")
    tp = [r["tpot_ms"] for r in rows if r["tpot_ms"] and not r["warmup"]]
    mean = st.mean(tp) if tp else None
    shas = {r["out_sha"] for r in rows if not r["warmup"]}
    print(
        f"TAG={a.tag} ctx={a.ctx} il={il_seen.pop()} TPOT_mean={mean} ms "
        f"(n={len(tp)}) out_sha={sorted(shas)}"
    )
    with open(a.jsonl, "a") as f:
        f.write(
            json.dumps(
                {
                    "tag": a.tag,
                    "ctx": a.ctx,
                    "prompt_sha": pdigest,
                    "tpot_mean": mean,
                    "rows": rows,
                    "ts": time.time(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
