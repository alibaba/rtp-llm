"""Per-token gap probe: streams a long-context request and prints the inter-token
time series, so one-time costs (migration, allocation) can be told apart from
steady-state per-step overhead."""

import argparse
import json
import statistics as st
import time
import urllib.request

HEAD = "请把下面的文本原样复述一遍并总结要点。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=26100)
    ap.add_argument("--ctx", type=int, default=63000)
    ap.add_argument("--out", type=int, default=128)
    a = ap.parse_args()

    n = int(a.ctx / 0.463)
    prompt = HEAD + ("深度学习模型推理性能分析。" * (n // 13 + 1))[:n]
    body = json.dumps(
        {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": a.out,
                "top_k": 1,
                "is_streaming": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{a.port}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    prev = None
    gaps = []
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            if not line.strip():
                continue
            now = time.time()
            if prev is not None:
                gaps.append((now - prev) * 1000)
            prev = now
    print(
        f"chunks={len(gaps) + 1} ttft={(0 if not gaps else 0):.0f} wall={time.time() - t0:.1f}s"
    )
    if not gaps:
        print("no streaming chunks — server may not support is_streaming")
        return
    print("first 24 gaps(ms):", " ".join(f"{g:.0f}" for g in gaps[:24]))
    print("last 12 gaps(ms) :", " ".join(f"{g:.0f}" for g in gaps[-12:]))
    print(
        f"median={st.median(gaps):.1f} mean={st.mean(gaps):.1f} max={max(gaps):.0f} "
        f"sum={sum(gaps) / 1000:.1f}s"
    )
    print("top8:", " ".join(f"{g:.0f}" for g in sorted(gaps, reverse=True)[:8]))


if __name__ == "__main__":
    main()
