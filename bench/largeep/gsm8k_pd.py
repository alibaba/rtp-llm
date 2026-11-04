#!/usr/bin/env python3
"""GSM8K through PD: P (TP4 prefill) -> D (EP16 decode). Each request is pinned to one D DP leader (round robin via
role_addrs), greedy, real MTP acceptance. Prompt and answer extraction follow the usual GSM8K "#### <number>" protocol (resumable JSONL).
usage: gsm8k_pd.py --p-url http://<prefill host>:<port> --d-leaders ip:port,... --data test.jsonl --out out.jsonl
       [--limit 0] [--concurrency 16] [--max-tokens 4096]"""
import argparse
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?"
GOLD_RE = re.compile(r"####\s*(" + NUM_RE + r")")
# identical to scripts/gsm8k_eval.py (comparable with the earlier 1319-question runs)
PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Please reason step by step, and put your final answer at the end of your "
    'response in the exact format "#### <number>" (just the number, no units, '
    "no thousands separators)."
)


def norm(s):
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred(text):
    for raw in reversed(GOLD_RE.findall(text)):
        v = norm(raw)
        if v is not None:
            return v, "marker"
    for raw in reversed(re.findall(NUM_RE, text)):
        v = norm(raw)
        if v is not None:
            return v, "fallback_lastnum"
    return None, "none"


def ask(idx, item, args):
    gold = norm(GOLD_RE.search(item["answer"]).group(1))
    ip, port = args.d_leaders[idx % len(args.d_leaders)].rsplit(":", 1)
    payload = {
        "model": "deepseek_v4",
        "messages": [
            {"role": "system", "content": "You are a careful math tutor."},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(question=item["question"]),
            },
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": False,
        "extra_configs": {
            "role_addrs": [
                {
                    "role": "DECODE",
                    "ip": ip,
                    "http_port": int(port),
                    "grpc_port": int(port) + 1,
                }
            ]
        },
    }
    err, content, usage, finish = None, "", {}, None
    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        try:
            req = urllib.request.Request(
                args.p_url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"].get("content") or ""
            finish = resp["choices"][0].get("finish_reason")
            usage = resp.get("usage") or {}
            if content.strip():
                err = None
                break
            err = "empty_content"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
            time.sleep(min(2**attempt, 20))
    pred, how = extract_pred(content)
    return {
        "idx": idx,
        "gold": gold,
        "pred": pred,
        "extract": how,
        "ok": err is None,
        "correct": err is None and pred is not None and abs(pred - gold) < 1e-6,
        "err": err,
        "finish_reason": finish,
        "completion_tokens": usage.get("completion_tokens"),
        "wall_s": round(time.time() - t0, 3),
        "group": idx % len(args.d_leaders),
        "output": content,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-url", required=True)
    ap.add_argument("--d-leaders", required=True)
    ap.add_argument(
        "--data", required=True, help="GSM8K test.jsonl (openai/gsm8k main test split)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()
    args.d_leaders = args.d_leaders.split(",")
    items = [json.loads(l) for l in open(args.data)]
    if args.limit:
        items = items[: args.limit]
    done = set()
    try:
        for l in open(args.out):
            r = json.loads(l)
            if r.get("ok"):
                done.add(r["idx"])
    except FileNotFoundError:
        pass
    todo = [(i, it) for i, it in enumerate(items) if i not in done]
    lock = threading.Lock()
    with open(args.out, "a") as f, ThreadPoolExecutor(args.concurrency) as ex:
        futs = [ex.submit(ask, i, it, args) for i, it in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    rows = {}
    for l in open(args.out):
        r = json.loads(l)
        if r["idx"] not in rows or r.get("ok"):
            rows[r["idx"]] = r
    ok = [r for r in rows.values() if r["ok"]]
    summ = {
        "answered": len(ok),
        "correct": sum(r["correct"] for r in ok),
        "total": len(items),
        "accuracy": round(sum(r["correct"] for r in ok) / max(len(ok), 1), 4),
        "errors": len(rows) - len(ok),
        "stop": sum(r["finish_reason"] == "stop" for r in ok),
        "avg_completion_tokens": round(
            sum(r["completion_tokens"] or 0 for r in ok) / max(len(ok), 1), 1
        ),
        "per_group_acc": {
            g: round(
                sum(r["correct"] for r in ok if r["group"] == g)
                / max(1, sum(1 for r in ok if r["group"] == g)),
                4,
            )
            for g in range(len(args.d_leaders))
        },
    }
    json.dump(summ, open(args.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
