#!/usr/bin/env python3
"""GSM8K accuracy of a (PD) deployment through its OpenAI-compatible endpoint.

usage: gsm8k_eval.py --data test.jsonl --endpoint http://<P>:<port>/v1/chat/completions --out results.jsonl
       [--limit 0] [--concurrency 4] [--max-tokens 10000] [--seed 42]
--data is the GSM8K test split as JSONL ({"question": ..., "answer": "... #### <number>"}, 1319 lines).

- Sends each question to an OpenAI-compatible /v1/chat/completions endpoint.
- Asks the model to end with `#### <number>`; extracts and compares with gold.
- Appends per-question results to a JSONL file (flush per line) => resumable.
- Thread pool with small configurable concurrency; retries with backoff.
"""
import argparse
import json
import math
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?"
GOLD_RE = re.compile(r"####\s*(" + NUM_RE + r")")


def norm_number(s):
    s = s.replace("$", "").replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def extract_gold(answer_text):
    m = GOLD_RE.search(answer_text)
    if not m:
        return None
    return norm_number(m.group(1))


def extract_pred(text):
    """Prefer last `#### N`; fall back to last number in the text."""
    matches = GOLD_RE.findall(text)
    if matches:
        v = norm_number(matches[-1])
        if v is not None:
            return v, "marker"
    nums = re.findall(NUM_RE, text)
    for raw in reversed(nums):
        v = norm_number(raw)
        if v is not None:
            return v, "fallback_lastnum"
    return None, "none"


PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Please reason step by step, and put your final answer at the end of your "
    "response in the exact format \"#### <number>\" (just the number, no units, "
    "no thousands separators)."
)


def post_json(endpoint, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask_one(idx, item, args, lock, out_f):
    gold = extract_gold(item["answer"])
    payload = {
        "messages": [
            {"role": "system", "content": "You are a careful math tutor."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(question=item["question"])},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": False,
    }
    attempt = 0
    err = None
    t0 = time.time()
    resp = None
    while attempt < args.retries:
        attempt += 1
        try:
            resp = post_json(args.endpoint, payload, args.timeout)
            content = resp["choices"][0]["message"].get("content") or ""
            if content.strip():
                break
            err = "empty_content"
            resp = None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            resp = None
            time.sleep(min(2 ** attempt, 30))
    wall = time.time() - t0

    rec = {
        "idx": idx,
        "gold": gold,
        "attempts": attempt,
        "wall_s": round(wall, 3),
    }
    if resp is not None:
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {}) or {}
        aux = resp.get("aux_info", {}) or {}
        pred, method = extract_pred(content)
        correct = (
            pred is not None and gold is not None
            and math.isclose(pred, gold, rel_tol=1e-6, abs_tol=1e-6)
        )
        rec.update({
            "ok": True,
            "correct": bool(correct),
            "pred": pred,
            "extract": method,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "server_cost_ms": aux.get("cost_time"),
            "question": item["question"],
            "output": content,
        })
    else:
        rec.update({"ok": False, "error": err, "question": item["question"]})

    with lock:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        state = done_stats
        state["n"] += 1
        if rec.get("correct"):
            state["c"] += 1
        print(f"[{state['n']}/{args.total}] idx={idx} "
              f"{'OK ' if rec.get('correct') else 'bad'} "
              f"tok={rec.get('completion_tokens')} "
              f"wall={rec['wall_s']}s", flush=True)
    return rec


done_stats = {"n": 0, "c": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:14000/v1/chat/completions")
    ap.add_argument("--data", required=True, help="GSM8K test split, JSONL with question / answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    items = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]
    args.total = len(items)

    done_idx = set()
    try:
        with open(args.out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        done_idx.add(rec["idx"])
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass

    todo = [(i, it) for i, it in enumerate(items) if i not in done_idx]
    print(f"total={args.total} done={len(done_idx)} todo={len(todo)} "
          f"concurrency={args.concurrency}", flush=True)

    lock = threading.Lock()
    with open(args.out, "a") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(ask_one, i, it, args, lock, out_f) for i, it in todo]
        n_err = 0
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"WORKER-ERROR: {e}", flush=True)

    # final summary from full results file
    n = c = 0
    tok_p = tok_c = 0
    fails = []
    with open(args.out) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ok"):
                fails.append(rec)
                continue
            n += 1
            c += int(bool(rec.get("correct")))
            tok_p += rec.get("prompt_tokens") or 0
            tok_c += rec.get("completion_tokens") or 0
            if not rec.get("correct"):
                fails.append(rec)
    acc = c / n if n else 0.0
    summary = {
        "answered": n, "correct": c, "accuracy": round(acc, 4),
        "failed_or_wrong": len(fails),
        "avg_prompt_tokens": round(tok_p / n, 1) if n else 0,
        "avg_completion_tokens": round(tok_c / n, 1) if n else 0,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
