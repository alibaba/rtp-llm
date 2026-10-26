#!/usr/bin/env python3
"""GSM8K on top of a long shared prefix, to validate prefix-reuse numerics.

  gsm8k_prefix_eval.py --prefix-file prompts_32k_prefix24k.json --gsm8k test.jsonl --n 60 --conc 4 --out out.jsonl

Every question is sent as <24K shared prefix text> + "\n\n" + question + the
usual "#### <number>" instruction, so with reuse_cache=1 the prefill side reuses
the prefix KV and only computes the question. Reports accuracy and the
server-side prefill reuse length distribution.
"""
import argparse
import json
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?"
GOLD_RE = re.compile(r"####\s*(" + NUM_RE + r")")
TEMPLATE = (
    "{prefix}\n\n以上材料与下面的数学题无关，请忽略。\n\n{question}\n\n"
    "Please reason step by step, and put your final answer at the end of your response in the exact "
    'format "#### <number>" (just the number, no units, no thousands separators).'
)


def num(s):
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def pred(text):
    m = GOLD_RE.findall(text)
    if m and num(m[-1]) is not None:
        return num(m[-1])
    for raw in reversed(re.findall(NUM_RE, text)):
        if num(raw) is not None:
            return num(raw)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--gsm8k", required=True, help="GSM8K test.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pf = json.load(open(a.prefix_file))
    prefix = pf["prompts"][0].split("\n\n[S-")[0]
    qs = [json.loads(l) for l in open(a.gsm8k)][: a.n]
    lock = threading.Lock()
    rows = []

    def run(i):
        q = qs[i]
        body = json.dumps(
            {
                "model": "deepseek_v4",
                "messages": [
                    {
                        "role": "user",
                        "content": TEMPLATE.format(
                            prefix=prefix, question=q["question"]
                        ),
                    }
                ],
                "max_tokens": a.max_tokens,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "aux_info": True,
            }
        ).encode()
        req = urllib.request.Request(
            "http://%s/v1/chat/completions" % a.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            text = r["choices"][0]["message"]["content"]
            aux = r.get("aux_info", {})
            err = None
        except Exception as e:
            text, aux, err = "", {}, repr(e)[:200]
        gold = num(GOLD_RE.search(q["answer"]).group(1))
        p = pred(text)
        row = {
            "idx": i,
            "gold": gold,
            "pred": p,
            "correct": p is not None and abs(p - gold) < 1e-6,
            "err": err,
            "prefill_reuse": aux.get("prefill_total_reuse_len"),
            "prompt_tokens": (aux.get("input_len")),
            "text_tail": text[-160:],
        }
        with lock:
            rows.append(row)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(run, range(len(qs))))
    rows.sort(key=lambda r: r["idx"])
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["correct"] for r in rows)
    errs = sum(1 for r in rows if r["err"])
    reuse = {}
    for r in rows:
        reuse[r["prefill_reuse"]] = reuse.get(r["prefill_reuse"], 0) + 1
    print(
        "acc %d/%d = %.2f%%  errors %d  prefill_reuse_len dist %s"
        % (
            ok,
            len(rows),
            100.0 * ok / len(rows),
            errs,
            dict(sorted(reuse.items(), key=lambda kv: str(kv[0]))),
        )
    )


if __name__ == "__main__":
    main()
