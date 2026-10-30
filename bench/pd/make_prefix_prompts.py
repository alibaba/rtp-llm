#!/usr/bin/env python3
"""Build a shared-prefix prompt set for PD prefix-hit measurements.

  make_prefix_prompts.py --base prompts_32k_x24.json --tokenizer /path/to/DeepSeek-V4-Flash \
      --hit 31744 --count 160 --seed 1000 --out prompts_32k_hit31k1k.json

The base file is a JSON list of distinct long city reports, each containing
"本报告独立分析<city>，". Every output prompt is:

  shared prefix (reports 0, 23, 22, 21 concatenated, cut to hit-4 tokens)
  + a unique suffix "[S-<k>-r<rand>] <report 1+k%20>" cut so the prompt is total-4 tokens
  + a question that asks which city the last report is about.

The chat template adds 4 tokens, so the prefill side sees --hit cached tokens and
--total tokens in all (both should be multiples of the KV block size, 256).
The output is {"prefix_tokens", "prompts", "suffix_cities"}; clients check the
answer against suffix_cities to catch wrong-segment answers.
"""
import argparse
import json
import random
import re

from transformers import AutoTokenizer

QUESTION = (
    "\n\n请先用一句话说明上面最后一份报告（以 [S- 开头的那一段）分析的是哪座城市，"
    "然后用大约150字详细概括这座城市在交通枢纽和教育资源两方面的特点。"
)
PREFIX_REPORTS = (0, 23, 22, 21)
TEMPLATE_TOKENS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--hit", type=int, required=True, help="prefix tokens the prefill side reuses"
    )
    ap.add_argument("--total", type=int, default=32768)
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument(
        "--seed", type=int, default=1000, help="per-prompt random tag seed base"
    )
    ap.add_argument("--tag", default="S", help="suffix head letter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    reports = json.load(open(a.base))

    def ntok(s):
        return len(tok.encode(s, add_special_tokens=False))

    base_ids = []
    for i in PREFIX_REPORTS:
        base_ids += tok.encode(reports[i] + "\n\n", add_special_tokens=False)
    target = a.hit - TEMPLATE_TOKENS
    prefix = tok.decode(base_ids[:target])
    assert ntok(prefix) == target, ntok(prefix)
    qn = ntok(QUESTION)
    prompts, cities = [], []
    for k in range(a.count):
        i = 1 + k % 20
        head = "\n\n[%s-%03d-r%d] " % (
            a.tag,
            k,
            random.Random(a.seed + k).randint(10000, 99999),
        )
        suffix_ids = tok.encode(reports[i], add_special_tokens=False)
        need = a.total - TEMPLATE_TOKENS - target - qn - ntok(head)
        prompts.append(prefix + head + tok.decode(suffix_ids[:need]) + QUESTION)
        cities.append(re.search(r"本报告独立分析(\S{2,4}?)，", reports[i]).group(1))
    kept = all(
        tok.encode(p, add_special_tokens=False)[:target] == base_ids[:target]
        for p in prompts[:12]
    )
    lens = sorted(set(ntok(p) for p in prompts[:12]))
    print(
        "prefix tokens %d (incl. template), prefix kept %s, prompt lengths %s"
        % (a.hit, kept, lens)
    )
    json.dump(
        {"prefix_tokens": a.hit, "prompts": prompts, "suffix_cities": cities},
        open(a.out, "w"),
    )


if __name__ == "__main__":
    main()
