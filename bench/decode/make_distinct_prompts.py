#!/usr/bin/env python3
"""Build fully distinct long prompts for decode-only benchmarks.

Every prompt starts with its own id/city/topic and its body is drawn from a
per-prompt shuffled vocabulary, so no two prompts share even the first KV block
and prefix reuse cannot shorten prefill or warm the attention reads.
The number of body sentences is binary-searched per prompt so that the chat
input length is ``--target`` tokens (within ``--tol``).
"""
import argparse
import json
import random

from transformers import AutoTokenizer

CITIES = ["杭州", "北京", "上海", "深圳", "成都", "西安", "南京", "苏州", "重庆", "武汉", "青岛",
          "厦门", "长沙", "昆明", "哈尔滨", "沈阳", "天津", "宁波", "无锡", "佛山", "东莞", "合肥",
          "福州", "济南", "大连", "珠海", "洛阳", "桂林", "三亚", "拉萨", "乌鲁木齐", "敦煌"]  # fmt: skip
TOPICS = ["地理格局", "历史沿革", "经济产业", "文化底蕴", "科技创新", "交通枢纽", "教育资源", "生态宜居",
          "城市规划", "人口结构", "旅游资源", "饮食文化", "体育赛事", "医疗卫生", "港口物流", "数字经济"]  # fmt: skip
WORDS = ["发展", "区域", "特色", "产业", "人口", "交通", "文化", "历史", "生态", "创新", "教育", "医疗",
         "旅游", "港口", "规划", "治理", "开放", "合作", "传统", "现代", "社区", "街道", "河流", "山脉",
         "气候", "物产", "工艺", "节庆", "建筑", "园林", "市场", "企业", "人才", "政策", "资本", "科研",
         "高校", "博物馆", "图书馆", "公园", "地铁", "机场", "铁路", "公路"]  # fmt: skip


def body_sentence(rng, city, topic):
    w = rng.sample(WORDS, 6)
    n = rng.randint(3, 97)
    return (
        f"{city}在{topic}上的第{n}项观察显示，{w[0]}与{w[1]}相互作用，带动{w[2]}和{w[3]}的变化，"
        f"并对{w[4]}、{w[5]}产生长期影响。"
    )


def build(i, n_sent):
    rng = random.Random(1000003 * i + 7)
    city = CITIES[i % len(CITIES)]
    topic = TOPICS[(i * 7) % len(TOPICS)]
    lead = f"专项研究档案 ZS-{20000 + i}（{city}·{topic}，第{i}卷）：以下为独立调查记录，请勿与其他档案混淆。"
    body = "".join(
        body_sentence(rng, city, TOPICS[rng.randrange(len(TOPICS))])
        for _ in range(n_sent)
    )
    tail = (
        f"请基于以上档案，从地理、历史、经济、文化、科技五个角度详细分析{city}在{topic}方面的独特魅力与发展潜力，"
        f"撰写一篇结构完整、内容详实的长篇分析报告，篇幅越长越好。请开始你的回答。"
    )
    return lead + body + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="model / tokenizer path")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--target", type=int, default=32700, help="input tokens")
    ap.add_argument("--tol", type=int, default=120)
    ap.add_argument(
        "--template-overhead",
        type=int,
        default=5,
        help="tokens the server chat template adds (5 for DeepSeek-V4-Flash)",
    )
    ap.add_argument(
        "--min-sentences",
        type=int,
        default=100,
        help="lower bound of the body-sentence binary search (use ~1 for prompts shorter than ~4K)",
    )
    ap.add_argument("--out", default="prompts_32k_x64_distinct.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def isl(p):
        return len(tok(p).input_ids) + args.template_overhead

    prompts, lens = [], []
    for i in range(args.count):
        lo, hi = args.min_sentences, 4000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if isl(build(i, mid)) <= args.target:
                lo = mid
            else:
                hi = mid - 1
        p = build(i, lo)
        n = isl(p)
        assert abs(n - args.target) <= args.tol, (i, n)
        prompts.append(p)
        lens.append(n)
    assert len({p[:200] for p in prompts}) == args.count
    first_blocks = {tuple(tok(p).input_ids[:256]) for p in prompts}
    print(
        f"N={args.count} ISL min={min(lens)} max={max(lens)} mean={sum(lens) / args.count:.0f}; "
        f"distinct first-256-token blocks={len(first_blocks)}"
    )
    json.dump(prompts, open(args.out, "w"), ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
