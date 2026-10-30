#!/usr/bin/env python3
"""Cost model: homogeneous 5kpro PD vs heterogeneous (5kpro P -> PPU D) vs homogeneous PPU PD.

DeepSeek-V4-Flash, 32K input, MTP 2+1 at acceptance 2.3, decode at TPOT <= 15 ms. All capacities are measured
(docs/project-progress.md §94-§97). For each prefix-hit ratio the script reports:

  1. prefill cost efficiency per GPU: 5kpro vs PPU (req/s per GPU, req/s per cost unit);
  2. the ideal (fractional) P:D ratio of each system, its QPS per GPU and QPS per cost unit, and the gain of the
     heterogeneous system over each homogeneous one in both;
  3. the best whole-machine mix of each system within the available machines (8 GPUs per machine; a 5kpro TP4
     decode instance is half a machine, a PPU TP4 prefill instance is half a machine), with its QPS, the idle
     share of the over-provisioned side and of the whole system (cost-weighted), and QPS per cost unit.

usage: pd_cost_model.py [--out-tokens 1024] [--price-5kpro 1.0] [--price-ppu 1.6] [--5kpro-hosts 4] [--ppu-hosts 5]
"""
import argparse
import itertools

HITS = [0.0, 16384 / 32768, 24576 / 32768, 28672 / 32768, 30720 / 32768, 31744 / 32768]
# Prefill instance capacity (req/s at saturation) per prefix-hit level, measured.
P_CAP = {
    # 5kpro-4 CEP4PP2 MTP3 fastgen, 8 GPUs (§96, §97.2): cold 0.9 sustained / 1.0 burst; hits: overload first-token rate
    "5kpro_cep4pp2": {
        "hw": "5kpro",
        "gpus": 8,
        "cap": [0.92, 1.74, 3.02, 4.86, 10.07, 11.66],
    },
    # ppu5 PPU TP4 x EP1, no MTP, 4 GPUs (§95.2, §97.3)
    "ppu_tp4": {"hw": "ppu", "gpus": 4, "cap": [0.48, 0.93, 1.78, 3.21, 5.14, 7.04]},
    # ppu5 PPU TP8 x EP1, 8 GPUs
    "ppu_tp8": {"hw": "ppu", "gpus": 8, "cap": [0.55, 1.04, 1.97, 3.51, 5.40, 5.31]},
}
# Decode instance capacity: output tok/s at TPOT <= 15 ms, MTP 2+1 forced acceptance 2.3.
D_CAP = {
    # ppu4 DEP8 + DSV4_PPU_DECODE_FUSIONS=1 (§97.4): cc80 15.2 ms / 5264 tok/s
    "ppu_dep8": {"hw": "ppu", "gpus": 8, "tok_s": 5264.0},
    # 5kpro-4 TP4 PDFUSION decode-only (§95.1): 15 ms line at cc8
    "5kpro_tp4": {"hw": "5kpro", "gpus": 4, "tok_s": 541.0},
}
SYSTEMS = {
    "纯 5kpro PD": ("5kpro_cep4pp2", "5kpro_tp4"),
    "异构 PD（5kpro P → PPU D）": ("5kpro_cep4pp2", "ppu_dep8"),
    "纯 PPU PD（P = TP4）": ("ppu_tp4", "ppu_dep8"),
}
HET = "异构 PD（5kpro P → PPU D）"
HOMO = ["纯 5kpro PD", "纯 PPU PD（P = TP4）"]
GPUS_PER_HOST = 8


def ideal(pc, P, d_rps, D, price):
    """Balanced fractional P:D: QPS per GPU and QPS per cost unit, and the P share of the cost."""
    p_gpu, d_gpu = P["gpus"] / pc, D["gpus"] / d_rps  # GPUs needed per 1 req/s
    p_cost, d_cost = p_gpu * price[P["hw"]], d_gpu * price[D["hw"]]
    return 1 / (p_gpu + d_gpu), 1 / (p_cost + d_cost), p_cost / (p_cost + d_cost)


def mixes(pc, P, d_rps, D, price, hosts):
    """All whole-machine mixes (nP, nD instances) that fit the host budget, best QPS per cost first."""
    out = []
    for nP, nD in itertools.product(range(1, 41), range(1, 41)):
        gp, gd = nP * P["gpus"], nD * D["gpus"]
        need = {"5kpro": 0, "ppu": 0}
        need[P["hw"]] += gp
        need[D["hw"]] += gd
        if any(need[k] % GPUS_PER_HOST for k in need):
            continue  # use whole machines
        if any(need[k] / GPUS_PER_HOST > hosts[k] for k in need):
            continue
        pq, dq = nP * pc, nD * d_rps
        qps = min(pq, dq)
        cost = gp * price[P["hw"]] + gd * price[D["hw"]]
        # idle capacity: of the over-provisioned side, and of the whole system weighted by cost
        if pq > dq:
            side, side_idle, sys_idle = (
                "P",
                1 - dq / pq,
                (1 - dq / pq) * gp * price[P["hw"]] / cost,
            )
        else:
            side, side_idle, sys_idle = (
                "D",
                1 - pq / dq,
                (1 - pq / dq) * gd * price[D["hw"]] / cost,
            )
        hosts_used = "%d×5kpro + %d×PPU" % (
            need["5kpro"] // GPUS_PER_HOST,
            need["ppu"] // GPUS_PER_HOST,
        )
        out.append(
            dict(
                nP=nP,
                nD=nD,
                qps=qps,
                cost=cost,
                gpus=gp + gd,
                side=side,
                side_idle=side_idle,
                sys_idle=sys_idle,
                qpc=qps / cost,
                hosts=hosts_used,
            )
        )
    out.sort(key=lambda m: (-m["qpc"], m["gpus"]))
    return out


def pick(cands, best_qpc):
    """Fewest machines among the mixes within 5% of the best QPS per cost."""
    ok = [m for m in cands if m["qpc"] >= 0.95 * best_qpc]
    return min(ok, key=lambda m: (m["gpus"], -m["qpc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument(
        "--price-5kpro", type=float, default=1.0, help="cost of one 5kpro GPU"
    )
    ap.add_argument("--price-ppu", type=float, default=1.6, help="cost of one PPU GPU")
    ap.add_argument("--5kpro-hosts", dest="h5", type=int, default=4)
    ap.add_argument("--ppu-hosts", dest="hp", type=int, default=5)
    a = ap.parse_args()
    price = {"5kpro": a.price_5kpro, "ppu": a.price_ppu}
    hosts = {"5kpro": a.h5, "ppu": a.hp}
    pct = lambda h: (
        "%.1f%%" % (100 * h) if h not in (0.5, 0.75) and h else "%.0f%%" % (100 * h)
    )

    print(
        "# 32K 输入，输出 %d，MTP 2+1 @ 2.3，TPOT ≤ 15ms；卡价 5kpro %.2f、PPU %.2f；机器上限 %d 台 5kpro、%d 台 PPU（每台 8 卡）\n"
        % (a.out_tokens, a.price_5kpro, a.price_ppu, a.h5, a.hp)
    )

    print("## 1. Prefill 单卡性价比（5kpro CEP4PP2 vs PPU TP4 / TP8）\n")
    print(
        "| 命中 | 5kpro req/s/卡 | PPU TP4 req/s/卡 | PPU TP8 req/s/卡 | 5kpro req/s/成本 | PPU TP4 req/s/成本 | PPU TP8 req/s/成本 | 5kpro / PPU TP4 性价比 | 5kpro / PPU TP8 性价比 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for i, h in enumerate(HITS):
        k = P_CAP["5kpro_cep4pp2"]["cap"][i] / 8
        t4 = P_CAP["ppu_tp4"]["cap"][i] / 4
        t8 = P_CAP["ppu_tp8"]["cap"][i] / 8
        kc, t4c, t8c = k / price["5kpro"], t4 / price["ppu"], t8 / price["ppu"]
        print(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f× | %.2f× |"
            % (pct(h), k, t4, t8, kc, t4c, t8c, kc / t4c, kc / t8c)
        )
    print()

    print("## 2. 理想配比（分数配比、无闲置）：每卡吞吐与每成本吞吐\n")
    print(
        "| 命中 | "
        + " | ".join("%s P:D / QPS每卡 / QPS每成本" % n for n in SYSTEMS)
        + " | 异构/纯5kpro 吞吐 | 异构/纯5kpro 成本 | 异构/纯PPU 吞吐 | 异构/纯PPU 成本 |"
    )
    print("|---|" + "---|" * (len(SYSTEMS) + 4))
    for i, h in enumerate(HITS):
        cells, vals = [], {}
        for name, (pk, dk) in SYSTEMS.items():
            P, D = P_CAP[pk], D_CAP[dk]
            d_rps = D["tok_s"] / a.out_tokens
            qpg, qpc, _ = ideal(P["cap"][i], P, d_rps, D, price)
            vals[name] = (qpg, qpc)
            cells.append("%.2f : 1 / %.3f / %.3f" % (d_rps / P["cap"][i], qpg, qpc))
        het = vals[HET]
        g = []
        for hn in HOMO:
            g += ["%.2f×" % (het[0] / vals[hn][0]), "%.2f×" % (het[1] / vals[hn][1])]
        print("| %s | %s | %s |" % (pct(h), " | ".join(cells), " | ".join(g)))
    print()

    print(
        "## 3. 整机整数配比（不超过机器上限；在每成本吞吐最优值 5% 以内取机器最少的组合）\n"
    )
    chosen = {}
    for name, (pk, dk) in SYSTEMS.items():
        P, D = P_CAP[pk], D_CAP[dk]
        d_rps = D["tok_s"] / a.out_tokens
        print(
            "### %s：P = %s（%d 卡/实例，%.2f-%.2f req/s），D = %s（%d 卡/实例，%.2f req/s）\n"
            % (name, pk, P["gpus"], P["cap"][0], P["cap"][-1], dk, D["gpus"], d_rps)
        )
        print(
            "| 命中 | 理想 P:D | 推荐 P:D（实例） | 机器 | 系统 QPS | 富余侧闲置 | 全系统闲置（按成本） | QPS/成本 | 理想 QPS/成本 | 次选 |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for i, h in enumerate(HITS):
            pc = P["cap"][i]
            cands = mixes(pc, P, d_rps, D, price, hosts)
            _, qpc_ideal, _ = ideal(pc, P, d_rps, D, price)
            if not cands:
                print(
                    "| %s | %.2f : 1 | 机器不够 | | | | | | %.3f | |"
                    % (pct(h), d_rps / pc, qpc_ideal)
                )
                continue
            m = pick(cands, cands[0]["qpc"])
            chosen[(name, i)] = m
            alt = next(
                (
                    c
                    for c in cands
                    if (c["nP"], c["nD"]) != (m["nP"], m["nD"])
                    and c["gpus"] < m["gpus"]
                ),
                None,
            )
            alt_s = (
                (
                    "%d:%d，%.2f QPS，全系统闲 %.0f%%"
                    % (alt["nP"], alt["nD"], alt["qps"], 100 * alt["sys_idle"])
                )
                if alt
                else "—"
            )
            print(
                "| %s | %.2f : 1 | %d : %d | %s | %.2f | %s 闲 %.0f%% | %.0f%% | %.3f | %.3f | %s |"
                % (
                    pct(h),
                    d_rps / pc,
                    m["nP"],
                    m["nD"],
                    m["hosts"],
                    m["qps"],
                    m["side"],
                    100 * m["side_idle"],
                    100 * m["sys_idle"],
                    m["qpc"],
                    qpc_ideal,
                    alt_s,
                )
            )
        print()

    print("## 4. 按推荐整数配比比较（QPS/成本，含闲置）\n")
    print(
        "| 命中 | 纯 5kpro QPS/成本 | 异构 QPS/成本 | 纯 PPU QPS/成本 | 异构 / 纯 5kpro | 异构 / 纯 PPU |"
    )
    print("|---|---|---|---|---|---|")
    for i, h in enumerate(HITS):
        q = [chosen.get((n, i)) for n in SYSTEMS]
        if any(x is None for x in q):
            continue
        k, het, ppu = q[0]["qpc"], q[1]["qpc"], q[2]["qpc"]
        print(
            "| %s | %.3f | %.3f | %.3f | %.2f× | %.2f× |"
            % (pct(h), k, het, ppu, het / k, het / ppu)
        )
    print()


if __name__ == "__main__":
    main()
