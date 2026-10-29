#!/usr/bin/env python3
"""Cost model: homogeneous 5kpro PD vs heterogeneous (5kpro P -> PPU D) vs homogeneous PPU PD.

DeepSeek-V4-Flash, 32K input, 1024 output, MTP 2+1 at acceptance 2.3, decode at TPOT <= 15 ms.
All capacities are measured (docs/project-progress.md §94-§97); see CAP below. Outputs, for each prefix-hit ratio:
the ideal P:D ratio, the best small-integer instance mix (<= MAX_INST instances of each), its system QPS, waste,
GPU count and QPS per GPU (and per cost unit when a card price ratio is given).

usage: pd_cost_model.py [--price-5kpro 1.0 --price-ppu 1.0] [--out-tokens 1024] [--max-inst 6]
"""
import argparse
import itertools

# Prefill instance capacity (req/s at saturation) per prefix-hit level, measured.
# hit = cached tokens / 32768. Cold = 0.
HITS = [0.0, 16384 / 32768, 24576 / 32768, 28672 / 32768, 30720 / 32768, 31744 / 32768]
P_CAP = {
    # 5kpro-4 CEP4PP2 MTP3 fastgen, 8 GPUs (§96, §97.1): cold 1.0 burst / 0.9 sustained; hits: overload first-token rate
    "5kpro_cep4pp2": {"gpus": 8, "cap": [0.92, 1.74, 3.02, 4.86, 10.07, 11.66]},
    # ppu5 PPU TP4 x EP1, no MTP, 4 GPUs (§95.2, §97.2): cc4 / cc8 completion rate
    "ppu_tp4": {"gpus": 4, "cap": [0.48, 0.93, 1.78, 3.21, 5.14, 7.04]},
    # ppu5 PPU TP8 x EP1, 8 GPUs
    "ppu_tp8": {"gpus": 8, "cap": [0.55, 1.04, 1.97, 3.51, 5.40, 5.31]},
}
# Decode instance capacity: output tok/s at TPOT <= 15 ms, MTP 2+1 forced acceptance 2.3.
D_CAP = {
    # ppu4 DEP8 + DSV4_PPU_DECODE_FUSIONS=1 (§94, §97.3)
    "ppu_dep8": {"gpus": 8, "tok_s": 4600.0},
    # 5kpro-4 TP4 PDFUSION decode-only (§95.1): 15 ms line at cc8
    "5kpro_tp4": {"gpus": 4, "tok_s": 541.0},
}
SYSTEMS = {
    "5kpro PD (同构)": ("5kpro_cep4pp2", "5kpro_tp4"),
    "异构 PD (5kpro P -> PPU D)": ("5kpro_cep4pp2", "ppu_dep8"),
    "PPU PD (同构, P=TP4)": ("ppu_tp4", "ppu_dep8"),
    "PPU PD (同构, P=TP8)": ("ppu_tp8", "ppu_dep8"),
}
HW = {
    "5kpro_cep4pp2": "5kpro",
    "5kpro_tp4": "5kpro",
    "ppu_tp4": "ppu",
    "ppu_tp8": "ppu",
    "ppu_dep8": "ppu",
}


def ideal_per_gpu(pc, p_gpus, d_rps, d_gpus):
    """QPS per GPU with a fractional, perfectly balanced P:D ratio (no waste)."""
    return 1.0 / (p_gpus / pc + d_gpus / d_rps)


def best_mix(pc, p_gpus, d_rps, d_gpus, price_p, price_d, max_inst):
    """Smallest integer mix (nP, nD) whose QPS per cost is within 5% of the best mix with <= max_inst of each.
    Prefers few instances: a 2:1 that wastes 4% beats a 5:6 that wastes 1%."""
    cands = []
    for nP, nD in itertools.product(range(1, max_inst + 1), range(1, max_inst + 1)):
        qps = min(nP * pc, nD * d_rps)
        gpus = nP * p_gpus + nD * d_gpus
        cost = nP * p_gpus * price_p + nD * d_gpus * price_d
        waste = 1 - qps / max(nP * pc, nD * d_rps)
        cands.append((qps / cost, nP + nD, nP, nD, qps, waste, gpus, cost))
    top = max(c[0] for c in cands)
    ok = [c for c in cands if c[0] >= 0.95 * top]
    c = min(ok, key=lambda c: (c[1], -c[0]))
    return c[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--max-inst", type=int, default=6)
    ap.add_argument(
        "--price-5kpro", type=float, default=1.0, help="relative price of one 5kpro GPU"
    )
    ap.add_argument(
        "--price-ppu", type=float, default=1.0, help="relative price of one PPU GPU"
    )
    a = ap.parse_args()
    price = {"5kpro": a.price_5kpro, "ppu": a.price_ppu}

    print(
        "# 32K input, %d output, MTP 2+1 @ 2.3, TPOT <= 15ms; prices: 5kpro %.2f, PPU %.2f per GPU\n"
        % (a.out_tokens, a.price_5kpro, a.price_ppu)
    )
    for name, (pk, dk) in SYSTEMS.items():
        P, D = P_CAP[pk], D_CAP[dk]
        d_rps = D["tok_s"] / a.out_tokens
        print(
            "## %s: P = %s (%d GPU), D = %s (%d GPU, %.2f req/s)"
            % (name, pk, P["gpus"], dk, D["gpus"], d_rps)
        )
        print(
            "| 命中 | P 实例 req/s | 理想 P:D | 整数配比 P:D | 系统 QPS | 瓶颈 | 闲置 | 卡数 | QPS/卡 | 理想 QPS/卡 | QPS/成本 |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for h, pc in zip(HITS, P["cap"]):
            ideal = d_rps / pc
            nP, nD, qps, waste, gpus, cost = best_mix(
                pc,
                P["gpus"],
                d_rps,
                D["gpus"],
                price[HW[pk]],
                price[HW[dk]],
                a.max_inst,
            )
            bott = "P" if nP * pc < nD * d_rps else "D"
            print(
                "| %.0f%% | %.2f | %.2f : 1 | %d : %d | %.2f | %s | %.0f%% | %d | %.3f | %.3f | %.3f |"
                % (
                    100 * h,
                    pc,
                    ideal,
                    nP,
                    nD,
                    qps,
                    bott,
                    100 * waste,
                    gpus,
                    qps / gpus,
                    ideal_per_gpu(pc, P["gpus"], d_rps, D["gpus"]),
                    qps / cost,
                )
            )
        print()


if __name__ == "__main__":
    main()
