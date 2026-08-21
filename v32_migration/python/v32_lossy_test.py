"""Unit test for the Tier-2 lossy attention path (mask + alias + prefetch)."""

import time

import torch
from v32_ctx_build import ctx


def test_lossy():
    dev = "cuda:0"
    torch.manual_seed(0)
    topk, hist = 2048, 8192
    nb = hist // 64  # 128 blocks
    kv_host = (
        torch.randn(hist, 576, dtype=torch.float32).to(torch.bfloat16).pin_memory()
    )
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    ctx.ctx_init()

    # engine table: block0=1, staging entries 1..32 -> phys 100..131,
    # tail 16 blocks resident (400..415), middle offloaded (0 sentinel)
    bt = torch.zeros(nb, dtype=torch.int32, device=dev)
    bt[0] = 1
    bt[1:33] = torch.arange(100, 132, dtype=torch.int32, device=dev)
    bt[nb - 16 :] = torch.arange(400, 416, dtype=torch.int32, device=dev)
    kbt = bt.unsqueeze(0).contiguous()

    jpos = torch.arange(1, 33, dtype=torch.int32)
    sb = torch.arange(100, 132, dtype=torch.int32)
    ctx.ctx_lossy_register(11, 0, jpos, sb, nb + 8, 0)
    assert ctx.ctx_lossy_has(11, 0)

    cold = torch.arange(40 * 64, 44 * 64, dtype=torch.int32)  # 4 cold blocks
    tail = torch.arange(hist - 512, hist, dtype=torch.int32)  # resident
    pad = torch.full((topk - cold.numel() - tail.numel(),), -1, dtype=torch.int32)
    sel = torch.cat([cold, tail, pad]).to(dev)
    ktr = sel.unsqueeze(0).clone()

    # step 1: cold -> dropped + miss exported
    ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 1, 8)
    torch.cuda.synchronize()
    o1 = ktr[0]
    assert torch.equal(o1[cold.numel() : cold.numel() + 512].cpu(), tail), "tail kept"
    assert int((o1[: cold.numel()] == -1).sum()) == cold.numel(), "cold dropped"
    t_, p_, m_, s_ = ctx.ctx_lossy_counters()
    print(f"step1 counters tail={t_} pool={p_} miss={m_} serves={s_}")
    assert t_ == 512 and p_ == 0 and m_ == 256 and s_ == 1

    # step 2: same sel -> misses were prefetched into slots 0..3 (jpos 1..4)
    ktr[0] = sel
    ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 2, 8)
    torch.cuda.synchronize()
    o2 = ktr[0]
    remapped = o2[: cold.numel()].cpu()
    assert int((remapped >= 0).sum()) == cold.numel(), "pool hits missing"
    t2, p2, m2, s2 = ctx.ctx_lossy_counters()
    print(f"step2 counters tail={t2} pool={p2} miss={m2} serves={s2}")
    assert p2 == 256, f"expected 256 pool hits, got {p2}"

    # verify remap + data: logical' = jpos*64+off, convert -> bt[jpos]*64+off
    for k in range(0, cold.numel(), 37):
        p_log = int(remapped[k])
        mp, off = p_log // 64, p_log % 64
        assert 1 <= mp <= 32, f"remap out of staging range: {p_log}"
        g = int(bt[mp]) * 64 + off
        got = pool[g].float().cpu()
        ref = kv_host[int(cold[k])].float()
        assert (got - ref).abs().max() < 1e-3, f"content mismatch at k={k}"
    print("remap + content PASS")

    # eviction: request more distinct blocks than slots (32) -> ring wraps
    many = torch.arange(
        36 * 64, 100 * 64, 64, dtype=torch.int32
    )  # 64 blocks, 1 tok each
    pad2 = torch.full((topk - many.numel(),), -1, dtype=torch.int32)
    sel2 = torch.cat([many, pad2]).to(dev)
    for stp in range(3, 15):
        ktr[0] = sel2
        ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, stp, 8)
        torch.cuda.synchronize()
    t3, p3, m3, s3 = ctx.ctx_lossy_counters()
    print(f"after churn: tail={t3} pool={p3} miss={m3} serves={s3}")
    assert p3 > p2, "no pool hits under churn"

    # timing: 61-layer serve, no misses (steady state)
    ktr[0] = sel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(61):
        ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 100 + i, 8)
    torch.cuda.synchronize()
    print(f"61x lossy_serve = {(time.perf_counter() - t0) * 1000:.2f}ms")

    ctx.ctx_lossy_release(11)
    assert not ctx.ctx_lossy_has(11, 0)

    # production regime: 63k history, 256 resident tail blocks, ~10% miss rate
    hist2 = 984 * 64
    nb2 = hist2 // 64
    bt2 = torch.zeros(nb2, dtype=torch.int32, device=dev)
    bt2[0] = 1
    bt2[1:33] = torch.arange(100, 132, dtype=torch.int32, device=dev)
    bt2[nb2 - 256 :] = torch.arange(500, 756, dtype=torch.int32, device=dev)
    kbt2 = bt2.unsqueeze(0).contiguous()
    kv2 = torch.empty(hist2, 576, dtype=torch.bfloat16).pin_memory()
    ctx.ctx_lossy_register(12, 0, jpos, sb, nb2 + 128, 0)

    n_tail = int(topk * 0.9)
    tail_pick = torch.randint(hist2 - 256 * 64, hist2, (n_tail,), dtype=torch.int32)
    miss_pick = torch.randint(
        33 * 64, (nb2 - 256) * 64, (topk - n_tail,), dtype=torch.int32
    )
    sel2 = torch.cat([tail_pick, miss_pick]).to(dev)
    resident2 = torch.randint(hist2 - 256 * 64, hist2, (topk,), dtype=torch.int32).to(
        dev
    )

    for tag, s_ in (("10% miss", sel2), ("all resident", resident2)):
        ktr2 = s_.unsqueeze(0).clone()
        for i in range(5):
            ktr2[0] = s_
            ctx.ctx_lossy_serve(12, 0, kv2, kbt2, ktr2, 0, pool, hist2, 500 + i, 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(61):
            ktr2[0] = s_
            ctx.ctx_lossy_serve(12, 0, kv2, kbt2, ktr2, 0, pool, hist2, 600 + i, 0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"61x lossy_serve ({tag}) = {dt:.2f}ms -> {dt/61*1000:.0f}us/layer")
    ctx.ctx_lossy_release(12)
    print("LOSSY TEST PASS")


if __name__ == "__main__":
    test_lossy()
