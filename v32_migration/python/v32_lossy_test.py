"""Unit tests for the Tier-2 lossy path and the scheme C lossless path.

Covers three things the A/B/C comparison depends on being correct:
  * the lossy hot pool now publishes a prefetched alias one step after the copy
    is issued, so the compute stream never waits on the copy stream;
  * an alias is never published before its data has actually landed, which is
    what makes the removal of the cross-stream wait safe rather than racy;
  * scheme C reproduces the baseline's selections exactly, and degrades to lossy
    (rather than reading garbage) only when the scratch is genuinely too small.
"""

import time

import torch
from v32_ctx_build import ctx


def _counters():
    """tail, pool-or-fetch, miss, serves, admit-requested, admit-skipped."""
    return tuple(ctx.ctx_lossy_counters())


def _delta(before):
    now = _counters()
    return tuple(a - b for a, b in zip(now, before))


def _build_table(nb, dev, n_tail, tail_phys0=400):
    """block0 resident, staging at table pos 1..32 (phys 100..131), tail resident."""
    bt = torch.zeros(nb, dtype=torch.int32, device=dev)
    bt[0] = 1
    bt[1:33] = torch.arange(100, 132, dtype=torch.int32, device=dev)
    bt[nb - n_tail :] = torch.arange(
        tail_phys0, tail_phys0 + n_tail, dtype=torch.int32, device=dev
    )
    return bt


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

    bt = _build_table(nb, dev, 16)
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

    # step 1: nothing has been exported yet, so cold is dropped and reported
    b0 = _counters()
    ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 1, 8)
    torch.cuda.synchronize()
    o1 = ktr[0]
    assert torch.equal(o1[cold.numel() : cold.numel() + 512].cpu(), tail), "tail kept"
    assert int((o1[: cold.numel()] == -1).sum()) == cold.numel(), "cold dropped"
    d = _delta(b0)
    print(f"step1 delta tail={d[0]} pool={d[1]} miss={d[2]} serves={d[3]}")
    assert d[:4] == (512, 0, 256, 1), d

    # step 2: the miss batch is consumed and the copy is issued, but the alias is
    # deliberately NOT published yet - publishing it here would require the
    # compute stream to wait for the copy, which is exactly what we removed.
    b2 = _counters()
    ktr[0] = sel
    ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 2, 8)
    torch.cuda.synchronize()
    d2 = _delta(b2)
    print(f"step2 delta tail={d2[0]} pool={d2[1]} miss={d2[2]} (copy in flight)")
    assert d2[1] == 0, "alias published before its data landed"
    assert int((ktr[0][: cold.numel()] == -1).sum()) == cold.numel()

    # step 3: the copy has landed, the alias is published, the hot pool hits
    b3 = _counters()
    ktr[0] = sel
    ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, 3, 8)
    torch.cuda.synchronize()
    o3 = ktr[0]
    d3 = _delta(b3)
    print(f"step3 delta tail={d3[0]} pool={d3[1]} miss={d3[2]}")
    assert d3[1] == 256, f"expected 256 pool hits one step after the copy, got {d3[1]}"

    # remap must resolve, through the UNCHANGED engine table, to the fetched data
    remapped = o3[: cold.numel()].cpu()
    bt_cpu = bt.cpu()
    for k in range(0, cold.numel(), 37):
        p_log = int(remapped[k])
        mp, off = p_log // 64, p_log % 64
        assert 1 <= mp <= 32, f"remap out of staging range: {p_log}"
        g = int(bt_cpu[mp]) * 64 + off
        got = pool[g].float().cpu()
        ref = kv_host[int(cold[k])].float()
        assert (got - ref).abs().max() < 1e-3, f"content mismatch at k={k}"
    print("remap + content PASS")

    # Eviction: ask for more distinct blocks than slots so the ring wraps. This is
    # the case where a stale alias would bite - a selection remapped to a slot
    # whose data has since been replaced - so content is verified every step, not
    # just hit counts.
    many = torch.arange(36 * 64, 100 * 64, 64, dtype=torch.int32)  # 64 blocks
    pad2 = torch.full((topk - many.numel(),), -1, dtype=torch.int32)
    sel2 = torch.cat([many, pad2]).to(dev)
    bch = _counters()
    checked = 0
    for stp in range(4, 20):
        ktr[0] = sel2
        ctx.ctx_lossy_serve(11, 0, kv_host, kbt, ktr, 0, pool, hist, stp, 8)
        torch.cuda.synchronize()
        out = ktr[0].cpu()
        src, dst = [], []
        for i in range(many.numel()):
            kl = int(out[i])
            if kl < 0:
                continue  # dropped this step; nothing to verify
            mp, off = kl // 64, kl % 64
            assert 1 <= mp <= 32, f"remap {kl} outside staging range at step {stp}"
            src.append(int(many[i]))
            dst.append(int(bt_cpu[mp]) * 64 + off)
        if src:
            got = pool[torch.tensor(dst, device=dev)].float().cpu()
            ref = kv_host[torch.tensor(src)].float()
            err = (got - ref).abs().max()
            assert err < 1e-3, f"stale alias at step {stp}: max err {err}"
            checked += len(src)
    dch = _delta(bch)
    print(f"churn delta tail={dch[0]} pool={dch[1]} miss={dch[2]} verified={checked}")
    assert dch[1] > 0, "no pool hits under churn"
    assert checked > 0, "churn verified nothing"

    ctx.ctx_lossy_release(11)
    assert not ctx.ctx_lossy_has(11, 0)


def _production_setup(dev, req_key, n_stg=32):
    """63k history, 256 resident tail blocks, ~10% of a top-2048 row missing."""
    topk = 2048
    hist = 984 * 64
    nb = hist // 64
    bt = _build_table(nb, dev, 256, tail_phys0=500)
    kbt = bt.unsqueeze(0).contiguous()
    kv = torch.randn(hist, 576, dtype=torch.float32).to(torch.bfloat16).pin_memory()
    jpos = torch.arange(1, 1 + n_stg, dtype=torch.int32)
    sb = torch.arange(100, 100 + n_stg, dtype=torch.int32)
    ctx.ctx_lossy_register(req_key, 0, jpos, sb, nb + 128, 0)

    n_tail = int(topk * 0.9)
    tail_pick = torch.randint(hist - 256 * 64, hist, (n_tail,), dtype=torch.int32)
    miss_pick = torch.randint(
        33 * 64, (nb - 256) * 64, (topk - n_tail,), dtype=torch.int32
    )
    sel = torch.cat([tail_pick, miss_pick]).to(dev)
    return topk, hist, nb, bt, kbt, kv, sel


def test_lossy_timing():
    dev = "cuda:0"
    torch.manual_seed(1)
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    topk, hist, nb, bt, kbt, kv, sel = _production_setup(dev, 12)
    resident = torch.randint(hist - 256 * 64, hist, (topk,), dtype=torch.int32).to(dev)

    for tag, s_ in (("10% miss", sel), ("all resident", resident)):
        ktr = s_.unsqueeze(0).clone()
        for i in range(5):
            ktr[0] = s_
            ctx.ctx_lossy_serve(12, 0, kv, kbt, ktr, 0, pool, hist, 500 + i, 8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(61):
            ktr[0] = s_
            ctx.ctx_lossy_serve(12, 0, kv, kbt, ktr, 0, pool, hist, 600 + i, 8)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"61x lossy_serve ({tag}) = {dt:.2f}ms -> {dt / 61 * 1000:.0f}us/layer")
    ctx.ctx_lossy_release(12)
    print("LOSSY TEST PASS")


def test_lossless():
    """Scheme C: every non-resident selection must come back exact."""
    dev = "cuda:0"
    torch.manual_seed(2)
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    topk, hist, nb, bt, kbt, kv, sel = _production_setup(dev, 21)

    ktr = sel.unsqueeze(0).clone()
    b0 = _counters()
    ctx.ctx_lossless_serve(21, 0, kv, kbt, ktr, 0, pool, hist, 0)
    torch.cuda.synchronize()
    d = _delta(b0)
    print(f"lossless delta tail={d[0]} fetched={d[1]} overflow={d[2]} serves={d[3]}")
    assert d[2] == 0, f"scratch overflowed ({d[2]}) - not lossless"
    assert d[0] + d[1] == topk, f"accounted {d[0] + d[1]} of {topk} selections"

    # every fetched selection must resolve, through the engine table, to its own
    # host-mirror row - that identity is the definition of lossless here
    sel_cpu, ktr_cpu, bt_cpu = sel.cpu(), ktr[0].cpu(), bt.cpu()
    src, dst = [], []
    for i in range(topk):
        p = int(sel_cpu[i])
        if p < 0:
            continue
        if int(bt_cpu[p // 64]) > 0:
            assert int(ktr_cpu[i]) == p, f"resident selection {i} was rewritten"
            continue
        kl = int(ktr_cpu[i])
        mp, off = kl // 64, kl % 64
        assert 1 <= mp <= 32, f"remap {kl} outside staging range"
        src.append(p)
        dst.append(int(bt_cpu[mp]) * 64 + off)
    assert len(src) == d[1], f"{len(src)} remapped vs {d[1]} counted"
    got = pool[torch.tensor(dst, device=dev)].float().cpu()
    ref = kv[torch.tensor(src)].float()
    err = (got - ref).abs().max()
    print(f"lossless content: {len(src)} tokens fetched, max err={err:.2e}")
    assert err < 1e-3, "fetched token content mismatch"

    t0 = time.perf_counter()
    for i in range(61):
        ktr[0] = sel
        ctx.ctx_lossless_serve(21, 0, kv, kbt, ktr, 0, pool, hist, 0)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"61x lossless_serve = {dt:.2f}ms -> {dt / 61 * 1000:.0f}us/layer")
    ctx.ctx_lossy_release(21)


def test_lossless_overflow():
    """Too few scratch slots must degrade to lossy, never read stale data."""
    dev = "cuda:0"
    torch.manual_seed(3)
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    n_stg = 8  # 8 * 64 = 512 scratch slots for ~205 misses -> still enough
    topk, hist, nb, bt, kbt, kv, sel = _production_setup(dev, 22, n_stg=n_stg)

    # force every selection to miss so the 512-slot scratch cannot cover 2048
    all_miss = torch.randint(
        (n_stg + 1) * 64, (nb - 256) * 64, (topk,), dtype=torch.int32
    ).to(dev)
    ktr = all_miss.unsqueeze(0).clone()
    b0 = _counters()
    ctx.ctx_lossless_serve(22, 0, kv, kbt, ktr, 0, pool, hist, 0)
    torch.cuda.synchronize()
    d = _delta(b0)
    print(f"overflow delta tail={d[0]} fetched={d[1]} overflow={d[2]}")
    assert d[1] == n_stg * 64, f"expected {n_stg * 64} fetched, got {d[1]}"
    assert d[0] + d[1] + d[2] == topk, f"accounting lost selections: {d}"
    assert d[2] > 0, "expected the scratch to overflow in this configuration"
    kt = ktr[0].cpu()
    assert int((kt == -1).sum()) == d[2], "overflowed selections must be dropped"
    ctx.ctx_lossy_release(22)
    print("LOSSLESS TEST PASS")


def test_staging_not_resident():
    """Selections inside the staging range must never count as resident.

    The engine keeps table positions 1..S pointing at real physical blocks while
    handing their contents to python, so testing residency on bt[j] > 0 alone let
    a selection in logical blocks 1..S through untouched and attention then read
    a different logical block's data. The lossy path must drop or remap it; the
    lossless path must fetch it and return the right bytes.
    """
    dev = "cuda:0"
    torch.manual_seed(5)
    topk, hist = 2048, 8192
    nb = hist // 64
    kv_host = (
        torch.randn(hist, 576, dtype=torch.float32).to(torch.bfloat16).pin_memory()
    )
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    ctx.ctx_init()
    bt = _build_table(nb, dev, 16)
    kbt = bt.unsqueeze(0).contiguous()
    bt_cpu = bt.cpu()
    jpos = torch.arange(1, 33, dtype=torch.int32)
    sb = torch.arange(100, 132, dtype=torch.int32)

    # pick tokens inside logical blocks 1..32, i.e. the staging range
    stg_sel = torch.arange(1 * 64, 5 * 64, dtype=torch.int32)
    pad = torch.full((topk - stg_sel.numel(),), -1, dtype=torch.int32)
    sel = torch.cat([stg_sel, pad]).to(dev)

    ctx.ctx_lossy_register(41, 0, jpos, sb, nb + 8, 0)
    ktr = sel.unsqueeze(0).clone()
    b0 = _counters()
    ctx.ctx_lossy_serve(41, 0, kv_host, kbt, ktr, 0, pool, hist, 1, 0)
    torch.cuda.synchronize()
    d = _delta(b0)
    print(f"lossy staging delta tail={d[0]} pool={d[1]} miss={d[2]}")
    assert d[0] == 0, f"{d[0]} staging selections wrongly counted resident"
    assert d[2] == stg_sel.numel(), "staging selections should be misses"
    assert int((ktr[0][: stg_sel.numel()] == -1).sum()) == stg_sel.numel()
    ctx.ctx_lossy_release(41)

    ctx.ctx_lossy_register(42, 0, jpos, sb, nb + 8, 0)
    ktr = sel.unsqueeze(0).clone()
    b0 = _counters()
    ctx.ctx_lossless_serve(42, 0, kv_host, kbt, ktr, 0, pool, hist, 0)
    torch.cuda.synchronize()
    d = _delta(b0)
    print(f"lossless staging delta tail={d[0]} fetched={d[1]} overflow={d[2]}")
    assert d[0] == 0, f"{d[0]} staging selections wrongly counted resident"
    assert d[1] == stg_sel.numel(), "staging selections should be fetched"
    out = ktr[0].cpu()
    src = [int(stg_sel[i]) for i in range(stg_sel.numel())]
    dst = [
        int(bt_cpu[int(out[i]) // 64]) * 64 + int(out[i]) % 64 for i in range(len(src))
    ]
    got = pool[torch.tensor(dst, device=dev)].float().cpu()
    ref = kv_host[torch.tensor(src)].float()
    err = (got - ref).abs().max()
    print(f"lossless staging content: max err={err:.2e}")
    assert err < 1e-3, "staging-range fetch returned wrong bytes"
    ctx.ctx_lossy_release(42)
    print("STAGING RESIDENCY TEST PASS")


def test_step_fast_path():
    """Step-armed fast path must match direct lossless serving exactly."""
    dev = "cuda:0"
    torch.manual_seed(6)
    pool3d = torch.zeros(1024, 64, 576, dtype=torch.bfloat16, device=dev)
    topk, hist, nb, bt, kbt, kv, sel = _production_setup(dev, 51)
    ctx.ctx_lossy_release(51)  # re-register with the retained mirror
    jpos = torch.arange(1, 33, dtype=torch.int32)
    sb = torch.arange(100, 132, dtype=torch.int32)

    ktr = sel.unsqueeze(0).clone()
    # not armed yet: must refuse, python slow path owns the layer
    assert not ctx.ctx_step_serve(900, 0, kbt, ktr, pool3d), "served without a plan"

    ctx.ctx_lossy_register(51, 0, jpos, sb, nb + 128, 0, kv)
    ctx.ctx_step_plan(900, 51, 0, hist, 0, 128)
    b0 = _counters()
    assert ctx.ctx_step_serve(900, 0, kbt, ktr, pool3d), "armed serve refused"
    torch.cuda.synchronize()
    d = _delta(b0)
    print(f"step-fast delta tail={d[0]} fetched={d[1]} overflow={d[2]}")
    assert d[2] == 0, "step fast path overflowed scratch"

    # identical lossless identity check: every remap resolves to its host row
    sel_cpu, ktr_cpu, bt_cpu = sel.cpu(), ktr[0].cpu(), bt.cpu()
    src, dst = [], []
    for i in range(topk):
        p = int(sel_cpu[i])
        if p < 0:
            continue
        if int(bt_cpu[p // 64]) > 0:
            assert int(ktr_cpu[i]) == p, f"resident selection {i} was rewritten"
            continue
        kl = int(ktr_cpu[i])
        mp, off = kl // 64, kl % 64
        assert 1 <= mp <= 32, f"remap {kl} outside staging range"
        src.append(p)
        dst.append(int(bt_cpu[mp]) * 64 + off)
    assert len(src) == d[1], f"{len(src)} remapped vs {d[1]} counted"
    flat = pool3d.reshape(-1, 576)
    got = flat[torch.tensor(dst, device=dev)].float().cpu()
    ref = kv[torch.tensor(src)].float()
    err = (got - ref).abs().max()
    print(f"step-fast content: {len(src)} tokens fetched, max err={err:.2e}")
    assert err < 1e-3, "step fast path fetched wrong bytes"

    # stale step must refuse
    ktr[0] = sel
    assert not ctx.ctx_step_serve(899, 0, kbt, ktr, pool3d), "stale step served"

    # release must disarm so a recycled key cannot reach freed mirrors
    ctx.ctx_lossy_release(51)
    ktr[0] = sel
    assert not ctx.ctx_step_serve(900, 0, kbt, ktr, pool3d), "served after release"

    # timing: re-register and run the per-layer call shape used in production
    ctx.ctx_lossy_register(51, 0, jpos, sb, nb + 128, 0, kv)
    ctx.ctx_step_plan(901, 51, 0, hist, 0, 128)
    for _ in range(5):
        ktr[0] = sel
        ctx.ctx_step_serve(901, 0, kbt, ktr, pool3d)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(61):
        ktr[0] = sel
        ctx.ctx_step_serve(901, 0, kbt, ktr, pool3d)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"61x step_serve = {dt:.2f}ms -> {dt / 61 * 1000:.0f}us/layer")
    ctx.ctx_lossy_release(51)
    print("STEP FAST PATH TEST PASS")


if __name__ == "__main__":
    test_lossy()
    test_lossy_timing()
    test_lossless()
    test_lossless_overflow()
    test_staging_not_resident()
    test_step_fast_path()
