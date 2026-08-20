"""Compile + unit-test v32_ctx on a GPU node."""

import os
import time

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
_d = os.path.dirname(os.path.abspath(__file__))
ctx = load(
    name="v32_ctx",
    sources=[
        os.path.join(_d, "v32_ctx.cu"),
        os.path.join(_d, "glm5port", "NoBlockCopy.cu"),
        os.path.join(_d, "glm5port", "SplitKvCacheCopy.cu"),
        os.path.join(_d, "glm5port", "sm_copy_kernel.cu"),
    ],
    extra_include_paths=[os.path.join(_d, "glm5port")],
    extra_cuda_cflags=["-O3"],
    verbose=False,
    build_directory=os.environ.get("V32_CTX_BUILD", "/home/admin/rtp-hol/v32ctx_build"),
)


def test():
    dev = "cuda:0"
    torch.manual_seed(0)
    topk, hist, S = 2048, 50000, 2048
    ctx.ctx_init()
    cap = hist + 512
    kv_host = torch.randn(cap, 576, dtype=torch.float32).to(torch.bfloat16).pin_memory()
    pool_slots = 300000
    pool = torch.zeros(pool_slots, 576, dtype=torch.bfloat16, device=dev)
    stg_blocks = torch.arange(100, 132, device=dev, dtype=torch.long)
    stg_slots = (
        stg_blocks[:, None] * 64 + torch.arange(64, device=dev)[None, :]
    ).reshape(-1)
    stg_logical = (
        torch.arange(1, 33, device=dev, dtype=torch.long)[:, None] * 64
        + torch.arange(64, device=dev)[None, :]
    ).reshape(-1)
    ctx.ctx_register(7, 0, kv_host, stg_slots, stg_logical, topk)
    assert ctx.ctx_has(7, 0)

    nb = (hist + 63) // 64
    bt = torch.zeros(nb, dtype=torch.int32, device=dev)
    bt[0] = 1
    bt[1:33] = stg_blocks.to(
        torch.int32
    )  # staging blocks at entries 1..32 (as production)
    bt[nb - 256 :] = torch.arange(400, 656, dtype=torch.int32, device=dev)
    sel = (torch.randperm(hist - 2112, device=dev)[:topk] + 2112).to(torch.int32)
    cur_slot = 999999

    # step 1: everything cold -> all misses enqueued
    g1 = ctx.ctx_serve(7, 0, sel, bt, pool.reshape(-1, 576), hist, 1, True).clone()
    torch.cuda.synchronize()
    print("dbg1 (nv,nm,staged,inbox,inflight):", ctx.ctx_debug(7, 0))
    print(
        "g1: neg1=", int((g1 == -1).sum()), "min=", int(g1.min()), "max=", int(g1.max())
    )
    time.sleep(1.5)  # let fetch thread gather
    print("dbg1b:", ctx.ctx_debug(7, 0))
    # step 2: drain happens inside; same sel -> hits from staging
    g2 = ctx.ctx_serve(7, 0, sel, bt, pool.reshape(-1, 576), hist, 2, True).clone()
    torch.cuda.synchronize()
    print("dbg2:", ctx.ctx_debug(7, 0))
    v1 = int((g1 >= 0).sum())
    v2 = int((g2 >= 0).sum())
    # reference warm count
    sl = sel.long()
    warm = int((bt[(sl // 64)] > 0).sum())
    print(f"step1 valid={v1} (warm~{warm}+cur) step2 valid={v2} (staged hits added)")
    assert v2 > v1, "staging hits did not materialize"
    # outputs are request-local logical positions; staged hits alias to entries
    # 1..32. map logical -> global slot through bt, then compare vs kv_host.
    g2v = g2[g2 >= 0].long()
    logical_set = set(stg_logical.tolist())
    staged_logical = [p for p in g2v.tolist() if p in logical_set][:64]
    assert staged_logical, "no staged logical aliases in output"
    ok = 0
    flat = pool.reshape(-1, 576)
    for p in staged_logical[:8]:
        s = int(bt[p // 64]) * 64 + p % 64  # what convert-to-global will compute
        row = flat[s].float()
        # search: compare against a sample of selected cold positions
        colds = sl[(bt[(sl // 64)] <= 0)][:4096]
        ref = kv_host[colds.cpu()].float().to(dev)
        d = (ref - row[None, :]).abs().sum(1)
        ok += int((d.min() < 1e-3))
    print(f"staged-row content matches kv_host for {ok}/8 sampled slots")
    assert ok >= 7
    # warm outputs must be pure logical positions (never global slots)
    warm_out = [p for p in g2v.tolist() if p not in logical_set and p < hist]
    assert warm_out and max(warm_out) < hist
    # timing: 61-layer serve
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(61):
        ctx.ctx_serve(7, 0, sel, bt, pool.reshape(-1, 576), hist, 3 + i, i % 4 == 0)
    torch.cuda.synchronize()
    print(f"61-layer ctx_serve cost={(time.perf_counter()-t0)*1000:.2f}ms")
    ctx.ctx_release(7)
    assert not ctx.ctx_has(7, 0)
    # micro-bench: score kernel + topk on 55k synthetic side store
    hist2 = 55000
    nb2 = (hist2 + 63) // 64
    idxp = torch.randint(0, 255, (nb2, 64, 132), dtype=torch.uint8, device=dev)
    qf = torch.randn(64, 128, device=dev).to(torch.float8_e4m3fn)
    wv = torch.rand(64, device=dev)
    kv_host2 = (
        torch.randn(hist2 + 512, 576, dtype=torch.float32)
        .to(torch.bfloat16)
        .pin_memory()
    )
    ctx.ctx_register(9, 0, kv_host2, stg_slots, stg_logical, 2048)
    bt2 = torch.ones(nb2, dtype=torch.int32, device=dev)
    qb = qf.unsqueeze(0)
    wb = wv.unsqueeze(0)
    btb = bt2.unsqueeze(0)
    ktb = torch.full((1, 2048), -1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(61):
        ctx.ctx_serve_full(
            9,
            0,
            qb,
            wb,
            idxp,
            btb,
            ktb,
            0,
            pool.reshape(-1, 576),
            hist2,
            2047,
            100 + i,
            False,
        )
    torch.cuda.synchronize()
    print(f"61x serve_full(55k) = {(time.perf_counter()-t0)*1000:.1f}ms")
    assert int((ktb >= 0).sum()) > 1500, "write-back into kernel_topk missing"
    sc = torch.randn(hist2, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(61):
        torch.topk(sc, 2047)
    torch.cuda.synchronize()
    print(f"61x topk(55k) = {(time.perf_counter()-t0)*1000:.1f}ms")
    ctx.ctx_release(9)
    # single-wave pool ops: bulk admit + batch append + tripwire
    poolL = torch.zeros(128, 64, 132, dtype=torch.uint8, device=dev)
    srcp = torch.randint(1, 255, (700, 8448), dtype=torch.uint8, device=dev)
    kbt2 = torch.zeros(2, 40, dtype=torch.int32, device=dev)
    kbt2[0, :20] = torch.arange(10, 30, dtype=torch.int32, device=dev)
    kbt2[1, :5] = torch.arange(50, 55, dtype=torch.int32, device=dev)
    ibt2 = torch.full((2, 40), -1, dtype=torch.int32, device=dev)
    ibt2[0, :20] = torch.arange(0, 20, dtype=torch.int32, device=dev)
    ibt2[1, :5] = torch.arange(20, 25, dtype=torch.int32, device=dev)
    kvl2 = torch.tensor([1230, 300], dtype=torch.int32, device=dev)
    exp2 = kbt2[:, 0].clone()
    exp2[1] = 999  # row1 identity mismatch -> tripwire, no write
    ok2 = torch.ones(2, dtype=torch.int32, device=dev)
    ctx.ctx_bulk_admit(poolL, srcp, kbt2, 0, ibt2[0], 0, 1229)
    ctx.ctx_batch_append(poolL, srcp, kbt2, ibt2, kvl2, exp2, ok2)
    torch.cuda.synchronize()

    def _tok(pool_blk_flat, src_blk_flat, off):
        a = torch.equal(
            pool_blk_flat[off * 128 : (off + 1) * 128],
            src_blk_flat[off * 128 : (off + 1) * 128],
        )
        b = torch.equal(
            pool_blk_flat[8192 + off * 4 : 8192 + off * 4 + 4],
            src_blk_flat[8192 + off * 4 : 8192 + off * 4 + 4],
        )
        return a and b

    assert _tok(
        poolL[3].reshape(-1), srcp[13], 17
    ), "bulk admit copy wrong"  # pos 3*64+17
    assert _tok(
        poolL[19].reshape(-1), srcp[29], 1229 % 64
    ), "batch append wrong"  # row0 cur
    assert int(ok2[0]) == 1 and int(ok2[1]) == 0, "tripwire flags wrong"
    assert int(poolL[20:25].sum()) == 0, "tripwired row must not be written"
    print("POOL OPS PASS")
    print("CTX TEST PASS")


if __name__ == "__main__":
    test()
