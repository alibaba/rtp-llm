"""Unit test for ctx_mirror_blocks_d2h: correctness + bandwidth vs the row path."""

import importlib.util as ilu
import os
import time

import torch

spec = ilu.spec_from_file_location(
    "v32_ctx", "/home/admin/rtp-hol/v32ctx_build/v32_ctx.so"
)
ctx = ilu.module_from_spec(spec)
spec.loader.exec_module(ctx)
ctx.ctx_init()

BS, ROW = 64, 576
NBLK = 1200
pool = torch.randn(NBLK * BS, ROW, dtype=torch.bfloat16, device="cuda")
host = torch.empty(NBLK * BS, ROW, dtype=torch.bfloat16, device="cpu", pin_memory=True)

# mirror logical blocks [0, 64) from scattered physical blocks
n = 64
phys = torch.randperm(NBLK - 1)[:n].to(torch.int32) + 1
ctx.ctx_mirror_blocks_d2h(pool, phys, host, 0, BS)
for i in range(n):
    exp = pool[int(phys[i]) * BS : (int(phys[i]) + 1) * BS].cpu()
    got = host[i * BS : (i + 1) * BS]
    assert torch.equal(exp, got), f"block {i} mismatch"
print("content PASS")

# offset destination
ctx.ctx_mirror_blocks_d2h(pool, phys[:8], host, 500 * BS, BS)
for i in range(8):
    exp = pool[int(phys[i]) * BS : (int(phys[i]) + 1) * BS].cpu()
    assert torch.equal(exp, host[(500 + i) * BS : (501 + i) * BS]), f"offset block {i}"
print("offset PASS")

# bandwidth: block path vs row path (4096 tokens, as one mirror chunk)
slots = (phys.long()[:, None] * BS + torch.arange(BS)[None, :]).reshape(-1)
host.zero_()
ctx.ctx_mirror_d2h(pool, slots, host[: n * BS])
assert torch.equal(
    pool[slots.cuda()].cpu(), host[: n * BS]
), "row path content mismatch"
print("row path content PASS")

mb = n * BS * ROW * 2 / 1e6
for _ in range(3):
    ctx.ctx_mirror_blocks_d2h(pool, phys, host, 0, BS)
t0 = time.perf_counter()
for _ in range(10):
    ctx.ctx_mirror_blocks_d2h(pool, phys, host, 0, BS)
dt = (time.perf_counter() - t0) / 10
print(f"block path: {dt*1e3:.3f} ms for {mb:.2f} MB -> {mb/dt/1e3:.1f} GB/s")

t0 = time.perf_counter()
for _ in range(10):
    ctx.ctx_mirror_d2h(pool, slots, host[: n * BS])
dt2 = (time.perf_counter() - t0) / 10
print(f"row path  : {dt2*1e3:.3f} ms for {mb:.2f} MB -> {mb/dt2/1e3:.1f} GB/s")
print(f"speedup x{dt2/dt:.1f}")

# per-request projection: 61 layers x (63k tokens / 4096 chunk)
chunks = 61 * (63000 // 4096 + 1)
print(f"projected per-request mirror: block={chunks*dt:.2f}s row={chunks*dt2:.2f}s")
print("MIRROR TEST PASS")
