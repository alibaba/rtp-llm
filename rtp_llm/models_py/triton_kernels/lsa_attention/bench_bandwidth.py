import torch

def bench(fn, w=5, i=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(i): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/i

print("device:", torch.cuda.get_device_name(0))
res = []
for nbytes in (256 << 20, 1 << 30, 2 << 30):
    n = nbytes // 4
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, device="cuda", dtype=torch.float32)
    ms = bench(lambda: b.copy_(a))          # read + write
    gbs = 2 * nbytes / (ms / 1e3) / 1e9
    res.append(("copy", nbytes >> 20, ms, gbs))
    ms2 = bench(lambda: a.sum())            # read only
    gbs2 = nbytes / (ms2 / 1e3) / 1e9
    res.append(("sum", nbytes >> 20, ms2, gbs2))
for k, mb, ms, gbs in res:
    print(f"{k} {mb} MiB: {ms:.3f} ms -> {gbs:.0f} GB/s")
mx = max(r[3] for r in res)
print(f"BEST achievable: {mx:.0f} GB/s ({mx/1000:.2f} TB/s)")
