"""Micro-benchmark: per-layer critical-path cost of the offload serve paths.

Models one decode step faithfully: 61 independently registered layer states, one
pre-built selection row per layer, and no per-iteration tensor writes inside the
timed region - an earlier version reset the selection row in the loop, which
added an ATen dispatch plus a kernel launch to every measurement.

Rows:
  pybind probe  : ctx_lossy_has x61 -> pybind dispatch + global lock + map lookup
  B(prefetch=0) : lossy, drop misses, no fetch -> scheme B's latency floor
  B(prefetch=8) : lossy + block-granular hot-pool prefetch on the copy stream
  C             : lossless token-granular gather on the compute stream
"""

import time

import torch
from v32_ctx_build import ctx

LAYERS = 61
TOPK = 2048
HIST_BLOCKS = 984
TAIL_BLOCKS = 256


def setup(dev, req_key, n_stg=32, miss_frac=0.10, seed=7):
    """Register all 61 layer states and build one selection row per layer."""
    torch.manual_seed(seed)
    hist = HIST_BLOCKS * 64
    nb = hist // 64
    bt = torch.zeros(nb, dtype=torch.int32, device=dev)
    bt[0] = 1
    bt[1 : 1 + n_stg] = torch.arange(100, 100 + n_stg, dtype=torch.int32, device=dev)
    bt[nb - TAIL_BLOCKS :] = torch.arange(
        500, 500 + TAIL_BLOCKS, dtype=torch.int32, device=dev
    )
    kbt = bt.unsqueeze(0).repeat(LAYERS, 1).contiguous()
    kv = torch.randn(hist, 576, dtype=torch.float32).to(torch.bfloat16).pin_memory()
    jpos = torch.arange(1, 1 + n_stg, dtype=torch.int32)
    sb = torch.arange(100, 100 + n_stg, dtype=torch.int32)
    for layer in range(LAYERS):
        ctx.ctx_lossy_register(req_key, layer, jpos, sb, nb + 128, 0)

    n_miss = int(TOPK * miss_frac)
    rows = []
    for _ in range(LAYERS):
        tail = torch.randint(
            hist - TAIL_BLOCKS * 64, hist, (TOPK - n_miss,), dtype=torch.int32
        )
        miss = torch.randint(
            (n_stg + 1) * 64, (nb - TAIL_BLOCKS) * 64, (n_miss,), dtype=torch.int32
        )
        rows.append(torch.cat([tail, miss]))
    sel = torch.stack(rows).to(dev)
    return hist, kbt, kv, sel, sel.clone(), n_miss


def release(req_key):
    ctx.ctx_lossy_release(req_key)


def timeit(step_fn, reset_fn, warmup=8, reps=7):
    """Return (critical_path_ms, total_issued_ms) for one step's 61 layer calls.

    serve rewrites the selection rows in place, so without a reset every step
    after the first would see already-remapped rows and do no work at all. The
    reset is one bulk copy issued outside the timed window.

    Two clocks matter. Syncing only the compute stream measures what actually
    lengthens a decode step. Syncing the device also waits for scheme B's private
    copy stream, which a real server would overlap with attention and the MoE
    GEMMs, so the difference is traffic B issues but can hide.
    """
    for s in range(warmup):
        reset_fn()
        torch.cuda.synchronize()
        step_fn(1000 + s)
    torch.cuda.synchronize()
    best_cp = best_tot = None
    for r in range(reps):
        reset_fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn(2000 + r)
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cp, tot = (t1 - t0) * 1000, (t2 - t0) * 1000
        best_cp = cp if best_cp is None else min(best_cp, cp)
        best_tot = tot if best_tot is None else min(best_tot, tot)
    return best_cp, best_tot


def main():
    dev = "cuda:0"
    ctx.ctx_init()
    pool = torch.zeros(1024 * 64, 576, dtype=torch.bfloat16, device=dev)
    print(
        f"geometry: {HIST_BLOCKS * 64 // 1000}k history, {TAIL_BLOCKS} resident tail "
        f"blocks, {LAYERS} layers/step, batch=1"
    )

    hist, kbt, kv, ktr, ktr0, _ = setup(dev, 90)

    def probe(step):
        for i in range(LAYERS):
            ctx.ctx_lossy_has(90, i)

    cp, tot = timeit(probe, lambda: None)
    print(
        f"\n  {'pybind probe':<14} crit {cp / LAYERS * 1000:5.2f} us/layer"
        "   (dispatch + lock + map, no kernel)"
    )
    release(90)

    def launch_probe(step):
        for _ in range(LAYERS):
            ctx.ctx_probe_launch(0)

    cp, tot = timeit(launch_probe, lambda: None)
    print(
        f"  {'launch floor':<14} crit {cp / LAYERS * 1000:5.2f} us/layer"
        "   (guard + one empty launch)"
    )

    for frac in (0.10, 0.20, 0.30):
        out = []
        for name, key, cap in (("B(prefetch=0)", 91, 0), ("B(prefetch=8)", 92, 8)):
            hist, kbt, kv, ktr, ktr0, n_miss = setup(dev, key, miss_frac=frac)

            def step_fn(step, key=key, hist=hist, kbt=kbt, kv=kv, ktr=ktr, cap=cap):
                for i in range(LAYERS):
                    ctx.ctx_lossy_serve(key, i, kv, kbt, ktr, i, pool, hist, step, cap)

            b0 = ctx.ctx_lossy_counters()
            cp, tot = timeit(step_fn, lambda ktr=ktr, ktr0=ktr0: ktr.copy_(ktr0))
            d = [a - b for a, b in zip(ctx.ctx_lossy_counters(), b0)]
            mb = cap * 64 * 576 * 2 * LAYERS / 1e6  # block-granular: whole blocks
            out.append((name, cp, tot, d, mb))
            release(key)

        hist, kbt, kv, ktr, ktr0, n_miss = setup(dev, 93, miss_frac=frac)

        def step_c(step, hist=hist, kbt=kbt, kv=kv, ktr=ktr):
            for i in range(LAYERS):
                ctx.ctx_lossless_serve(93, i, kv, kbt, ktr, i, pool, hist, 0)

        b0 = ctx.ctx_lossy_counters()
        cp, tot = timeit(step_c, lambda ktr=ktr, ktr0=ktr0: ktr.copy_(ktr0))
        d = [a - b for a, b in zip(ctx.ctx_lossy_counters(), b0)]
        out.append(("C(lossless)", cp, tot, d, n_miss * 576 * 2 * LAYERS / 1e6))
        release(93)

        print(f"\nmiss_frac={frac:.2f}  ({n_miss} of {TOPK} selections non-resident)")
        for name, cp, tot, d, mb in out:
            per_serve = d[1] / max(d[3], 1)
            print(
                f"  {name:<14} crit {cp / LAYERS * 1000:5.2f} us/layer  "
                f"total {tot / LAYERS * 1000:5.2f}   H2D {mb:5.1f} MB/step   "
                f"recovered {per_serve:3.0f}/serve"
            )


if __name__ == "__main__":
    main()
