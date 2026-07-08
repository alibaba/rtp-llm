"""Real capture-path timing (GPU).

Does not hand-build PyAttentionInputs (hand-building drifts with the schema and
frequently goes all N/A); the inputs are taken from the engine's real
attention_inputs and LayerKVCache, and only qkv is synthesized (randn; latency
depends only on shape and memory access, it does not measure precision). Each kv
grid point does one real-machine capture+replay timing, going through the same
path and the same kernel as the engine's actual capture, so the schema is always
correct.

Hard constraints:
  - The throwaway cuda graph uses a private pool (does not pass the shared pool);
    otherwise, after release, returning to the shared pool and being reused by
    the actual capture would corrupt it.
  - After each (impl, kv) capture, immediately del g, synchronize, empty_cache;
    otherwise the private pool reservation grows monotonically until OOM.
  - The caller must ensure this module fully finishes and syncs before entering
    the engine's actual capture, to avoid nested capture.
  - Only modify sequence_lengths on the clone, never touch the engine's shared
    attn_inputs.
"""

from __future__ import annotations

import copy
import logging
import statistics
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_L2_FLUSH_MB = 256

# Each bench uses a dedicated private graph pool, to avoid crossing pool space with the engine's actual capture.
_bench_graph_pool_id: Optional[int] = None


def _get_bench_pool() -> int:
    """Return the bench-dedicated CUDA graph memory pool handle (process-unique)."""
    global _bench_graph_pool_id
    if _bench_graph_pool_id is None:
        _bench_graph_pool_id = torch.cuda.graph_pool_handle()
    return _bench_graph_pool_id


def _clone_attn_inputs(attn_inputs):
    """Deep clone PyAttentionInputs -- .clone() every Tensor field, copy scalars/None as-is.

    Purpose: any modification to attn_inputs during the bench phase (FlashInfer
    plan writing back buffer pointers, workspace allocation, etc.) does not leak
    onto the engine's shared attn_inputs object.
    """
    cloned = copy.copy(
        attn_inputs
    )  # shallow copy the C++ binding object (copies scalar fields)
    # clone Tensors field by field (prevent reference leaks; list[Tensor] is cloned element by element too)
    for attr in dir(cloned):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(cloned, attr)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            try:
                setattr(cloned, attr, val.clone())
            except Exception:
                pass  # skip read-only properties
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            try:
                setattr(cloned, attr, [t.clone() for t in val])
            except Exception:
                pass
    # Sanity: the reflection-based clone silently skips read-only properties (except:
    # pass above). Fail loud if a field the benchmark relies on ended up aliasing the
    # engine-shared original, instead of silently polluting shared inputs.
    for _f in ("input_lengths", "sequence_lengths"):
        _orig = getattr(attn_inputs, _f, None)
        _new = getattr(cloned, _f, None)
        if isinstance(_orig, torch.Tensor) and isinstance(_new, torch.Tensor):
            assert (
                _orig.data_ptr() != _new.data_ptr()
            ), f"_clone_attn_inputs: field '{_f}' was not independently cloned"
    return cloned


def _make_l2_filler(fill_mode: str):
    """L2 preprocessing closure (run before each iteration, outside timing). store = zero_() a 256MB buffer each iteration to flush L2.
    Returns (do_fill, keepalive_bufs)."""
    if fill_mode == "store":
        buf = torch.empty(_L2_FLUSH_MB * 1024 * 1024, dtype=torch.int8, device="cuda")
        return (lambda: buf.zero_()), (buf,)
    if fill_mode == "none":
        return (lambda: None), ()
    raise ValueError(f"unknown l2 fill_mode: {fill_mode!r} (store/none)")


def _bench_eager(fn, warmup: int, iters: int, l2_fill_mode: str) -> List[float]:
    """eager (non-graph) CUDA event per-call timing, returns a list of per-call durations (us).

    Includes fixed launch overhead, but the relative ordering between backends at
    the same bs is still valid (the difference in decode kernel compute is the main
    signal). Used as a baseline for and escape hatch from graph timing.
    """
    do_fill, _keep = _make_l2_filler(l2_fill_mode)
    torch.cuda.synchronize()
    for _ in range(warmup):
        do_fill()
        fn()
    torch.cuda.synchronize()

    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        do_fill()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    times = [st[i].elapsed_time(en[i]) * 1000.0 for i in range(iters)]
    del do_fill, _keep
    torch.cuda.synchronize()
    return times


def _bench_graph(
    fn,
    warmup: int,
    iters: int,
    l2_fill_mode: str,
    prepare_fn=None,
) -> List[float]:
    """CUDA Graph capture+replay timing, closer to real graph latency, with no launch overhead.

    Safe-isolation measures:
      1. Use a private pool, not sharing pool space with the engine's actual capture.
      2. Do a side-stream warmup before capture, so the backend's lazy-init
         allocations complete before capture; otherwise dynamic allocs get
         captured into the graph and replay behavior becomes undefined.
      3. After capture, do the replay timing, and del g at the end to release the
         private pool reservation.
      4. synchronize plus empty_cache throughout, to ensure nothing is left behind.

    If prepare_fn is provided, it is called before each replay to simulate
    production's per-step prepare_for_cuda_graph_replay overhead (including
    FlashInfer plan and sync), and is included in the timing interval, so the
    result reflects real per-step latency rather than pure kernel time.
    """
    do_fill, _keep = _make_l2_filler(l2_fill_mode)

    # --- warmup on side stream (not counted into the default stream's graph dependencies) ---
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(max(3, warmup)):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # --- capture (private pool) ---
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_get_bench_pool()):
        fn()
    torch.cuda.synchronize()

    # --- replay warmup ---
    for _ in range(warmup):
        do_fill()
        if prepare_fn:
            prepare_fn()
        g.replay()
    torch.cuda.synchronize()

    # --- replay timing (includes prepare_fn, reflects real per-step cost) ---
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        do_fill()
        st[i].record()
        if prepare_fn:
            prepare_fn()
        g.replay()
        en[i].record()
    torch.cuda.synchronize()
    times = [st[i].elapsed_time(en[i]) * 1000.0 for i in range(iters)]

    # --- cleanup: release the graph's reservation in the private pool ---
    del g
    del do_fill, _keep
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return times


def _qkv_dim(attn_configs) -> int:
    return (
        attn_configs.head_num + 2 * attn_configs.kv_head_num
    ) * attn_configs.size_per_head


def bench_backend(
    impl_cls,
    attn_configs,
    attn_inputs,
    layer_kv_cache,
    parallelism_config,
    *,
    warmup: int = 10,
    iters: int = 50,
    l2_fill_mode: str = "store",
    use_graph: bool = False,
) -> Optional[float]:
    """Time one backend class on the engine's real capture inputs, returning the median duration (us).

    use_graph=False (default) uses eager CUDA event timing, with zero interference
    to graph state.
    use_graph=True uses throwaway CUDA graph capture+replay timing, which is more
    accurate, and uses triple isolation to avoid contaminating the engine's actual
    capture:
      1. Deep-copy attn_inputs, so the buffer pointers held by the bench impl do
         not leak back onto the engine's shared object.
      2. A private graph pool, which after release does not return to the engine's
         shared pool to be reused by the actual capture.
      3. Explicitly clean up the FlashInfer decode_wrapper, to avoid plan residue
         in the global workspace.

    Neither mode rewrites the original attn_inputs: graph mode operates on a clone,
    and eager mode never modifies it. If construction, forward, or capture fails,
    returns None (this impl is marked N/A in this bucket and dropped by the upper
    layer). qkv is synthesized with randn; kv_cache uses the engine's real layer
    cache.
    """
    bs = int(attn_inputs.input_lengths.size(0))
    qkv = torch.randn(
        bs, _qkv_dim(attn_configs), dtype=attn_configs.dtype, device="cuda"
    )
    # graph mode: operate on a clone, to prevent reference leaks
    bench_inputs = _clone_attn_inputs(attn_inputs) if use_graph else attn_inputs
    if use_graph:
        # The bench harness always captures+replays (real CG path). The engine's
        # select_decode_backend does NOT propagate is_cuda_graph onto attn_inputs
        # (it is set later, at winner instantiation), so it is still False here.
        # Building the impl in non-CG mode and then CG-capturing+replaying it is
        # undefined behavior (crash / hang / grossly inflated replay). Force the
        # clone into CG mode so the wrapper's fixed-buffer setup matches the
        # capture+replay the harness performs and the engine's real capture path.
        bench_inputs.is_cuda_graph = True
    try:
        # 3-arg unconditionally (all impls take parallelism_config=None); a TypeError
        # fallback would mask a real internal TypeError, see instantiate_decode_impl.
        impl = impl_cls(attn_configs, bench_inputs, parallelism_config)

        def fn():
            impl.forward(qkv, layer_kv_cache, 0)

        torch.cuda.synchronize()
        fn()  # warmup + verify it can forward
        torch.cuda.synchronize()

        if use_graph:
            # build prepare_fn: simulate production's prepare_cuda_graph call before each replay
            # (includes per-step overhead such as FlashInfer plan(), counted into timing)
            prepare_fn = None
            _prepare_cg = getattr(impl, "prepare_cuda_graph", None)
            if callable(_prepare_cg):
                # bind bench_inputs via default arg to avoid late-binding closure risk
                prepare_fn = lambda bi=bench_inputs: _prepare_cg(bi)  # noqa: E731
            times = _bench_graph(fn, warmup, iters, l2_fill_mode, prepare_fn=prepare_fn)
        else:
            times = _bench_eager(fn, warmup, iters, l2_fill_mode)

        # explicitly clean up the FlashInfer wrapper's plan state (if any)
        _cleanup_flashinfer_impl(impl)
        # Drop closures that still reference `impl` (fn, and the graph prepare
        # closure) BEFORE del impl, so CPython's refcount releases impl -- and its
        # workspace back to the pool -- this call rather than one call late.
        del fn
        if use_graph:
            del prepare_fn, _prepare_cg
        del impl
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return statistics.median(times)
    except Exception as e:  # general defense: a real exception -> N/A
        logger.warning(
            "[dispatcher] bench %s @ bs=%d N/A: %r", impl_cls.__name__, bs, e
        )
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
    finally:
        del qkv
        if use_graph:
            del bench_inputs


def _set_uniform_seq_len(cloned_inputs, bs: int, kv: int) -> None:
    """On the clone, set the kv length to a uniform kv (the same kv for the whole batch).

    Real-machine dumps show that in the engine's decode capture inputs, only
    sequence_lengths (host) actually encodes the kv length; the other length
    tensors and the block table are placeholder 0 during capture and filled by the
    engine at replay time. When the block table is all 0, any logical block maps to
    physical block 0, which can cover any kv <= max_seq_len. So changing only
    sequence_lengths safely changes the kernel's kv workload without going out of
    bounds (physical block 0 is always valid). The tensor stays host and pinned,
    consistent with the engine.
    """
    sl = torch.full((bs,), int(kv), dtype=torch.int32)
    try:
        sl = sl.pin_memory()
    except Exception:
        pass
    cloned_inputs.sequence_lengths = sl


def bench_backend_grid(
    impl_cls,
    attn_configs,
    attn_inputs,
    layer_kv_cache,
    parallelism_config,
    kv_list: List[int],
    *,
    warmup: int = 10,
    iters: int = 50,
    l2_fill_mode: str = "store",
    use_graph: bool = True,
) -> Optional[List[float]]:
    """Multi-kv-point real-machine timing: for each kv, set sequence_lengths=kv on a clone, and time capture+replay separately.

    Returns a list of median_us aligned with kv_list; if any kv fails, returns None
    (this impl is marked N/A in this bucket). Each kv uses an independent clone
    (required in graph mode: the graph binds that clone's tensor addresses; and it
    does not contaminate the engine's shared inputs).
    """
    bs = int(attn_inputs.input_lengths.size(0))
    qkv = torch.randn(
        bs, _qkv_dim(attn_configs), dtype=attn_configs.dtype, device="cuda"
    )
    # Fill KV cache block 0 with random data so the kernel operates on non-trivial
    # values (avoids degenerate all-zero softmax).  Block table remains all-zero
    # (all logical blocks → physical block 0), so only block 0 is actually read.
    # Block 0 is the reserved sentinel (BlockPool allocates from 1), never
    # assigned to real requests; attention kernels bound reads by sequence_lengths,
    # so residual random data here does not affect inference correctness.
    try:
        kv_base = layer_kv_cache.kv_cache_base
        if kv_base is not None and kv_base.dim() >= 1 and kv_base.shape[0] > 0:
            kv_base[0].normal_()
    except Exception:
        pass
    lats: List[float] = []
    try:
        for kv in kv_list:
            bench_inputs = _clone_attn_inputs(attn_inputs)
            # Force CG mode on the clone: the harness always captures+replays, so the
            # wrapper must be built in CG mode (fixed buffers). is_cuda_graph is still
            # False on the engine inputs at selection time; leaving it False makes the
            # impl non-CG, and CG-capturing a non-CG wrapper is undefined behavior
            # (crash / hang / inflated replay). See bench_backend for details.
            bench_inputs.is_cuda_graph = True
            _set_uniform_seq_len(bench_inputs, bs, kv)
            # 3-arg unconditionally (all impls take parallelism_config=None); a TypeError
            # fallback would mask a real internal TypeError, see instantiate_decode_impl.
            impl = impl_cls(attn_configs, bench_inputs, parallelism_config)

            def fn():
                impl.forward(qkv, layer_kv_cache, 0)

            torch.cuda.synchronize()
            fn()  # warmup + verify it can forward
            torch.cuda.synchronize()

            prepare_fn = None
            if use_graph:
                _prepare_cg = getattr(impl, "prepare_cuda_graph", None)
                if callable(_prepare_cg):
                    # bind bench_inputs via default arg to avoid late-binding closure risk
                    prepare_fn = lambda bi=bench_inputs: _prepare_cg(bi)  # noqa: E731
            times = (
                _bench_graph(fn, warmup, iters, l2_fill_mode, prepare_fn=prepare_fn)
                if use_graph
                else _bench_eager(fn, warmup, iters, l2_fill_mode)
            )
            lats.append(statistics.median(times))
            _cleanup_flashinfer_impl(impl)
            # Drop closures referencing `impl` before del so its workspace returns
            # to the pool this iteration, not one iteration late (GC lag).
            del fn
            if use_graph:
                del prepare_fn, _prepare_cg
            del impl, bench_inputs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return lats
    except Exception as e:  # general defense: a real exception -> N/A
        logger.warning(
            "[dispatcher] bench_grid %s @ bs=%d N/A: %r", impl_cls.__name__, bs, e
        )
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
    finally:
        del qkv


def _cleanup_flashinfer_impl(impl) -> None:
    """Explicitly clean up the FlashInfer impl's wrapper state, to avoid plan residue contaminating subsequent capture.

    FlashInfer's BatchDecodeWithPagedKVCacheWrapper.plan() writes plan metadata
    into the workspace buffer. If the bench impl's workspace buffer returns to the
    global pool and is then taken by the actual capture's impl, the old plan data
    may interfere with the new plan()'s realloc decision. Explicitly resetting the
    key flags removes this hazard.
    """
    # PyFlashinferDecodeImpl -> fmha_impl -> decode_wrapper
    fmha_impl = getattr(impl, "fmha_impl", None)
    wrapper = getattr(fmha_impl, "decode_wrapper", None) if fmha_impl else None
    if wrapper is None:
        # not a FlashInfer backend, no cleanup needed
        return
    # reset cuda_graph-related flags, to prevent the next wrapper misreading after the buffer is returned to the pool
    try:
        wrapper._use_cuda_graph = False
        wrapper._fixed_batch_size = 0
    except Exception:
        pass
    # if end_forward exists (older FlashInfer API), call it
    end_fn = getattr(wrapper, "end_forward", None)
    if callable(end_fn):
        try:
            end_fn()
        except Exception:
            pass
