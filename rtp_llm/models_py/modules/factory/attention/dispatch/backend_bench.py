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
  - After each (impl, kv) capture, release the graph, closures, implementation,
    workspace, and temporary tensors before synchronizing and emptying the cache;
    otherwise private-pool reservations can accumulate until OOM.
  - The caller must ensure this module fully finishes and syncs before entering
    the engine's actual capture, to avoid nested capture.
  - Only modify sequence-length fields on the clone, never touch the engine's
    shared attn_inputs.
"""

from __future__ import annotations

import copy
import statistics
from typing import List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.benchmark_workspace import (
    benchmark_workspace_scope,
)

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
        val = getattr(cloned, attr)
        if isinstance(val, torch.Tensor):
            cloned_val = val.clone()
            try:
                setattr(cloned, attr, cloned_val)
            except AttributeError:
                pass  # skip read-only properties
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            cloned_val = [t.clone() for t in val]
            try:
                setattr(cloned, attr, cloned_val)
            except AttributeError:
                pass
    # Read-only properties cannot be replaced. Fail loud if a field the benchmark
    # relies on ended up aliasing the engine-shared original.
    for _f in (
        "input_lengths",
        "sequence_lengths",
        "sequence_lengths_plus_1_device",
    ):
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


def _bench_graph(
    fn,
    warmup: int,
    iters: int,
    l2_fill_mode: str,
    prepare_fn=None,
    attention_layer_count: int = 1,
) -> List[float]:
    """CUDA Graph capture+replay model-step score, including host overhead.

    Safe-isolation measures:
      1. Use a private pool, not sharing pool space with the engine's actual capture.
      2. Do a side-stream warmup before capture, so the backend's lazy-init
         allocations complete before capture; otherwise dynamic allocs get
         captured into the graph and replay behavior becomes undefined.
      3. After capture, do the replay timing, and del g at the end to release the
         private pool reservation.
      4. On healthy completion, synchronize and release unused cached allocations.

    If prepare_fn is provided, it is called before each replay to simulate
    production's per-step prepare_for_cuda_graph_replay overhead (including
    FlashInfer plan and sync), and is included in the timing interval. The
    production implementation prepares once per model step and is then reused
    by every attention layer. For multi-layer models, replay-only cost is
    measured on the same graph and scaled by attention_layer_count.
    """
    if attention_layer_count < 1:
        raise ValueError("attention_layer_count must be positive")
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

    times = _time_graph_replays(g, do_fill, iters, prepare_fn)
    if attention_layer_count > 1:
        replay_only_times = (
            times
            if prepare_fn is None
            else _time_graph_replays(g, do_fill, iters, prepare_fn=None)
        )
        times = _model_step_times(times, replay_only_times, attention_layer_count)
        del replay_only_times

    # The caller drops the graph closures, implementation, workspace, and
    # temporary tensors before performing allocator cleanup.
    del g
    del do_fill, _keep
    return times


def _time_graph_replays(g, do_fill, iters: int, prepare_fn) -> List[float]:
    """Measure graph replay, optionally including one per-step prepare call."""
    # Synchronizing each start event excludes the asynchronous L2 filler while
    # establishing the timing boundary before host-side replay preparation.
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        do_fill()
        st[i].record()
        st[i].synchronize()
        if prepare_fn:
            prepare_fn()
        g.replay()
        en[i].record()
    torch.cuda.synchronize()
    times = [st[i].elapsed_time(en[i]) * 1000.0 for i in range(iters)]
    return times


def _model_step_times(
    combined_times: List[float],
    replay_only_times: List[float],
    attention_layer_count: int,
) -> List[float]:
    """Scale one prepare plus one replay to one prepare plus all layer replays."""
    if attention_layer_count < 1:
        raise ValueError("attention_layer_count must be positive")
    replay_us = statistics.median(replay_only_times)
    return [
        combined_us + (attention_layer_count - 1) * replay_us
        for combined_us in combined_times
    ]


def _qkv_dim(attn_configs) -> int:
    return (
        attn_configs.head_num + 2 * attn_configs.kv_head_num
    ) * attn_configs.size_per_head


def _fill_synthetic_kv_block(kv_base) -> None:
    """Fill the benchmarked cache block through a universally supported dtype."""
    if kv_base is None or kv_base.dim() < 1 or kv_base.shape[0] == 0:
        return
    block = kv_base[0]
    random_block = torch.randn(block.shape, dtype=torch.float32, device=block.device)
    block.copy_(random_block)


def _set_uniform_seq_len(cloned_inputs, bs: int, kv: int) -> None:
    """Set host and replay-device KV-length mirrors on benchmark inputs.

    The host tensor is consumed when an implementation is constructed. Production
    replay also refreshes implementation state from sequence_lengths_plus_1_device,
    so the benchmark must update that mirror in place before capture and replay.
    The block table remains unchanged: logical blocks still map to the valid
    physical block 0 used by the benchmark fixture.
    """
    fields = (
        ("sequence_lengths", "cpu"),
        ("sequence_lengths_plus_1_device", "cuda"),
    )
    tensors = {}
    for field, expected_device in fields:
        tensor = getattr(cloned_inputs, field, None)
        context = f"field={field} bs={bs} kv={kv}"
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{context}: required tensor is missing")
        if tensor.dtype != torch.int32:
            raise ValueError(
                f"{context}: expected dtype=torch.int32, got {tensor.dtype}"
            )
        if tensor.device.type != expected_device:
            raise ValueError(
                f"{context}: expected device={expected_device}, got {tensor.device}"
            )
        if tensor.dim() < 1 or tensor.size(0) < bs:
            raise ValueError(
                f"{context}: expected first dimension >= {bs}, got shape={tuple(tensor.shape)}"
            )
        tensors[field] = tensor

    cloned_inputs.sequence_lengths = torch.full(
        (bs,), int(kv), dtype=torch.int32
    ).pin_memory()
    tensors["sequence_lengths_plus_1_device"][:bs].fill_(int(kv) + 1)


def bench_backend_grid(
    impl_cls,
    attn_configs,
    attn_inputs,
    layer_kv_cache,
    parallelism_config,
    kv_list: List[int],
    *,
    attention_layer_count: int = 1,
    warmup: int = 10,
    iters: int = 50,
    l2_fill_mode: str = "store",
) -> Optional[List[float]]:
    """Multi-kv-point real-machine timing: for each kv, set sequence_lengths=kv on a clone, and time capture+replay separately.

    Returns model-step-equivalent median_us values aligned with kv_list. Prepare
    runs once per step while attention replay cost is scaled to the model's
    attention layer count. A real instance reporting no CUDA Graph support
    returns None as a normal unavailable-candidate result; probe exceptions
    propagate to the selector's fatal boundary. Each kv uses an independent
    clone so captured addresses cannot contaminate shared inputs.
    """
    bs = int(attn_inputs.input_lengths.size(0))
    with benchmark_workspace_scope():
        qkv = torch.randn(
            bs, _qkv_dim(attn_configs), dtype=attn_configs.dtype, device="cuda"
        )
        # Fill KV cache block 0 with random data so the kernel operates on non-trivial
        # values (avoids degenerate all-zero softmax). Block table remains all-zero
        # (all logical blocks -> physical block 0), so only block 0 is actually read.
        _fill_synthetic_kv_block(layer_kv_cache.kv_cache_base)
        lats: List[float] = []
        for kv in kv_list:
            bench_inputs = _clone_attn_inputs(attn_inputs)
            # Force CG mode on the clone: the harness always captures+replays, so the
            # wrapper must be built in CG mode (fixed buffers). is_cuda_graph is still
            # False on the engine inputs at selection time; leaving it False makes the
            # impl non-CG, and CG-capturing a non-CG wrapper is undefined behavior
            # (crash / hang / inflated replay).
            bench_inputs.is_cuda_graph = True
            _set_uniform_seq_len(bench_inputs, bs, kv)
            # 3-arg unconditionally (all impls take parallelism_config=None); a TypeError
            # fallback would mask a real internal TypeError, see instantiate_decode_impl.
            impl = impl_cls(attn_configs, bench_inputs, parallelism_config)
            if not impl.support_cuda_graph():
                del impl, bench_inputs, qkv
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return None

            def fn():
                impl.forward(qkv, layer_kv_cache, 0)

            torch.cuda.synchronize()
            fn()  # warmup + verify it can forward
            torch.cuda.synchronize()

            prepare_fn = None
            _prepare_cg = getattr(impl, "prepare_cuda_graph", None)
            if callable(_prepare_cg):
                # bind bench_inputs via default arg to avoid late-binding closure risk
                prepare_fn = lambda bi=bench_inputs: _prepare_cg(bi)  # noqa: E731
            times = _bench_graph(
                fn,
                warmup,
                iters,
                l2_fill_mode,
                prepare_fn=prepare_fn,
                attention_layer_count=attention_layer_count,
            )
            lats.append(statistics.median(times))
            # Drop closures before impl so its temporary workspace becomes
            # unreferenced before healthy-path allocator cleanup.
            del fn
            del prepare_fn, _prepare_cg
            del impl, bench_inputs, times
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del qkv
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return lats
