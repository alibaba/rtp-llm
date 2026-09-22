"""Startup warmup for the V4.1 prefill ABI (MXFP8 group32 and FP4 KV).

Run from model initialization, including the second initialization that binds
the real cache. Dummy launches use private outputs; model weights and KV data
are never overwritten. The ordinary startup request is still responsible for
the complete forward/collective path.
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial

import torch

from rtp_llm.models_py.modules.dsv4 import dsv4_kernel_jit_warmup as common
from rtp_llm.utils.warmup import model_warm_up_enabled

_DENSE_WARMED: set[tuple] = set()
_SHARED_WARMED: set[tuple] = set()
_ENGRAM_WARMED: set[tuple] = set()


def collect_v41_dense_shapes(model):
    """Keep MXFP8 separate from V4's group128 FP8/FP4 scale contracts."""
    linears, outputs = {}, {}
    for name, module in model.named_modules():
        if type(module).__name__ == "V41MXFP8Linear":
            key = (int(module.N), int(module.K))
            linears.setdefault(key, (name, module))
        if type(module).__name__ == "AttentionV41FP8":
            weight = getattr(module, "_wo_a_stk_w", None)
            if isinstance(weight, torch.Tensor):
                key = tuple(int(dim) for dim in weight.shape)
                outputs.setdefault(key, (name, module))
    return linears, outputs


def _dense_m_grid(max_m, n, k, num_sms, groups=1):
    # Share V4's exhaustive heuristic-layout scan, but launch the actual V41
    # wrapper: it supplies recipe=(1,1,32), including the activation quantizer.
    grid = set(
        common._generate_dense_gemm_warmup_m_grid(
            max_m=max_m,
            n_value=n,
            k_value=k,
            kind="v41_fp8_batched" if groups > 1 else "v41_fp8",
            num_sms=num_sms,
            num_groups=groups,
        )
    )
    # Quantization dispatch and Triton alignment variants are independent of
    # the GEMM layout. Cover both sides of the 4M-element v2 threshold.
    threshold = (4 * 1024 * 1024 + k - 1) // k
    grid.update(
        m for m in (1, 2, 3, 4, 16, 17, threshold - 1, threshold) if 0 < m <= max_m
    )
    return tuple(sorted(grid))


@torch.inference_mode()
def warmup_v41_dense_jit(model, *, max_m, device):
    if not model_warm_up_enabled() or not common._is_cuda_device(device):
        return
    common._assert_not_capturing()
    linears, outputs = collect_v41_dense_shapes(model)
    sms = common._get_deep_gemm_num_sms(device)
    key = (
        str(device),
        max_m,
        tuple(sorted(linears)),
        tuple(sorted(outputs)),
        sms,
        os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto"),
        os.environ.get("DSV41_FUSED_OUTPUT_PROJECTION", "1"),
    )
    if key in _DENSE_WARMED:
        return

    def launch():
        for (n, k), (name, module) in linears.items():
            grid = _dense_m_grid(max_m, n, k, sms)
            logging.info(
                "[DSV41 DenseGEMM] %s N=%d K=%d M=%s recipe=32", name, n, k, grid
            )
            for m in grid:
                x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
                common._run_deepgemm_warmup_launch_with_retry(
                    "DSV41 DenseGEMM",
                    f"{name} M={m}",
                    partial(module, x),
                    device=device,
                )
                del x
            common._sync_cuda(device)
            common._release_cuda_cache(device)
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_output_projection

        for (groups, n, k), (name, attn) in outputs.items():
            grid = _dense_m_grid(max_m, n, k, sms, groups)
            logging.info("[DSV41 BatchedFP8Einsum] %s M=%s recipe=32", name, grid)
            for m in grid:
                x = torch.zeros(
                    (m, attn.n_heads, attn.head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                freqs = torch.ones(
                    (m, attn.rope_head_dim // 2), dtype=torch.complex64, device=device
                )
                if _v41_output_projection.is_supported(
                    x, freqs, attn._wo_a_stk_w, attn._wo_a_stk_s
                ):
                    common._run_deepgemm_warmup_launch_with_retry(
                        "DSV41 BatchedFP8Einsum",
                        f"{name} M={m}",
                        partial(
                            _v41_output_projection.grouped_output_projection,
                            x,
                            freqs,
                            attn._wo_a_stk_w,
                            attn._wo_a_stk_s,
                        ),
                        device=device,
                    )
                del x, freqs
            common._sync_cuda(device)
            common._release_cuda_cache(device)

    common._run_deepgemm_warmup_launches_serialized("DSV41 DenseGEMM", launch)
    _DENSE_WARMED.add(key)


def _shared_expert_signatures(model):
    """Collect the standalone native-group32 path, without running its executor."""
    signatures = {}
    for name, moe in model.named_modules():
        executor = getattr(moe, "_shared_executor", None)
        if getattr(executor, "name", None) != "mxfp8" or getattr(
            moe, "_routed_includes_shared", False
        ):
            continue
        shared = moe.shared_experts
        dim, inter = int(shared.w13.K), int(shared.w13.N) // 2
        if (
            dim <= 0
            or inter <= 0
            or int(shared.w13.N) != 2 * inter
            or (int(shared.w2.N), int(shared.w2.K)) != (dim, inter)
        ):
            raise ValueError(f"invalid V41 shared-expert geometry: {name}")
        strategy = moe._strategy
        # Mega returns its BF16 buffer; the other standalone strategies return
        # contiguous FP32 accumulators. Do not call forward/setup_runtime here:
        # those mutate shared runtime buffers and can enter EP collectives.
        routed = getattr(strategy, "_mega_y", None)
        if isinstance(routed, torch.Tensor):
            if routed.ndim != 2 or routed.shape[1] != dim:
                raise ValueError(f"invalid V41 routed output geometry: {name}")
            routed_dtype, routed_stride = routed.dtype, tuple(routed.stride())
        elif strategy.name == "mega":
            routed_dtype, routed_stride = torch.bfloat16, (dim, 1)
        elif strategy.name in ("local_loop", "grouped_fp4", "deepep"):
            routed_dtype, routed_stride = torch.float32, (dim, 1)
        else:
            raise ValueError(f"unknown V41 routed output contract: {strategy.name}")
        clamp = max(float(shared.swiglu_limit), 0.0)
        signatures.setdefault((dim, inter, clamp, routed_dtype, routed_stride), name)
    return signatures


@torch.inference_mode()
def warmup_v41_shared_expert_jit(model, *, max_m, device):
    """Warm standalone MXFP8 activation/combine using only private tensors."""
    if not model_warm_up_enabled() or not common._is_cuda_device(device) or max_m <= 0:
        return
    common._assert_not_capturing()
    signatures = _shared_expert_signatures(model)
    if not signatures:
        return
    from rtp_llm.models_py.modules.dsv4.moe._shared_expert_triton import (
        fused_moe_epilogue,
    )
    from rtp_llm.models_py.modules.dsv4.moe._silu_mul_bf16_triton import (
        silu_mul_split_bf16,
    )

    # The fused BF16 kernel does not specialize M. One row warms every token
    # count; width/clamp still come from the real shared-expert projections.
    rows_grid = (1,)
    fused_add = os.environ.get("DSV4_SHARED_EXPERT_BF16_ADD", "0") != "1"
    key = (str(device), frozenset(signatures), rows_grid, fused_add)
    if key in _SHARED_WARMED:
        return
    for (dim, inter, clamp, routed_dtype, routed_stride), name in signatures.items():
        for rows in rows_grid:
            gate_up = torch.zeros(
                (rows, 2 * inter), dtype=torch.bfloat16, device=device
            )
            common._run_triton_warmup_launch_with_retry(
                "DSV41 SharedExpert",
                f"{name} silu M={rows} D={inter} clamp={clamp}",
                partial(silu_mul_split_bf16, gate_up, clamp_limit=clamp),
                device=device,
            )
        if fused_add:
            # add_cast explicitly does_not_specialize M. One launch covers
            # every token count for this dtype/stride contract. In particular,
            # MXFP8's shared result is BF16; add_cast promotes both inputs
            # to FP32 inside the kernel before adding.
            rows = min(max_m, 2)
            routed = torch.empty_strided(
                (rows, dim), routed_stride, dtype=routed_dtype, device=device
            ).zero_()
            shared = torch.zeros((rows, dim), dtype=torch.bfloat16, device=device)
            common._run_triton_warmup_launch_with_retry(
                "DSV41 SharedExpert",
                f"{name} add_cast N={dim} routed={routed_dtype} stride={routed_stride}",
                partial(fused_moe_epilogue, routed, shared, torch.bfloat16),
                device=device,
            )
    common._sync_cuda(device)
    _SHARED_WARMED.add(key)
    logging.info(
        "[DSV41 SharedExpert] JIT warmup done; contracts=%d silu_M=%s fused_add=%s",
        len(signatures),
        rows_grid,
        fused_add,
    )


@torch.inference_mode()
def warmup_v41_engram_jit(model, *, max_m, device):
    if not model_warm_up_enabled() or not common._is_cuda_device(device):
        return
    state = getattr(model, "_engram_hash_state", None)
    if state is None:
        return
    common._assert_not_capturing()
    from rtp_llm.models_py.modules.dsv4.engram import gated_engram_residual

    layers = [layer.engram for layer in model.v4.layers if layer.engram is not None]
    if not layers or max_m <= 0:
        return
    layout = state.layout
    # HostEngramEmbedding passes the physical SM count captured at UVA setup
    # to lookup_host_rows. A DeepGEMM SM cap must not truncate lookup coverage.
    lookup_sms = tuple(int(layer.embed_tokens._num_sms) for layer in layers)
    if min(lookup_sms) <= 0:
        raise ValueError("Engram lookup requires a positive physical SM count")
    sms = max(lookup_sms)
    key = (
        str(device),
        layout.layer_ids,
        layout.n_hash_cols,
        layout.head_dim,
        tuple(
            (
                tuple(layer.q_weight.shape),
                layer.eps,
                tuple(layer.embed_tokens.weight.shape),
            )
            for layer in layers
        ),
        max_m,
        lookup_sms,
    )
    if key in _ENGRAM_WARMED:
        return
    # The UVA lookup uses GRID=min(ceil(tokens*heads/16),SMs) as constexpr.
    # Every reachable grid is represented, including the saturated grid and
    # aligned/unaligned token-count variants. Table contents stay on the host.
    limit = min(max_m, (sms * 16 + layout.n_hash_cols - 1) // layout.n_hash_cols + 16)
    for rows in range(1, limit + 1):
        windows = torch.zeros(
            (rows, layout.max_ngram_size), dtype=torch.int32, device=device
        )
        dead = torch.zeros_like(windows, dtype=torch.bool)
        hashes = None

        def hash_launch():
            nonlocal hashes
            hashes = state.hash_token_windows(windows, dead_mask=dead)

        common._run_triton_warmup_launch_with_retry(
            "DSV41 Engram", f"hash rows={rows}", hash_launch, device=device
        )
        for layer in layers:
            ids = hashes[:, layer.layer_hash_index].contiguous()
            common._run_triton_warmup_launch_with_retry(
                "DSV41 Engram",
                f"lookup rows={rows}",
                partial(layer.embed_tokens, ids, device),
                device=device,
            )
        if rows in (1, 16):
            for layer in layers:
                hc, dim = layer.q_weight.shape
                hidden = torch.zeros(
                    (rows, hc, dim), dtype=torch.bfloat16, device=device
                )
                kv = torch.zeros(
                    (rows, (hc + 1) * dim), dtype=torch.bfloat16, device=device
                )
                mask = torch.ones(rows, dtype=torch.bool, device=device)
                for token_mask in (None, mask):
                    common._run_triton_warmup_launch_with_retry(
                        "DSV41 Engram",
                        f"inject rows={rows} mask={token_mask is not None}",
                        partial(
                            gated_engram_residual,
                            hidden,
                            kv,
                            layer.q_weight,
                            layer.k_weight,
                            layer.eps,
                            token_mask,
                        ),
                        device=device,
                    )
    common._sync_cuda(device)
    _ENGRAM_WARMED.add(key)
    logging.info(
        "[DSV41 Engram] JIT warmup done; lookup grid covered through %d tokens", limit
    )


def _prefill_topology(model):
    config = getattr(model.parallelism_config, "prefill_cp_config", None)
    enabled = config is not None and bool(config.is_enabled())
    cp_size = max(int(model.parallelism_config.tp_size), 1) if enabled else 1
    return cp_size, bool(getattr(config, "kv_cache_sharded", False))


def resolve_v41_prefill_warmup_max_m(model, cp_size):
    """Bound rank-local rows by both workspace capacity and scheduler inputs."""
    capacity = max(int(model._resolve_prefill_q_token_capacity()), 1)
    total_tokens = int(getattr(model, "_max_prefill_batch_tokens", 0))
    if total_tokens <= 0:
        # Compatibility with older initialization resources: the allocated
        # workspace remains the safe upper bound when no scheduler cap exists.
        return capacity
    if cp_size <= 1:
        return min(capacity, total_tokens)
    batch = common.resolve_cp_metadata_warmup_max_batch_size(
        model._max_context_batch_size, model._max_generate_batch_size
    )
    # Each nonempty request contributes two equally padded CP chunks. Bound
    # the sum of per-request ceil divisions without assuming equal lengths.
    parts = 2 * cp_size
    padded = 2 * ((total_tokens + parts - 1) // parts)
    padded += 2 * (min(batch, total_tokens) - 1)
    return max(min(capacity, padded), 1)


@torch.inference_mode()
def warmup_v41_prefill_jit(model, *, device):
    """Same initialization/health-gate lifecycle as V4, with V41 kernel ABIs."""
    device = torch.device(device)
    if (
        not model_warm_up_enabled()
        or model._is_decode_role
        or not common._is_cuda_device(device)
    ):
        return
    common._assert_not_capturing()
    from rtp_llm.models_py.modules.dsv4.fp8._v41_attention_jit_warmup import (
        warmup_v41_attention_jit,
    )
    from rtp_llm.models_py.modules.dsv4.hc.v41_jit_warmup import warmup_v41_hc_jit

    cp_size, sharded = _prefill_topology(model)
    # MoE's separate chunk cap does not bound attention, Engram, or mHC.
    max_m = resolve_v41_prefill_warmup_max_m(model, cp_size)
    max_batch_size = common.resolve_cp_metadata_warmup_max_batch_size(
        model._max_context_batch_size, model._max_generate_batch_size
    )
    start = time.monotonic()
    logging.info(
        "[DSV41 Prefill] JIT warmup start max_m=%d max_batch_size=%d "
        "max_batch_tokens=%d cp=%d sharded=%s cache_bound=%s seq_size_per_block=%s",
        max_m,
        max_batch_size,
        int(getattr(model, "_max_prefill_batch_tokens", 0)),
        cp_size,
        sharded,
        model.kv_cache is not None,
        getattr(model.kv_cache, "seq_size_per_block", None),
    )
    common.warmup_prefill_cp_metadata_jit(
        is_decode_role=False,
        cp_enabled=cp_size > 1,
        cp_size=cp_size,
        max_batch_size=max_batch_size,
        # Only shared CP metadata here. V41 has different SWA/FP4 pool ABIs,
        # warmed below using private pools with the bound cache's layout.
        fp8_kv_cache=False,
        kv_cache_sharded=sharded,
        device=device,
    )
    # Shared experts can retain V4's FP8/FP4 linears even though attention
    # and Engram use MXFP8. MegaMoE's symmetric-memory kernels already warm
    # themselves in strategy.setup_runtime(), outside this local lock.
    common.warmup_dense_gemm_jit(
        common._collect_dsv4_dense_gemm_shapes(model), max_m=max_m, device=device
    )
    warmup_v41_dense_jit(model, max_m=max_m, device=device)
    warmup_v41_shared_expert_jit(model, max_m=max_m, device=device)
    warmup_v41_hc_jit(model.v4, max_m=max_m, device=device)
    warmup_v41_engram_jit(model, max_m=max_m, device=device)
    warmup_v41_attention_jit(
        model.v4,
        max_seq_len=int(model._v4_args.max_seq_len),
        max_m=max_m,
        max_batch_size=max_batch_size,
        cp_size=cp_size,
        cp_rank=common._dist_rank() % cp_size,
        kv_cache_sharded=sharded,
        device=device,
        kv_cache=model.kv_cache,
    )
    common._sync_cuda(device)
    common._release_cuda_cache(device)
    logging.info(
        "[DSV41 Prefill] JIT warmup done in %.2fs cache_bound=%s",
        time.monotonic() - start,
        model.kv_cache is not None,
    )
