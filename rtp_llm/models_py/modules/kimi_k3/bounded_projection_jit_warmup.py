"""Bounded K3 projection/FLA preparation using loaded model geometry.

DeepGEMM's heuristic-selected layouts still depend on M with compiled_dims=nk.
Reuse DSV41's representative grid for matching FP8/DG98 BF16 layouts; retain
DSV4's bounded BF16 grid. These samples do not establish exhaustive coverage.
"""

from __future__ import annotations

import logging
import os
import time


def _tensor_key(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _projection_rows(model, init_resource):
    return model._max_moe_tokens


def _fp8_projections(model):
    """Retain one actual outer wrapper per operand/dispatch layout."""
    prepared = set()
    for name, module in model.named_modules():
        if type(module).__name__ != "CudaFp8GEMMLinear":
            continue
        weight, scales = module.weight, module.weight_scales
        if weight.device.type != "cuda":
            continue
        key = (
            _tensor_key(weight),
            _tensor_key(scales),
            bool(module.scale_ue8m0),
            module._flashinfer_linear is not None,
        )
        if key not in prepared:
            prepared.add(key)
            yield name, module


def _kv_up_projections(model):
    """KV-up wrappers are lazy FMHA objects, not registered model children."""
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )
    from rtp_llm.utils.model_weight import W

    prepared = set()
    config = model.config.attn_config
    splits = (
        int(config.nope_head_dim),
        int(config.rope_head_dim),
        int(config.v_head_dim),
    )
    for layer in model.layers:
        if type(layer.attention).__name__ != "KimiK3MLA":
            continue
        weights = model.weight.weights[layer.index]
        weight = weights[W.mla_kv_b_w]
        fp8 = str(weight.dtype) in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz")
        if weight.device.type != "cuda" or not (
            fp8 or str(weight.dtype) == "torch.bfloat16"
        ):
            continue
        scales = weights[W.mla_kv_b_s] if fp8 else None
        key = _tensor_key(weight), _tensor_key(scales) if fp8 else None, splits
        if key not in prepared:
            prepared.add(key)
            if fp8:
                from rtp_llm.models_py.modules.factory.linear.factory import (
                    LinearFactory,
                )

                projection = LinearFactory.create_linear_from_weights(
                    weights,
                    W.mla_kv_b_w,
                    W.mla_kv_b_s,
                    None,
                    model.config.attention_projection_quant_config,
                )
            else:
                projection = CudaF16Linear(weight)
            yield layer.index, projection, splits


def _kv_up_rows(model, init_resource, layer, projection, splits, current_rows):
    from rtp_llm.models_py.modules.factory.attention.mla_prefill_plan import (
        plan_expanded_kv_capacity,
    )

    batch = int(init_resource.max_context_batch_size)
    sequence = int(model.config.max_seq_len)
    if batch <= 0 or sequence <= 0:
        raise ValueError("K3 KV-up requires positive context batch and sequence limits")
    # Real histories are limited per request; physical Q can add TP-1 padding.
    request_bound = batch * sequence + max(int(model.parallelism_config.tp_size), 1) - 1
    maximum = min(current_rows, request_bound)
    if model.kv_cache is None:
        return maximum
    n = projection.weight.shape[0]
    heads = n // (splits[0] + splits[2])
    budget = float(os.environ.get("RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB", "6.0"))
    for cache in model.kv_cache.get_layer_cache_groups(layer):
        # This is the same producer+attention storage recipe as K3 MLA.
        factor = 3 if str(cache.kv_cache_base.dtype) == "torch.float8_e4m3fn" else 2
        _, capacity = plan_expanded_kv_capacity(
            (0,),
            (1,),
            page_tokens=int(cache.seq_size_per_block),
            bytes_per_token=heads * sum(splits) * factor,
            budget_gib=budget,
        )
        # Budget zero disables production chunking. Preserve its actual request
        # limit, rather than silently applying the default six-GiB budget.
        bound = request_bound if budget == 0 else min(capacity, request_bound)
        maximum = max(maximum, bound)
    return maximum


def _fla_l2norm_specs(model, max_rows):
    """Cover FLA's NB specializations for the full packed cuLA input."""
    devices = {}
    for layer in model.layers:
        if type(layer.attention).__name__ != "KimiK3KDA":
            continue
        prefill = layer.attention.prefill
        heads, dim = int(prefill.local_num_k_heads), int(prefill.head_k_dim)
        if heads <= 0 or dim != 128 or prefill.conv_weights.device.type != "cuda":
            continue
        # cuLA normalizes the whole packed Q/K; cache pages no longer bound a
        # call. FLA flattens rows and heads before selecting constexpr NB.
        nb = (max_rows * heads + 65535) // 65536
        key = str(prefill.conv_weights.device)
        if nb > 0:
            previous = devices.get(key, (prefill.conv_weights.device, 0))
            devices[key] = previous[0], max(previous[1], nb)
    for device, nb in devices.values():
        yield device, tuple(range(1, nb + 1))


def _sample_m_values(max_rows, n, k, *, fp8, k_major, num_sms):
    """k_major selects verified FP8 K/K or BF16 K/MN operand dispatch."""
    from rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup import (
        _dense_gemm_m_grid,
        _generate_dense_gemm_warmup_m_grid,
    )

    if fp8 and k_major:
        return _generate_dense_gemm_warmup_m_grid(
            max_m=max_rows,
            n_value=n,
            k_value=k,
            kind="v41_fp8",
            num_sms=num_sms,
        )
    if not fp8 and k_major:
        from importlib.metadata import PackageNotFoundError, version

        try:
            matching_dg = version("deep-gemm") == "2.8.2+98a672e.cu132"
        except PackageNotFoundError:
            matching_dg = False
        if matching_dg:
            # Advisory DG98 default block-multiple policy, not a proof of
            # arbitrary runtime heuristic state or complete traffic closure.
            logging.info(
                "K3 DG98 BF16 K/MN advisory samples: default block multiples; "
                "runtime overrides unverified, retaining DSV4 fallback grid"
            )
            representatives = _generate_dense_gemm_warmup_m_grid(
                max_m=max_rows,
                n_value=n,
                k_value=k,
                kind="dg98_bf16_k_mn",
                num_sms=num_sms,
            )
            return tuple(
                sorted(set(representatives) | set(_dense_gemm_m_grid(max_rows)))
            )
    return tuple(_dense_gemm_m_grid(max_rows))


def _projection_scratch_bytes(rows, k, n, *, fp8):
    # BF16 source/output, plus FP8 quantizer output and a conservative FP32 SF
    # allocation. Loaded weights are retained without a duplicate allocation.
    return rows * (2 * k + 2 * n + (k if fp8 else 0)) + (
        ((rows + 3) // 4 * 4) * ((k + 127) // 128) * 4 if fp8 else 0
    )


def _check_projection_memory(device, needed):
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    if needed > free:
        raise RuntimeError(
            f"K3 directed projection scratch needs at least {needed} bytes, "
            f"but CUDA reports {free} free on {device}; configured expansion "
            "budget/request limits cannot be prepared with current free memory"
        )


def _warmup_fp8(module, samples, *, head_splits=None):
    import torch

    for rows in samples:
        hidden = torch.zeros(
            (rows, module.K), dtype=torch.bfloat16, device=module.weight.device
        )
        if head_splits is not None and module.supports_skip_head_mid(
            hidden, head_splits
        ):
            output = module.forward_skip_head_mid(hidden, head_splits)
        else:
            output = module(hidden)  # Match MLA's real unsupported-skip fallback.
        del hidden, output


def _warmup_bf16(projection, splits, samples):
    import torch

    for rows in samples:
        hidden = torch.zeros(
            (rows, projection.weight.shape[1]),
            dtype=torch.bfloat16,
            device=projection.weight.device,
        )
        if not projection.supports_skip_head_mid(hidden, splits):
            return False
        output = projection.forward_skip_head_mid(hidden, splits)
        del hidden, output
    return True


def _warmup_fla(device, buckets):
    import torch
    from fla.modules.l2norm import l2norm_fwd_kernel

    # NB is redundant in FLA 0.5.0+rtp.3a9ce1c.6. T is do_not_specialize, so
    # prepare the real dependency kernel with bounded scratch, not huge Q/K.
    x = torch.zeros((8, 128), device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    rstd = torch.empty((8,), device=device, dtype=torch.float32)
    for nb in buckets:
        l2norm_fwd_kernel[(1,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=1e-6,
            T=8,
            D=128,
            BD=128,
            NB=nb,
            BT=8,
            num_warps=1,
            num_stages=3,
            num_ctas=1,
        )


def warmup_kimi_k3_projection_jit(model, init_resource):
    from rtp_llm.utils.warmup import model_warm_up_enabled

    if not model_warm_up_enabled():
        return
    import torch

    if not torch.cuda.is_available():
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "K3 directed JIT warmup cannot run during CUDA Graph capture"
        )
    from rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup import (
        _get_deep_gemm_num_sms,
        _release_cuda_cache,
        _run_deepgemm_warmup_launches_serialized,
    )

    max_rows = _projection_rows(model, init_resource)
    if max_rows <= 0:
        return

    @torch.inference_mode()
    def prepare():
        for name, module in _fp8_projections(model):
            device = module.weight.device
            samples = _sample_m_values(
                max_rows,
                module.N,
                module.K,
                fp8=True,
                k_major=(
                    module.weight.stride(-1) == 1
                    and torch.cuda.get_device_capability(device)[0] == 10
                ),
                num_sms=_get_deep_gemm_num_sms(device),
            )
            started = time.perf_counter()
            logging.info(
                "K3 bounded FP8 JIT: module=%s N=%d K=%d launches=%d sampled_M=%s scratch_MiB=%.2f",
                name,
                module.N,
                module.K,
                len(samples),
                samples,
                _projection_scratch_bytes(
                    max(samples, default=0), module.K, module.N, fp8=True
                )
                / 2**20,
            )
            _warmup_fp8(module, samples)
            torch.cuda.synchronize(device)
            _release_cuda_cache(device)
            logging.info(
                "K3 bounded FP8 JIT done: module=%s cost=%.3fs",
                name,
                time.perf_counter() - started,
            )
        if init_resource.is_decode_role:
            return
        for layer, projection, splits in _kv_up_projections(model):
            device = projection.weight.device
            n, k = projection.weight.shape
            fp8 = str(projection.weight.dtype) in (
                "torch.float8_e4m3fn",
                "torch.float8_e4m3fnuz",
            )
            probe = torch.zeros((1, k), dtype=torch.bfloat16, device=device)
            supported = projection.supports_skip_head_mid(probe, splits)
            del probe
            if not supported and not fp8:
                logging.info(
                    "K3 BF16 KV-up JIT skipped: layer=%d backend unsupported", layer
                )
                continue
            kv_rows = _kv_up_rows(
                model, init_resource, layer, projection, splits, max_rows
            )
            bf16_k_mn = (
                not fp8
                and supported
                and type(projection).__name__ == "CudaF16Linear"
                and str(projection.weight.dtype) == "torch.bfloat16"
                and projection.weight.stride(-2) == 1
                and projection.weight.stride(-1) == n
                and torch.cuda.get_device_capability(device)[0] == 10
            )
            samples = _sample_m_values(
                kv_rows,
                n,
                k,
                fp8=fp8,
                # BF16 weight.T is contiguous: DG receives A=K/B=MN.
                # Other skip-head layouts retain the bounded fallback grid.
                k_major=(
                    bf16_k_mn
                    or (
                        fp8
                        and not supported
                        and projection.weight.stride(-1) == 1
                        and torch.cuda.get_device_capability(device)[0] == 10
                    )
                ),
                num_sms=_get_deep_gemm_num_sms(device) if fp8 or bf16_k_mn else 1,
            )
            packed_n = n // (splits[0] + splits[2]) * sum(splits) if supported else n
            started = time.perf_counter()
            logging.info(
                "K3 bounded KV-up JIT: layer=%d fp8=%s skip_head_mid=%s splits=%s sampled_M=%s scratch_MiB=%.2f",
                layer,
                fp8,
                supported,
                splits,
                samples,
                _projection_scratch_bytes(max(samples, default=0), k, packed_n, fp8=fp8)
                / 2**20,
            )
            _release_cuda_cache(device)
            _check_projection_memory(
                device,
                _projection_scratch_bytes(
                    max(samples, default=0), k, packed_n, fp8=fp8
                ),
            )
            if fp8:
                _warmup_fp8(projection, samples, head_splits=splits)
                prepared = True
            else:
                prepared = _warmup_bf16(projection, splits, samples)
            torch.cuda.synchronize(device)
            _release_cuda_cache(device)
            if not prepared:
                logging.info(
                    "K3 BF16 KV-up JIT skipped: layer=%d backend unsupported", layer
                )
            else:
                logging.info(
                    "K3 KV-up JIT done: layer=%d fp8=%s cost=%.3fs",
                    layer,
                    fp8,
                    time.perf_counter() - started,
                )
        for device, buckets in _fla_l2norm_specs(model, max_rows):
            logging.info(
                "K3 FLA l2norm JIT: device=%s NB=%s scratch_bytes=4128", device, buckets
            )
            _warmup_fla(device, buckets)
            torch.cuda.synchronize(device)

    _run_deepgemm_warmup_launches_serialized("K3 bounded projections", prepare)
