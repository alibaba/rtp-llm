"""Explicit adapters for the legacy and DeepJIT-backed DeepGEMM Mega APIs."""

import importlib.metadata
import inspect
import re


def deep_gemm_uses_deepjit(version: str | None = None) -> bool:
    """Inspect wheel metadata without fixing a device or constructing the JIT."""
    if version is None:
        try:
            version = importlib.metadata.version("deep_gemm")
        except importlib.metadata.PackageNotFoundError:
            return False
    parsed = re.match(r"(\d+)\.(\d+)", version)
    return parsed is not None and tuple(map(int, parsed.groups())) >= (2, 8)


def mega_moe_uses_shared32(deep_gemm) -> bool:
    parameters = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
    required = {"shared_l1_weights", "shared_l2_weights", "recipe", "activation_clamp"}
    if not {"shared_l1_weights", "shared_l2_weights"}.intersection(parameters):
        return False
    missing = required.difference(parameters)
    if missing:
        raise RuntimeError(
            f"DeepGEMM Mega API is missing parameters: {sorted(missing)}"
        )
    return "shared_recipe" not in parameters


def mega_moe_shared_kwargs(deep_gemm, block_size: int) -> dict:
    parameters = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
    if not {"shared_l1_weights", "shared_l2_weights"}.issubset(parameters):
        raise RuntimeError("This DeepGEMM Mega API has no fused shared-expert support")
    shared32 = mega_moe_uses_shared32(deep_gemm)
    if block_size == 32 and shared32:
        return {}
    if block_size == 128 and not shared32:
        return {"shared_recipe": (1, 128, 128)}
    if block_size == 128:
        raise RuntimeError(
            "This DeepGEMM Mega API requires shared32; V4 shared128 must use "
            "routed Mega plus the independent shared128 executor"
        )
    raise RuntimeError(
        f"DeepGEMM Mega-SE does not support shared block size {block_size}"
    )


def mega_moe_activation_kwargs(deep_gemm, block_size: int) -> dict:
    if block_size == 128:
        return {}
    if block_size != 32:
        raise ValueError(f"unsupported shared FP8 block size: {block_size}")
    parameters = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
    if "round_swiglu_to_bf16" not in parameters:
        raise RuntimeError(
            "V4.1 Mega requires the explicit BF16 activation-rounding patch"
        )
    return {"round_swiglu_to_bf16": True}


def mega_moe_combine_kwargs(deep_gemm, block_size: int) -> dict:
    if block_size not in (32, 128):
        raise ValueError(f"unsupported shared FP8 block size: {block_size}")
    parameters = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
    if "torch_sum_combine" in parameters:
        return {"torch_sum_combine": True}
    return {}


def mega_moe_numerics_kwargs(deep_gemm, block_size: int) -> dict:
    """Validate the selected model's math before allocating or warming buffers."""
    activation = mega_moe_activation_kwargs(deep_gemm, block_size)
    combine = mega_moe_combine_kwargs(deep_gemm, block_size)
    if block_size == 32 and not combine:
        raise RuntimeError("V4.1 Mega requires the explicit torch_sum_combine patch")
    return {**activation, **combine}


def mega_moe_dispatch_kwargs(deep_gemm, use_fp8_dispatch: bool) -> dict:
    parameters = inspect.signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters
    if "mma_type" in parameters:
        if not use_fp8_dispatch:
            raise ValueError("The RTP FP4 routed Mega path requires FP8 dispatch")
        return {"mma_type": "fp8xfp4"}
    if "use_fp8_dispatch" not in parameters:
        raise RuntimeError("DeepGEMM buffer API has no recognized dispatch parameter")
    return {"use_fp8_dispatch": bool(use_fp8_dispatch)}


def mega_moe_symm_buffer_bytes(
    deep_gemm,
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    num_shared_experts: int = 0,
) -> int:
    if group_size <= 0 or num_max_tokens_per_rank <= 0:
        raise ValueError("Mega group size and token capacity must be positive")
    dispatch = mega_moe_dispatch_kwargs(deep_gemm, use_fp8_dispatch)
    alignment_getter = getattr(deep_gemm._C, "get_token_alignment_for_mega_moe", None)
    if alignment_getter is not None:
        alignment = int(alignment_getter())
        if alignment <= 0:
            raise ValueError("DeepGEMM returned an invalid Mega token alignment")
        num_max_tokens_per_rank = (
            (num_max_tokens_per_rank + alignment - 1) // alignment * alignment
        )
    args = (
        group_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        next(iter(dispatch.values())),
        activation,
    )
    if "mma_type" in dispatch:
        args += (num_shared_experts,)
    elif num_shared_experts:
        raise RuntimeError(
            "This legacy DeepGEMM buffer API has no shared-expert layout"
        )
    size, slice_callback = deep_gemm._C.get_symm_buffer_size_for_mega_moe(*args)
    if int(size) <= 0 or not callable(slice_callback):
        raise RuntimeError("DeepGEMM returned an invalid symmetric buffer layout")
    return int(size)


def validate_mega_moe_buffer_bytes(buffer, estimated_bytes: int) -> int:
    actual = int(buffer.buffer.nbytes)
    if actual != estimated_bytes:
        raise RuntimeError(
            f"DeepGEMM symmetric buffer size mismatch: actual={actual}, "
            f"estimated={estimated_bytes}"
        )
    return actual


def prepare_mega_shared_scale(deep_gemm, scale, mn: int, k: int, block_size: int):
    import torch

    if block_size not in (32, 128):
        raise ValueError(f"unsupported shared FP8 block size: {block_size}")
    if block_size == 128:
        if scale.dtype == torch.int32:
            return scale
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"expected shared128 UE8M0 scale, got {scale.dtype}")
        return deep_gemm.transform_sf_into_required_layout(
            scale.float(), mn, k, (128, 128), num_groups=None
        )
    if mn % 32 or k % 128:
        raise ValueError(f"shared32 requires aligned MN/K, got {mn}/{k}")
    if scale.dtype == torch.int32:
        if tuple(scale.shape) != (mn, k // 128):
            raise ValueError(
                f"invalid packed shared32 scale shape: {tuple(scale.shape)}"
            )
        return scale
    if scale.dtype == torch.uint8:
        values = scale.contiguous().view(torch.float8_e8m0fnu).float()
    elif scale.dtype in (torch.float8_e8m0fnu, torch.float32):
        values = scale.float()
    else:
        raise TypeError(f"expected shared32 FP32/UE8M0 scale, got {scale.dtype}")
    if tuple(values.shape) == (mn // 32, k // 32):
        values = values.repeat_interleave(32, dim=0)
    elif tuple(values.shape) != (mn, k // 32):
        raise ValueError(f"invalid raw shared32 scale shape: {tuple(values.shape)}")
    mantissa, exponent = torch.frexp(values)
    if not bool(((mantissa == 0.5) & (exponent >= -126) & (exponent <= 128)).all()):
        raise ValueError("shared32 scales must be finite positive UE8M0 values")
    return deep_gemm.transform_sf_into_required_layout(
        values, mn, k, (1, 32), num_groups=None
    )


def mega_moe_jit_token_counts(deep_gemm, cfg, capacity: int) -> list[int]:
    """Enumerate the installed new API's actual tiling, including empty ranks.

    At the pinned DeepJIT revision all token-dependent template choices derive
    from block_m. The legacy wave heuristic remains in the existing caller.
    """
    if capacity < cfg.max_tokens_per_rank:
        raise ValueError("Mega buffer capacity is smaller than the configured limit")
    if cfg.max_tokens_per_rank < 0:
        raise ValueError("Mega configured token limit must be nonnegative")
    representatives = []
    seen = set()
    for tokens in range(cfg.max_tokens_per_rank + 1):
        block_m = int(
            deep_gemm.get_block_m_for_mega_moe(
                cfg.ep_size,
                cfg.n_routed_experts,
                capacity,
                tokens,
                cfg.n_activated_experts,
                "fp8xfp4",
            )
        )
        if block_m not in seen:
            seen.add(block_m)
            representatives.append(tokens)
    if representatives[-1] != cfg.max_tokens_per_rank:
        representatives.append(cfg.max_tokens_per_rank)
    return representatives
