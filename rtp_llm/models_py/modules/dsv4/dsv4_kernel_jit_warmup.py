"""Startup JIT warmup for DSV4 prefill kernels.

The real startup gRPC warmup exercises full forward paths, but some JIT keys
are driven by batch-only constexpr branches or by DeepGEMM's M-derived dense
GEMM heuristics.  This module directly launches the small set of kernels whose
cold compiles are otherwise likely to happen after the health gate opens.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

_DEFAULT_DENSE_GEMM_M_GRID = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
    393216,
    524288,
    786432,
    1048576,
]

_BRANCH_KERNEL_JIT_WARMED_KEYS: set[tuple] = set()
_DENSE_GEMM_JIT_WARMED_KEYS: set[tuple] = set()


def _dist_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _dist_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        logging.exception("[DSV4 KernelWarmup] distributed barrier failed")
        raise


def _is_cuda_device(device: torch.device) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _assert_not_capturing() -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DSV4 kernel JIT warmup must not run inside CUDA graph capture")


def _sync_cuda(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize(device)


def _release_cuda_cache(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.empty_cache()


def _positive_ints(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            continue
        if ivalue > 0:
            result.append(ivalue)
    return result


def _collect_dsv4_branch_kernel_configs(v4: Any, v4_args: Any) -> Dict[str, tuple]:
    """Collect branch-kernel constexpr configs from the live V4 model."""

    ratios = _positive_ints(getattr(v4_args, "compress_ratios", []))
    if not ratios and v4 is not None:
        ratios = _positive_ints(
            getattr(getattr(layer, "attn", None), "compress_ratio", 0)
            for layer in getattr(v4, "layers", [])
        )
    ratios = sorted(set(ratios))

    window_size = int(getattr(v4_args, "window_size", 0) or 0)
    if window_size <= 0 and v4 is not None and getattr(v4, "layers", None):
        window_size = int(getattr(v4.layers[0].attn, "window_size", 0) or 0)
    if window_size <= 0:
        window_size = 1

    rope_head_dim = int(getattr(v4_args, "rope_head_dim", 0) or 0)
    if rope_head_dim <= 0 and v4 is not None and getattr(v4, "layers", None):
        rope_head_dim = int(getattr(v4.layers[0].attn, "rope_head_dim", 0) or 0)
    if rope_head_dim <= 0:
        rope_head_dim = 64

    topk = int(getattr(v4_args, "index_topk", 0) or 0)

    combine_configs = {(window_size, 1, 0)}
    compressor_configs = set()
    for ratio in ratios:
        combine_configs.add((window_size, ratio, topk))
        compressor_configs.add((512, rope_head_dim, ratio, ratio == 4))
        if ratio == 4:
            compressor_configs.add((128, rope_head_dim, ratio, True))

    return {
        "combine": tuple(sorted(combine_configs)),
        "compressor": tuple(sorted(compressor_configs)),
    }


@torch.inference_mode()
def warmup_compressor_combine_branch_kernels(
    *,
    device: torch.device,
    v4: Any = None,
    v4_args: Any = None,
    compress_ratio: Optional[int] = None,
    rope_head_dim: Optional[int] = None,
    window_size: Optional[int] = None,
    topk: Optional[int] = None,
    overlap: Optional[bool] = None,
) -> None:
    """Compile batched/varlen Triton branches that B=1 gRPC warmup misses."""

    device = torch.device(device)
    if not _is_cuda_device(device):
        return
    _assert_not_capturing()

    if v4 is not None or v4_args is not None:
        configs = _collect_dsv4_branch_kernel_configs(v4, v4_args)
    else:
        if compress_ratio is None or rope_head_dim is None or window_size is None:
            raise ValueError("branch warmup requires v4/v4_args or explicit config")
        ratio = int(compress_ratio)
        configs = {
            "combine": ((int(window_size), ratio, int(topk or 0)),),
            "compressor": (
                (
                    512,
                    int(rope_head_dim),
                    ratio,
                    bool(overlap),
                ),
            ),
        }

    warmup_key = (configs["combine"], configs["compressor"], str(device))
    if warmup_key in _BRANCH_KERNEL_JIT_WARMED_KEYS:
        return

    _dist_barrier()
    rank = _dist_rank()
    if rank == 0:
        logging.info(
            "[DSV4 BranchKernels] JIT warmup start: combine=%s compressor=%s",
            configs["combine"],
            configs["compressor"],
        )
    t0 = time.time()
    for cfg in configs["combine"]:
        window, ratio, cfg_topk = cfg
        _warmup_combine_topk_swa_indices_cp(
            window_size=window,
            compress_ratio=ratio,
            topk=cfg_topk,
            device=device,
        )
    for cfg in configs["compressor"]:
        head_dim, rope_dim, ratio, cfg_overlap = cfg
        _warmup_fused_kv_compress_norm_rope_insert(
            head_dim=head_dim,
            rope_head_dim=rope_dim,
            compress_ratio=ratio,
            overlap=cfg_overlap,
            device=device,
        )
    _sync_cuda(device)
    _dist_barrier()
    if rank == 0:
        logging.info("[DSV4 BranchKernels] JIT warmup done in %.2fs", time.time() - t0)
    _BRANCH_KERNEL_JIT_WARMED_KEYS.add(warmup_key)


def _warmup_combine_topk_swa_indices_cp(
    *,
    window_size: int,
    compress_ratio: int,
    topk: int,
    device: torch.device,
) -> None:
    from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
        combine_topk_swa_indices_cp,
    )

    num_tokens = 32
    topk_width = max(int(topk), 0)
    topk_indices = torch.zeros((num_tokens, topk_width), dtype=torch.int32, device=device)
    global_positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    m_value = max(int(window_size) + max(topk_width, 1) + 128, 256)
    n_value = 64 if topk_width else 0

    combine_topk_swa_indices_cp(
        topk_indices=topk_indices,
        global_positions=global_positions,
        sp_int=0,
        window_size=int(window_size),
        compress_ratio=int(compress_ratio),
        topk=topk_width,
        M=m_value,
        N=n_value,
        req_id_per_token=None,
        prefix_lengths=None,
    )

    req_id_per_token = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    prefix_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    combine_topk_swa_indices_cp(
        topk_indices=topk_indices,
        global_positions=global_positions,
        sp_int=0,
        window_size=int(window_size),
        compress_ratio=int(compress_ratio),
        topk=topk_width,
        M=m_value,
        N=n_value,
        req_id_per_token=req_id_per_token,
        prefix_lengths=prefix_lengths,
    )


def _warmup_fused_kv_compress_norm_rope_insert(
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    device: torch.device,
) -> None:
    from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
        INDEXER_ENTRY_BYTES,
        KV_ENTRY_BYTES,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
        run_fused_compress_kv_write,
    )

    head_dim = int(head_dim)
    rope_head_dim = int(rope_head_dim)
    compress_ratio = int(compress_ratio)
    coff = 1 + int(bool(overlap))
    num_tokens = 32
    positions = (
        torch.arange(num_tokens, dtype=torch.int64, device=device) * compress_ratio
        + (compress_ratio - 1)
    )
    max_position = int(positions[-1].item()) if num_tokens else 0
    n_raw = max(max_position + 1, coff * compress_ratio)
    raw_width = coff * head_dim

    token_to_req = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.zeros((num_tokens,), dtype=torch.int64, device=device)
    state_block_size = 256
    state_blocks = max(max_position // state_block_size + 1, 1)
    block_table = torch.zeros((1, state_blocks), dtype=torch.int32, device=device)
    state_cache = torch.zeros(
        (state_blocks, state_block_size, 2 * raw_width),
        dtype=torch.float32,
        device=device,
    )
    kv_raw = torch.zeros((n_raw, raw_width), dtype=torch.float32, device=device)
    score_raw = torch.zeros_like(kv_raw)
    ape = torch.zeros((compress_ratio, raw_width), dtype=torch.float32, device=device)
    rms_norm_weight = torch.ones((head_dim,), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(
        (max_position + 1, rope_head_dim), dtype=torch.float32, device=device
    )

    kv_block_size = 64
    entry_bytes = KV_ENTRY_BYTES if head_dim == 512 else INDEXER_ENTRY_BYTES
    kv_cache = torch.zeros((1, kv_block_size, entry_bytes), dtype=torch.uint8, device=device)

    for batched in (False, True):
        kwargs = {}
        if batched:
            kwargs["seq_start_per_req"] = torch.zeros((1,), dtype=torch.int32, device=device)
            kwargs["cu_seq_per_req"] = torch.tensor([0, n_raw], dtype=torch.int32, device=device)
        run_fused_compress_kv_write(
            state_cache=state_cache,
            token_to_req_indices=token_to_req,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=1.0e-6,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kv_slot_mapping=kv_slot_mapping,
            kv_raw=kv_raw,
            score_raw=score_raw,
            ape=ape,
            seq_start=0,
            disable_raw_path=False,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=bool(overlap),
            **kwargs,
        )


def _shape_key(kind: str, n_value: int, k_value: int) -> tuple[str, int, int]:
    return (str(kind), int(n_value), int(k_value))


def _maybe_add_shape(
    shapes: Dict[tuple[str, int, int], dict],
    key: tuple[str, int, int],
    info: dict,
) -> None:
    if key not in shapes:
        shapes[key] = info


def _collect_dsv4_dense_gemm_shapes(model: Any) -> Dict[tuple[str, int, int], dict]:
    """Collect representative dense DeepGEMM shapes from the live DSV4 model."""

    shapes: Dict[tuple[str, int, int], dict] = {}
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "CudaFp8DeepGEMMLinear" and hasattr(module, "N") and hasattr(module, "K"):
            weight = getattr(module, "weight", None)
            scale = getattr(module, "weight_scales", None)
            if weight is None or scale is None:
                continue
            key = _shape_key("fp8", int(module.N), int(module.K))
            _maybe_add_shape(
                shapes,
                key,
                {
                    "name": module_name,
                    "module": module,
                    "weight": weight,
                    "scale": scale,
                    "scale_ue8m0": bool(getattr(module, "scale_ue8m0", scale.dtype == torch.int32)),
                },
            )
            continue

        if getattr(module, "storage", None) == "fp4":
            weight = getattr(module, "weight", None)
            scale = getattr(module, "scale_gemm", None)
            if weight is None or scale is None:
                continue
            n_value = int(getattr(module, "out_features", weight.shape[0]))
            k_value = int(getattr(module, "in_features", weight.shape[1] * 2))
            key = _shape_key("fp8_fp4", n_value, k_value)
            _maybe_add_shape(
                shapes,
                key,
                {"name": module_name, "weight": weight, "scale": scale},
            )

    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "GroupedFP4Strategy":
            _collect_grouped_fp4_strategy_shapes(shapes, module_name, module)
        elif cls_name == "LocalLoopStrategy":
            _collect_local_loop_strategy_shapes(shapes, module_name, module)
    return shapes


def _collect_grouped_fp4_strategy_shapes(
    shapes: Dict[tuple[str, int, int], dict],
    module_name: str,
    strategy: Any,
) -> None:
    for name, weight_attr, scale_attr in (
        ("grouped_w13", "_w13", "_s13_dense_t"),
        ("grouped_w2", "_w2", "_s2_dense_t"),
    ):
        if not hasattr(strategy, weight_attr) or not hasattr(strategy, scale_attr):
            continue
        weight_stack = getattr(strategy, weight_attr)
        scale_stack_t = getattr(strategy, scale_attr)
        if weight_stack.dim() != 3:
            continue
        expert_idx = torch.zeros((1,), dtype=torch.long, device=weight_stack.device)
        weight = weight_stack[0]
        scale = torch.index_select(scale_stack_t, 0, expert_idx).squeeze(0).transpose(0, 1)
        n_value = int(weight.shape[0])
        k_value = int(weight.shape[1]) * 2
        key = _shape_key("fp8_fp4", n_value, k_value)
        _maybe_add_shape(
            shapes,
            key,
            {"name": f"{module_name}.{name}", "weight": weight, "scale": scale},
        )


def _collect_local_loop_strategy_shapes(
    shapes: Dict[tuple[str, int, int], dict],
    module_name: str,
    strategy: Any,
) -> None:
    for name, weight_attr, scale_attr in (
        ("local_w1", "_W1_w", "_W1_s_gemm_t"),
        ("local_w2", "_W2_w", "_W2_s_gemm_t"),
        ("local_w3", "_W3_w", "_W3_s_gemm_t"),
    ):
        if not hasattr(strategy, weight_attr) or not hasattr(strategy, scale_attr):
            continue
        weight_stack = getattr(strategy, weight_attr)
        scale_stack_t = getattr(strategy, scale_attr)
        if weight_stack.dim() != 3:
            continue
        expert_idx = torch.zeros((1,), dtype=torch.long, device=weight_stack.device)
        weight = weight_stack[0]
        scale = torch.index_select(scale_stack_t, 0, expert_idx).squeeze(0).transpose(0, 1)
        n_value = int(weight.shape[0])
        k_value = int(weight.shape[1]) * 2
        key = _shape_key("fp8_fp4", n_value, k_value)
        _maybe_add_shape(
            shapes,
            key,
            {"name": f"{module_name}.{name}", "weight": weight, "scale": scale},
        )


@torch.inference_mode()
def warmup_dense_gemm_jit(
    shapes: Dict[tuple[str, int, int], dict],
    *,
    max_m: int,
    device: torch.device,
) -> None:
    """Compile representative DeepGEMM dense GEMM M buckets at startup."""

    device = torch.device(device)
    if not _is_cuda_device(device) or not shapes:
        return
    _assert_not_capturing()

    m_grid = [int(m) for m in _DEFAULT_DENSE_GEMM_M_GRID if 0 < int(m) <= int(max_m)]
    if not m_grid:
        return

    shape_keys = tuple(sorted(shapes.keys()))
    warmup_key = (int(max_m), shape_keys, tuple(m_grid), str(device))
    if warmup_key in _DENSE_GEMM_JIT_WARMED_KEYS:
        return

    _dist_barrier()
    rank = _dist_rank()
    if rank == 0:
        logging.info(
            "[DSV4 DenseGEMM] JIT warmup start: %d shapes x %d M values: %s",
            len(shape_keys),
            len(m_grid),
            shape_keys,
        )
    t0 = time.time()
    for key in shape_keys:
        info = shapes[key]
        for m_value in m_grid:
            _launch_dummy_gemm(key=key, info=info, m_value=m_value, device=device)
        _sync_cuda(device)
        _release_cuda_cache(device)
    _dist_barrier()
    if rank == 0:
        logging.info("[DSV4 DenseGEMM] JIT warmup done in %.2fs", time.time() - t0)
    _DENSE_GEMM_JIT_WARMED_KEYS.add(warmup_key)


def _create_dummy_fp8_input(
    *,
    m_value: int,
    k_value: int,
    device: torch.device,
    scale_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    a = torch.zeros((m_value, k_value), dtype=torch.float8_e4m3fn, device=device)
    scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=a.shape,
        device=device,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if scale.dtype == torch.int32:
        scale.fill_(0x7F7F7F7F)
    else:
        scale.fill_(1.0)
    return a, scale


def _launch_dummy_gemm(
    *,
    key: tuple[str, int, int],
    info: dict,
    m_value: int,
    device: torch.device,
) -> None:
    kind, n_value, k_value = key
    if kind == "fp8":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        scale_ue8m0 = bool(info.get("scale_ue8m0", True))
        a, a_scale = _create_dummy_fp8_input(
            m_value=m_value,
            k_value=k_value,
            device=device,
            scale_ue8m0=scale_ue8m0,
        )
        out = torch.empty((m_value, n_value), dtype=torch.bfloat16, device=device)
        fp8_gemm_nt(
            (a, a_scale),
            (info["weight"], info["scale"]),
            out,
            c=None,
            disable_ue8m0_cast=not scale_ue8m0,
        )
        del a, a_scale, out
        return

    if kind == "fp8_fp4":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt

        a, a_scale = _create_dummy_fp8_input(
            m_value=m_value,
            k_value=k_value,
            device=device,
            scale_ue8m0=True,
        )
        out = torch.empty((m_value, n_value), dtype=torch.bfloat16, device=device)
        fp8_fp4_gemm_nt(
            (a, a_scale),
            (info["weight"], info["scale"]),
            out,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )
        del a, a_scale, out
        return

    raise ValueError(f"unknown dense GEMM warmup kind={kind!r}")
