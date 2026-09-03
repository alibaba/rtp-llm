"""Startup capability and ABI checks for unified DSV4 Mega decode."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

from .mega_csa_weights import (
    GEOMETRY_BY_DIM,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    MAX_BATCH,
    O_LORA_RANK,
    PRO_GEOMETRY,
    ROPE_DIM,
)
from .mega_hca_weights import HCA_COMPRESS_RATIO, HCA_STATE_WIDTH

_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
_EXTENSION_SYMBOLS_BY_COMPONENT = {
    "csa": (
        "front_mixed_gemm_csa",
        "geometry_csa",
        "hc_reduce_fuse_out",
        "mla_o_inv_rope_quant",
        "mqa_logits_fp8_decode_out",
        "wq_b_proj_gemm_merged_csa",
    ),
    "hca": (
        "front_mixed_gemm_hca",
        "geometry_hca",
        "hc_reduce_fuse_out",
        "mla_o_inv_rope_quant",
        "q_rmsnorm_rope_cuda_",
        "wq_b_proj_gemm_merged_hca",
    ),
    "moe_front": ("Dsv4MoeFrontPlan", "geometry_moe_front"),
}
_REQUIRED_EXTENSION_SYMBOLS = tuple(
    sorted(
        {name for names in _EXTENSION_SYMBOLS_BY_COMPONENT.values() for name in names}
    )
)
_REQUIRED_EXTENSION_PARAMETERS = {
    "hc_reduce_fuse_out": (
        "attn_norm_w",
        "mix_out",
        "xq_out",
        "xsf_out",
        "pdl",
    ),
    "front_mixed_gemm_csa": ("hc_mix", "main_state", "idx_state", "pdl"),
    "wq_b_proj_gemm_merged_csa": (
        "indexer_fp8",
        "idx_cache",
        "idx_state_block_table",
        "swa_cache",
        "iq_dst",
        "pdl",
    ),
    "mqa_logits_fp8_decode_out": (
        "schedule_meta",
        "comp_state",
        "comp_state_block_table",
        "cmp_cache",
        "query_x",
        "query_out",
        "pdl",
    ),
    "front_mixed_gemm_hca": (
        "normalized_mix",
        "state_slot_mapping",
        "state_kv",
        "state_gate",
        "pdl",
    ),
    "wq_b_proj_gemm_merged_hca": (
        "state_ring_entries",
        "window_cache",
        "compressed_cache",
        "window_page_tokens",
        "compressed_page_tokens",
    ),
    "q_rmsnorm_rope_cuda_": ("q", "freqs_cis", "positions", "eps"),
    "mla_o_inv_rope_quant": (
        "input",
        "positions",
        "rope_cos",
        "rope_sin",
        "output_fp8",
        "output_scale",
    ),
}
_DEEP_GEMM_SYMBOLS_BY_COMPONENT = {
    "csa": (
        "get_num_sms",
        "get_paged_mqa_logits_metadata",
        "tf32_hc_prenorm_gemm",
    ),
    "hca": ("tf32_hc_prenorm_gemm",),
    "moe_front": ("tf32_hc_prenorm_gemm",),
}
_REQUIRED_DEEP_GEMM_SYMBOLS = tuple(
    sorted(
        {name for names in _DEEP_GEMM_SYMBOLS_BY_COMPONENT.values() for name in names}
    )
)


def _model_geometry_reason(args: Any) -> Optional[str]:
    geometry = GEOMETRY_BY_DIM.get(int(args.dim))
    if geometry is None:
        return f"unsupported hidden size {args.dim}; expected {sorted(GEOMETRY_BY_DIM)}"
    expected = (
        ("q_lora_rank", args.q_lora_rank, geometry.q_lora_rank),
        ("n_heads", args.n_heads, geometry.main_heads),
        ("head_dim", args.head_dim, HEAD_DIM),
        ("rope_head_dim", args.rope_head_dim, ROPE_DIM),
        ("o_groups", args.o_groups, geometry.o_groups),
        ("o_lora_rank", args.o_lora_rank, O_LORA_RANK),
        ("index_n_heads", args.index_n_heads, INDEX_HEADS),
        ("index_head_dim", args.index_head_dim, INDEX_HEAD_DIM),
    )
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, actual, wanted in expected
        if int(actual) != wanted
    ]
    return "model geometry mismatch: " + "; ".join(mismatches) if mismatches else None


def _requirements(
    table: Mapping[str, Sequence[str]], components: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        sorted({name for component in components for name in table[component]})
    )


def _runtime_unavailable_reason(
    device: torch.device,
    components: Sequence[str],
) -> tuple[Optional[str], Optional[Any]]:
    device = torch.device(device)
    if device.type != "cuda":
        return f"CUDA is required, got {device}", None
    try:
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:
        return f"failed to query CUDA capability for {device}: {exc}", None
    if capability not in _SUPPORTED_CAPABILITIES:
        return (
            f"sm_100a or sm_103a is required, got sm_{capability[0]}{capability[1]}",
            None,
        )

    try:
        from rtp_kernel import dsv4_mega
    except Exception as exc:
        return f"failed to import rtp_kernel.dsv4_mega: {exc}", None

    required_symbols = _requirements(_EXTENSION_SYMBOLS_BY_COMPONENT, components)
    missing = [
        name
        for name in required_symbols
        if not callable(getattr(dsv4_mega, name, None))
    ]
    if missing:
        return "rtp-kernel is missing DSV4 Mega ABI: " + ", ".join(missing), None

    incompatible = []
    for function_name in required_symbols:
        parameters = _REQUIRED_EXTENSION_PARAMETERS.get(function_name)
        if parameters is None:
            continue
        try:
            signature = inspect.signature(getattr(dsv4_mega, function_name))
        except (TypeError, ValueError) as exc:
            incompatible.append(f"{function_name} has no inspectable signature ({exc})")
            continue
        absent = [name for name in parameters if name not in signature.parameters]
        if absent:
            incompatible.append(f"{function_name} missing {','.join(absent)}")
    if incompatible:
        return (
            "rtp-kernel DSV4 Mega ABI is incompatible: " + "; ".join(incompatible),
            None,
        )

    try:
        import deep_gemm
    except Exception as exc:
        return f"failed to import DeepGEMM: {exc}", None
    deep_gemm_symbols = _requirements(_DEEP_GEMM_SYMBOLS_BY_COMPONENT, components)
    missing = [name for name in deep_gemm_symbols if not hasattr(deep_gemm, name)]
    if missing:
        return "DeepGEMM is missing required DSV4 API: " + ", ".join(missing), None
    return None, dsv4_mega


def _mapping_mismatch(
    name: str, actual: Any, expected: Mapping[str, int]
) -> Optional[str]:
    if not isinstance(actual, Mapping):
        return f"rtp-kernel {name} geometry is not a mapping"
    mismatched = {
        key: (actual.get(key), wanted)
        for key, wanted in expected.items()
        if actual.get(key) != wanted
    }
    return f"rtp-kernel {name} geometry mismatch: {mismatched}" if mismatched else None


def _compiled_geometry_reason(
    args: Any, dsv4_mega: Any, *, require_moe_front: bool
) -> Optional[str]:
    geometry = GEOMETRY_BY_DIM[int(args.dim)]
    csa_suffix = "" if geometry is PRO_GEOMETRY else "_flash"
    hca_suffix = "_pro" if geometry is PRO_GEOMETRY else "_flash"
    try:
        csa_geometry = dsv4_mega.geometry_csa()
        hca_geometry = dsv4_mega.geometry_hca()
        front_geometry = (
            dsv4_mega.geometry_moe_front(int(args.dim)) if require_moe_front else None
        )
    except Exception as exc:
        return f"failed to query rtp-kernel DSV4 Mega ABI: {exc}"

    reason = _mapping_mismatch(
        "CSA",
        csa_geometry,
        {
            f"n_main{csa_suffix}": geometry.n_main,
            "n_index": INDEX_HEADS * INDEX_HEAD_DIM,
            f"n_merged{csa_suffix}": geometry.n_merged,
            f"num_main_heads{csa_suffix}": geometry.main_heads,
            "num_index_heads": INDEX_HEADS,
            "slot_dtype_bits": 64,
        },
    )
    if reason is not None:
        return reason
    reason = _mapping_mismatch(
        "HCA",
        hca_geometry,
        {
            f"n_q{hca_suffix}": geometry.n_main,
            f"front_n_fp8{hca_suffix}": geometry.front_fp8_rows,
            "compress_ratio": HCA_COMPRESS_RATIO,
            "state_width": HCA_STATE_WIDTH,
            "slot_dtype_bits": 64,
        },
    )
    if reason is not None or not require_moe_front:
        return reason
    return _mapping_mismatch(
        "MoE-front",
        front_geometry,
        {
            "abi_version": 1,
            "kernel_contract_version": 2,
            "hidden": int(args.dim),
            "hc_mult": int(args.hc_mult),
            "experts": int(args.n_routed_experts),
            "topk": int(args.n_activated_experts),
            "max_m": MAX_BATCH,
        },
    )


def require_mega_runtime(device: torch.device, components: Sequence[str]) -> Any:
    """Return the extension module or raise for missing hardware/dependencies."""

    reason, dsv4_mega = _runtime_unavailable_reason(device, components)
    if reason is not None:
        raise RuntimeError(reason)
    assert dsv4_mega is not None
    return dsv4_mega


def mega_decode_unavailable_reason(args: Any, device: torch.device) -> Optional[str]:
    """Return why the complete CSA/HCA/MoE-front Mega path is unavailable."""

    if not bool(args.fp8_kv_cache):
        return "FP8 KV cache is required"
    if int(args.tp_size) != 1:
        return f"TP1 is required, got TP{args.tp_size}"
    geometry_reason = _model_geometry_reason(args)
    if geometry_reason is not None:
        return geometry_reason

    require_moe_front = int(args.ep_size) > 1
    components = ("csa", "hca", "moe_front") if require_moe_front else ("csa", "hca")
    reason, dsv4_mega = _runtime_unavailable_reason(device, components)
    if reason is not None:
        return reason
    assert dsv4_mega is not None
    return _compiled_geometry_reason(
        args, dsv4_mega, require_moe_front=require_moe_front
    )


__all__ = ["mega_decode_unavailable_reason", "require_mega_runtime"]
