from pathlib import Path
from typing import Optional

import torch
from aiter.jit.core import AITER_CSRC_DIR, compile_ops
from aiter.ops.mha import cmdGenFunc_mha_batch_prefill

_GRAPH_SOURCE = Path(__file__).with_name("csrc") / "mha_batch_prefill_graph_pybind.cu"


def _graph_build_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dropout_randval: torch.Tensor,
    rng_state: torch.Tensor,
    block_table: torch.Tensor,
    sanitized_block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    page_claims: torch.Tensor,
    linear_v: bool,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
):
    build_args = cmdGenFunc_mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        0.0,
        False,
        True,
        -1,
        -1,
        0,
        False,
        False,
        out,
        None,
        None,
        q_descale,
        k_descale,
        v_descale,
        None,
        None,
        block_table,
        seqlen_k,
        None,
        None,
    )
    build_args["md_name"] = f"rtp_llm_graph_v3_{build_args['md_name']}"
    build_args["srcs"] = [
        f"{AITER_CSRC_DIR}/kernels/mha_common.cu",
        f"{AITER_CSRC_DIR}/py_itfs_ck/mha_batch_prefill_kernels.cu",
        str(_GRAPH_SOURCE),
        f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_batch_prefill.cu",
    ]
    return build_args


@compile_ops(
    "module_mha_batch_prefill",
    fc_name="mha_batch_prefill_graph",
    gen_func=_graph_build_args,
)
def _mha_batch_prefill_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dropout_randval: torch.Tensor,
    rng_state: torch.Tensor,
    block_table: torch.Tensor,
    sanitized_block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    page_claims: torch.Tensor,
    linear_v: bool,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...


_ready_specializations = set()
_failed_specializations = set()


def _specialization_key(
    dtype: torch.dtype, output_dtype: torch.dtype, has_descale: bool
):
    return dtype, output_dtype, has_descale


def graph_prefill_is_ready(
    dtype: torch.dtype, output_dtype: torch.dtype, has_descale: bool
) -> bool:
    return (
        _specialization_key(dtype, output_dtype, has_descale) in _ready_specializations
    )


def graph_prefill_has_failed(
    dtype: torch.dtype, output_dtype: torch.dtype, has_descale: bool
) -> bool:
    return (
        _specialization_key(dtype, output_dtype, has_descale) in _failed_specializations
    )


def graph_prefill_is_ready_for(output_dtype: torch.dtype, has_descale: bool) -> bool:
    return any(
        ready_output_dtype == output_dtype and ready_has_descale == has_descale
        for _, ready_output_dtype, ready_has_descale in _ready_specializations
    )


def mha_batch_prefill_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dropout_randval: torch.Tensor,
    rng_state: torch.Tensor,
    block_table: torch.Tensor,
    sanitized_block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    page_claims: Optional[torch.Tensor] = None,
    linear_v: bool = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if page_claims is None:
        page_claims = sanitized_block_table
    key = _specialization_key(q.dtype, out.dtype, q_descale is not None)
    if key in _failed_specializations:
        raise RuntimeError(f"AIter graph prefill specialization is disabled: {key}")
    try:
        result = _mha_batch_prefill_graph(
            q,
            k,
            v,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            out,
            softmax_lse,
            dropout_randval,
            rng_state,
            block_table,
            sanitized_block_table,
            seqlen_k,
            page_claims,
            linear_v,
            q_descale,
            k_descale,
            v_descale,
        )
    except Exception:
        _failed_specializations.add(key)
        raise
    _ready_specializations.add(key)
    return result
