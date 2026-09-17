import threading
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
    scratch_block_table: torch.Tensor,
    linear_v: bool,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    k_scratch: Optional[torch.Tensor] = None,
    v_scratch: Optional[torch.Tensor] = None,
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
    build_args["md_name"] = f"rtp_llm_graph_v4_{build_args['md_name']}"
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
    scratch_block_table: torch.Tensor,
    linear_v: bool,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    k_scratch: Optional[torch.Tensor] = None,
    v_scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...


_ready_specializations = set()
_failed_specializations = set()
_specialization_lock = threading.Lock()


def _tensor_signature(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _specialization_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    sanitized_block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    linear_v: bool,
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
):
    return (
        _tensor_signature(q),
        _tensor_signature(k),
        _tensor_signature(v),
        _tensor_signature(out),
        tuple(block_table.shape),
        tuple(sanitized_block_table.shape),
        max_seqlen_q,
        max_seqlen_k,
        linear_v,
        (
            _tensor_signature(q_descale),
            _tensor_signature(k_descale),
            _tensor_signature(v_descale),
        ),
    )


def _public_specialization_key(key):
    return key[0][2], key[3][2], key[9][0] is not None


def graph_prefill_is_ready(
    dtype: torch.dtype, output_dtype: torch.dtype, has_descale: bool
) -> bool:
    public_key = dtype, output_dtype, has_descale
    with _specialization_lock:
        return any(
            _public_specialization_key(key) == public_key
            for key in _ready_specializations
        )


def graph_prefill_has_failed(
    dtype: torch.dtype, output_dtype: torch.dtype, has_descale: bool
) -> bool:
    public_key = dtype, output_dtype, has_descale
    with _specialization_lock:
        return any(
            _public_specialization_key(key) == public_key
            for key in _failed_specializations
        )


def graph_prefill_is_ready_for(output_dtype: torch.dtype, has_descale: bool) -> bool:
    with _specialization_lock:
        return any(
            ready_output_dtype == output_dtype and ready_has_descale == has_descale
            for _, ready_output_dtype, ready_has_descale in map(
                _public_specialization_key, _ready_specializations
            )
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
    scratch_block_table: Optional[torch.Tensor] = None,
    linear_v: bool = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    k_scratch: Optional[torch.Tensor] = None,
    v_scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if linear_v:
        if scratch_block_table is None or k_scratch is None or v_scratch is None:
            raise ValueError(
                "linear-V graph prefill requires a scratch block table and K/V scratch tensors"
            )
        if scratch_block_table.data_ptr() == sanitized_block_table.data_ptr():
            raise ValueError(
                "linear-V scratch block table must not alias the sanitized live-page table"
            )
        if k_scratch.data_ptr() == k.data_ptr() or v_scratch.data_ptr() == v.data_ptr():
            raise ValueError("linear-V K/V scratch must not alias the live cache")
    elif scratch_block_table is None:
        scratch_block_table = sanitized_block_table

    key = _specialization_key(
        q,
        k,
        v,
        out,
        block_table,
        sanitized_block_table,
        max_seqlen_q,
        max_seqlen_k,
        linear_v,
        q_descale,
        k_descale,
        v_descale,
    )

    def invoke() -> torch.Tensor:
        return _mha_batch_prefill_graph(
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
            scratch_block_table,
            linear_v,
            q_descale,
            k_descale,
            v_descale,
            k_scratch,
            v_scratch,
        )

    with _specialization_lock:
        if key in _failed_specializations:
            raise RuntimeError(f"AIter graph prefill specialization is disabled: {key}")
        if key not in _ready_specializations:
            try:
                result = invoke()
            except Exception:
                _failed_specializations.add(key)
                raise
            _ready_specializations.add(key)
            return result

    return invoke()
