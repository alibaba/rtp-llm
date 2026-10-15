"""fmha_validation; validation/layout planning does not own mutation state."""

import torch


def _validate_main_paged_geometry(
    self,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_bases: torch.Tensor,
    query_lengths: torch.Tensor,
    device: torch.device,
    graph_capture: bool = False,
) -> int:
    """Validate all pages that a paged prefill/decode may observe."""
    batch_size = int(sequence_bases.numel())
    if (
        sequence_bases.dim() != 1
        or query_lengths.dim() != 1
        or int(query_lengths.numel()) != batch_size
        or batch_size <= 0
    ):
        raise RuntimeError("qwen4_exp QSA paged sequence metadata has invalid geometry")
    if graph_capture and (
        sequence_bases.device.type != "cpu" or query_lengths.device.type != "cpu"
    ):
        raise RuntimeError("qwen4_exp sparse GQA Graph requires host lengths")
    invalid_lengths = (
        bool(torch.any(sequence_bases < 0).item())
        or bool(torch.any(query_lengths <= 0).item())
        if graph_capture
        else False
    )
    if not graph_capture:
        sequence_bases = sequence_bases.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        query_lengths = query_lengths.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        invalid_lengths = bool(torch.any(sequence_bases < 0).item()) or bool(
            torch.any(query_lengths <= 0).item()
        )
    if invalid_lengths:
        raise RuntimeError(
            "qwen4_exp QSA paged sequence bases must be non-negative and "
            "query lengths positive"
        )

    page_size = int(self.attn_configs.kernel_tokens_per_block)
    if page_size <= 0:
        raise RuntimeError(
            f"qwen4_exp sparse GQA has invalid kernel page size {page_size}"
        )
    if page_size & (page_size - 1) or self.head_dim & (self.head_dim - 1):
        raise RuntimeError(
            "qwen4_exp QSA paged attention requires power-of-two page and head sizes"
        )
    if (
        block_table is None
        or block_table.dim() != 2
        or int(block_table.shape[0]) != batch_size
        or block_table.dtype != torch.int32
        or block_table.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA paged attention requires a device int32 main block table"
        )

    if cache is None or cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise RuntimeError(
            "qwen4_exp QSA paged attention requires a BF16 or FP8 E4M3 main KV cache"
        )
    if cache.device != device:
        raise RuntimeError(
            "qwen4_exp QSA paged main cache and activations must share a device"
        )
    if cache.dim() == 2:
        required_width = 2 * self.kv_head_num * page_size * self.head_dim
        if int(cache.shape[1]) < required_width or cache.stride(1) != 1:
            raise RuntimeError(
                "qwen4_exp QSA paged packed main-cache geometry is invalid"
            )
        cache_blocks = int(cache.shape[0])
    elif cache.dim() == 5:
        if tuple(cache.shape[1:]) != (
            2,
            self.kv_head_num,
            page_size,
            self.head_dim,
        ):
            raise RuntimeError("qwen4_exp QSA paged main-cache geometry is invalid")
        expected_inner_strides = (
            self.kv_head_num * page_size * self.head_dim,
            page_size * self.head_dim,
            self.head_dim,
            1,
        )
        if tuple(cache.stride()[1:]) != expected_inner_strides:
            raise RuntimeError(
                "qwen4_exp QSA paged main cache must use HND inner layout"
            )
        cache_blocks = int(cache.shape[0])
    else:
        raise RuntimeError(
            "qwen4_exp QSA paged main cache must be packed 2-D or HND 5-D"
        )

    if (
        not graph_capture
        and block_table.numel()
        and int(block_table.max().item()) >= cache_blocks
    ):
        raise RuntimeError(
            "qwen4_exp QSA paged main block table contains an "
            "out-of-range physical block"
        )
    visible_ends = sequence_bases + query_lengths
    required_columns = (visible_ends + page_size - 1) // page_size
    if bool(torch.any(required_columns > int(block_table.shape[1])).item()):
        raise RuntimeError(
            "qwen4_exp QSA paged main block table does not cover the visible KV"
        )
    if graph_capture:
        return page_size
    columns = torch.arange(
        int(block_table.shape[1]), device=block_table.device
    ).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
    if bool(torch.any(invalid).item()):
        raise RuntimeError(
            "qwen4_exp QSA paged visible KV resolves to an unallocated "
            "or out-of-range physical block"
        )
    return page_size


def validate_qsa_before_side_write(
    self,
    qsa_runtime,
    indexer,
    hidden_states: torch.Tensor,
) -> None:
    """Validate side and main paged geometry before either cache is written."""
    if self._selected_indices is not None:
        raise RuntimeError("qwen4_exp sparse GQA has an unconsumed selection")
    self._paged_kv_scale_args(qsa_runtime.main_cache)
    if self._is_draft_graph():
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
            draft_graph_lanes,
        )

        if not qsa_runtime.is_mtp_draft:
            raise RuntimeError(
                "qwen4_exp draft Graph requires explicit fixed-capacity MTP inputs"
            )
        draft_graph_lanes(self.attn_inputs, int(hidden_states.shape[0]))
        return
    qsa_runtime.validate_before_projection(
        indexer=indexer,
        token_count=int(hidden_states.shape[0]),
        device=hidden_states.device,
    )
    is_incremental_prefill = self._is_prefix_reuse_prefill()
    if self.is_prefill and not self.is_target_verify and not is_incremental_prefill:
        return
    if qsa_runtime.main_cache is None:
        raise RuntimeError("qwen4_exp QSA paged attention requires the main KV cache")
    if qsa_runtime.main_inputs is not self.attn_inputs:

        def _inputs_brief(inputs) -> str:
            try:
                prefixes = inputs.prefix_lengths
                prefix_head = prefixes[:4].tolist() if prefixes is not None else None
            except Exception as error:  # pragma: no cover - diagnostics only
                prefix_head = f"unavailable({error!r})"
            return (
                f"id={id(inputs)}"
                f" is_prefill={getattr(inputs, 'is_prefill', None)}"
                f" is_target_verify={getattr(inputs, 'is_target_verify', None)}"
                f" prefix_head={prefix_head}"
            )

        raise RuntimeError(
            "qwen4_exp QSA paged main attention inputs are inconsistent"
            f" [runtime {_inputs_brief(qsa_runtime.main_inputs)} |"
            f" impl {_inputs_brief(self.attn_inputs)} |"
            f" impl_flags prefill={self.is_prefill}"
            f" verify={self.is_target_verify}"
            f" mtp_draft={self._is_mtp_draft}"
            f" incremental={is_incremental_prefill}]"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "qwen4_exp QSA paged attention currently requires BF16 hidden states"
        )
    if not hidden_states.is_cuda:
        raise RuntimeError("qwen4_exp QSA paged attention requires CUDA tensors")

    if self.is_target_verify or is_incremental_prefill:
        sequence_bases = self.attn_inputs.prefix_lengths
        query_lengths = self.attn_inputs.input_lengths
    else:
        sequence_bases = self.attn_inputs.sequence_lengths
        query_lengths = torch.ones_like(sequence_bases)
    if int(hidden_states.shape[0]) != int(query_lengths.sum().item()):
        raise RuntimeError(
            "qwen4_exp QSA paged token count does not match batch/query geometry"
        )
    self._validate_main_paged_geometry(
        qsa_runtime.main_cache.kv_cache_base,
        self.attn_inputs.kv_cache_kernel_block_id_device,
        sequence_bases,
        query_lengths,
        hidden_states.device,
        graph_capture=bool(
            self.attn_inputs.is_cuda_graph
            and (not self.is_prefill or self.is_target_verify)
        ),
    )
