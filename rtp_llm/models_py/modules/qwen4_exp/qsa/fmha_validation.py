"""fmha_validation; validation/layout planning does not own mutation state."""

import torch

from .validation import DeviceChecks


def main_geometry_checks(checks, bases, lengths, table, page_size, pages, padded):
    """Queue safe wide-integer checks; the caller owns the single host read."""
    bases = bases.to(device=table.device, dtype=torch.int64, non_blocking=True)
    lengths = lengths.to(device=table.device, dtype=torch.int64, non_blocking=True)
    capacity = int(table.shape[1]) * page_size
    bounded_bases = bases.clamp(0, capacity)
    exceeds = (bases > capacity) | (lengths > capacity - bounded_bases)
    ends = bounded_bases + lengths.clamp(0, capacity + 1)
    required = torch.div(ends, page_size, rounding_mode="floor") + (
        ends % page_size != 0
    )
    columns = torch.arange(table.shape[1], device=table.device)[None]
    invalid = (columns < required[:, None]) & ((table <= 0) | (table >= pages))
    if padded:
        active = table.ne(0).any(dim=1)
        exceeds = exceeds & active
        invalid = invalid & active[:, None]
    checks.add(
        (bases < 0).any() | (lengths <= 0).any(),
        "qwen4_exp QSA paged sequence bases must be non-negative and query lengths positive",
    )
    checks.add(
        exceeds.any(),
        "qwen4_exp QSA paged main block table does not cover the visible KV",
    )
    checks.add(
        invalid.any(),
        "qwen4_exp QSA paged visible KV resolves to an unallocated or out-of-range physical block",
    )


def _validate_main_cache_layout(self, cache, block_table, batch_size, device):
    """Validate shared paged-cache layout without reading device tensor values."""
    page_size = int(self.attn_configs.kernel_tokens_per_block)
    if page_size <= 0:
        raise RuntimeError(
            f"qwen4_exp sparse GQA has invalid kernel page size {page_size}"
        )
    if (
        page_size & (page_size - 1)
        or self.head_dim <= 0
        or self.head_dim & (self.head_dim - 1)
    ):
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

    if int(block_table.shape[1]) == 0:
        raise RuntimeError(
            "qwen4_exp sparse GQA block table must contain a logical block"
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

    return page_size, cache_blocks


def _try_validate_main_paged_geometry(
    self,
    cache,
    block_table,
    sequence_bases,
    query_lengths,
    device,
    checks=None,
):
    batch_size = int(sequence_bases.numel())
    if not (
        isinstance(cache, torch.Tensor)
        and cache.is_cuda
        and cache.ndim in (2, 5)
        and cache.shape[0] <= 2**31 - 1
        and isinstance(block_table, torch.Tensor)
        and block_table.ndim == 2
        and block_table.shape[1] <= 1024
        and sequence_bases.dtype in (torch.int32, torch.int64)
        and query_lengths.dtype in (torch.int32, torch.int64)
        and batch_size <= 256
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) == (12, 0)
    ):
        return None
    page_size, cache_blocks = _validate_main_cache_layout(
        self, cache, block_table, batch_size, device
    )
    from .paged_validation import geometry_flags, validate_geometry

    # Preserve int64 metadata until validation; narrowing first can hide overflow.
    bases = sequence_bases.to(device=device, non_blocking=True)
    lengths = query_lengths.to(device=device, non_blocking=True)
    args = (
        lengths,
        bases,
        block_table,
        page_size,
        cache_blocks,
        bool(self.attn_inputs.is_s_padded),
    )
    if checks is None:
        validate_geometry(*args)
    else:
        checks.add_flags(
            geometry_flags(*args),
            [
                "qwen4_exp QSA paged sequence bases must be non-negative and query lengths positive",
                None,
                None,
                None,
                "qwen4_exp QSA paged main block table does not cover the visible KV",
                "qwen4_exp QSA paged visible KV resolves to an unallocated or out-of-range physical block",
            ],
        )
    return page_size


def _validate_main_paged_geometry(
    self,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_bases: torch.Tensor,
    query_lengths: torch.Tensor,
    device: torch.device,
    graph_capture: bool = False,
    checks=None,
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
    if graph_capture:
        if sequence_bases.device.type != "cpu" or query_lengths.device.type != "cpu":
            raise RuntimeError("qwen4_exp sparse GQA Graph requires host lengths")
        bases, lengths = sequence_bases.tolist(), query_lengths.tolist()
        if any(p < 0 or n <= 0 for p, n in zip(bases, lengths)):
            raise RuntimeError(
                "qwen4_exp QSA paged sequence bases must be non-negative and "
                "query lengths positive"
            )
        page_size, _ = _validate_main_cache_layout(
            self, cache, block_table, batch_size, device
        )
        if any(
            p + n > int(block_table.shape[1]) * page_size
            for p, n in zip(bases, lengths)
        ):
            raise RuntimeError(
                "qwen4_exp QSA paged main block table does not cover the visible KV"
            )
        return page_size
    for receipt in getattr(self, "_model_prefill_geometry", None) or ():
        if receipt.matches(self, cache, block_table, sequence_bases, query_lengths):
            return receipt.page_size
    page_size = _try_validate_main_paged_geometry(
        self,
        cache,
        block_table,
        sequence_bases,
        query_lengths,
        device,
        checks,
    )
    if page_size is not None:
        return page_size
    page_size, cache_blocks = _validate_main_cache_layout(
        self, cache, block_table, batch_size, device
    )
    own_checks = checks is None
    checks = DeviceChecks() if checks is None else checks
    main_geometry_checks(
        checks,
        sequence_bases,
        query_lengths,
        block_table,
        page_size,
        cache_blocks,
        bool(self.attn_inputs.is_s_padded),
    )
    if own_checks:
        checks.verify()
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
    from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
        Qwen4ExpQSARuntimeContext,
    )

    combined = not self.attn_inputs.is_cuda_graph and isinstance(
        qsa_runtime, Qwen4ExpQSARuntimeContext
    )
    if not combined:
        qsa_runtime.validate_before_projection(
            indexer=indexer,
            token_count=int(hidden_states.shape[0]),
            device=hidden_states.device,
        )
    is_incremental_prefill = self._is_prefix_reuse_prefill()
    if self.is_prefill and not self.is_target_verify and not is_incremental_prefill:
        if combined:
            qsa_runtime.validate_before_projection(
                indexer=indexer,
                token_count=int(hidden_states.shape[0]),
                device=hidden_states.device,
            )
        return
    if qsa_runtime.main_cache is None:
        raise RuntimeError("qwen4_exp QSA paged attention requires the main KV cache")
    if qsa_runtime.main_inputs is not self.attn_inputs:
        raise RuntimeError("qwen4_exp QSA paged main attention inputs are inconsistent")
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
    graph_capture = bool(
        self.attn_inputs.is_cuda_graph
        and (not self.is_prefill or self.is_target_verify)
    )
    checks = None
    if combined:
        # The model's side-writer preflight combines these read-only flags with
        # its position/CU and pool checks, before the first cache mutation.
        checks = DeviceChecks()
        object.__setattr__(qsa_runtime, "_prewrite_checks", checks)
    token_mismatch = query_lengths.sum() != int(hidden_states.shape[0])
    if checks is not None and token_mismatch.is_cuda:
        checks.add(
            token_mismatch,
            "qwen4_exp QSA paged token count does not match batch/query geometry",
        )
    elif bool(token_mismatch.item()):
        raise RuntimeError(
            "qwen4_exp QSA paged token count does not match batch/query geometry"
        )
    kwargs = {"checks": checks} if checks is not None else {}
    page_size = self._validate_main_paged_geometry(
        qsa_runtime.main_cache.kv_cache_base,
        self.attn_inputs.kv_cache_kernel_block_id_device,
        sequence_bases,
        query_lengths,
        hidden_states.device,
        graph_capture=graph_capture,
        **kwargs,
    )
    if not graph_capture and not is_incremental_prefill:
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.prefill_preflight import (
            MainGeometryReceipt,
        )

        self._qsa_main_geometry = (
            MainGeometryReceipt(
                self,
                qsa_runtime.main_cache.kv_cache_base,
                self.attn_inputs.kv_cache_kernel_block_id_device,
                sequence_bases,
                query_lengths,
                page_size,
            ),
            checks,
            int(hidden_states.shape[0]),
        )
    if combined:
        qsa_runtime.validate_before_projection(
            indexer=indexer,
            token_count=int(hidden_states.shape[0]),
            device=hidden_states.device,
        )

    # Reuse this invocation's already validated side-plan producer in the main
    # Graph reader. It stays layer-local, is consumed once and is reset by the
    # main-cache transaction owner; no position or history survives a call.
    plan = getattr(qsa_runtime, "_target_verify_plan", None)
    if self.is_target_verify and self.attn_inputs.is_cuda_graph and plan is not None:
        self._qsa_target_visible_lengths = (
            self.attn_inputs,
            int(hidden_states.shape[0]),
            plan.visible_lengths,
        )


def consume_main_geometry(self, cache, table, bases, tokens):
    """Reuse only a successfully accepted prewrite check, once in this call."""
    pending = getattr(self, "_qsa_main_geometry", None)
    self._qsa_main_geometry = None
    if pending is None:
        return False
    receipt, checks, checked_tokens = pending
    lengths = (
        self.attn_inputs.input_lengths if self.is_target_verify else receipt.owners[-1]
    )
    return (
        checked_tokens == tokens
        and (checks is None or checks.verified)
        and receipt.matches(self, cache, table, bases, lengths)
    )
