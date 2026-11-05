"""fmha_inputs; validation/layout planning does not own mutation state."""

from typing import Optional

import torch

from rtp_llm.ops.compute_ops import LayerKVCache

from .fmha_plans import (
    IncrementalPrefillInputs,
    PagedInputs,
    PrefillRequest,
    _PrefillRequests,
)
from .fmha_validation import _validate_main_cache_layout


def _prepare_paged_inputs(
    self,
    qkv: torch.Tensor,
    kv_cache: Optional[LayerKVCache],
    selected_indices: torch.Tensor,
) -> PagedInputs:
    """Validate every paged-reader invariant before the main KV writer runs."""
    if qkv.dim() != 2:
        raise ValueError(
            f"qwen4_exp sparse GQA qkv must be rank 2, got {tuple(qkv.shape)}"
        )
    expected_qkv_width = self.q_width + 2 * self.kv_width
    if int(qkv.shape[1]) != expected_qkv_width:
        raise ValueError(
            "qwen4_exp sparse GQA qkv width must be "
            f"{expected_qkv_width}, got {qkv.shape[1]}"
        )
    if qkv.dtype != torch.bfloat16 or not qkv.is_cuda:
        raise ValueError("qwen4_exp sparse paged GQA requires BF16 CUDA qkv")
    if kv_cache is None or kv_cache.kv_cache_base is None:
        raise RuntimeError("qwen4_exp sparse GQA decode requires the main KV cache")

    if self.is_target_verify:
        lengths_source = self.attn_inputs.prefix_lengths
    else:
        lengths_source = self.attn_inputs.sequence_lengths
    if lengths_source.dim() != 1:
        raise ValueError("qwen4_exp sparse GQA sequence metadata must be rank 1")
    batch = int(lengths_source.numel())
    tokens = int(qkv.shape[0])
    if batch <= 0 or tokens % batch:
        raise ValueError(
            f"qwen4_exp sparse GQA decode tokens={tokens} do not divide batch={batch}"
        )
    query_len = tokens // batch

    if selected_indices.dim() == 2:
        if int(selected_indices.shape[0]) != tokens:
            raise ValueError(
                "packed decode selected_indices must have one row per token"
            )
        selected_indices = selected_indices.view(
            batch, query_len, int(selected_indices.shape[1])
        )
    if selected_indices.dim() != 3 or tuple(selected_indices.shape[:2]) != (
        batch,
        query_len,
    ):
        raise ValueError(
            "decode selected_indices must be [B, query_len, K], got "
            f"{tuple(selected_indices.shape)}"
        )
    if selected_indices.dtype != torch.int32:
        raise ValueError("decode selected_indices must be int32")
    if selected_indices.device != qkv.device:
        raise ValueError("decode selected_indices and qkv must share one CUDA device")

    if self.is_target_verify:
        input_lengths = self.attn_inputs.input_lengths
        if input_lengths.numel() != batch or bool(
            torch.any(input_lengths != query_len).item()
        ):
            raise ValueError(
                "qwen4_exp target-verify input lengths must match query_len"
            )
    graph_paged = bool(
        self.attn_inputs.is_cuda_graph
        and (not self.is_prefill or self.is_target_verify)
    )
    if graph_paged and lengths_source.device.type != "cpu":
        raise RuntimeError("qwen4_exp sparse GQA Graph requires host lengths")
    if graph_paged and bool(torch.any(lengths_source < 0).item()):
        raise ValueError("qwen4_exp sparse GQA sequence lengths must be non-negative")
    cached = getattr(self, "_qsa_target_visible_lengths", None)
    self._qsa_target_visible_lengths = None
    if graph_paged and self.is_target_verify and cached is not None:
        owner_inputs, owner_tokens, kv_lens = cached
        if not (
            owner_inputs is self.attn_inputs
            and owner_tokens == tokens
            and isinstance(kv_lens, torch.Tensor)
            and kv_lens.device == qkv.device
            and kv_lens.dtype == torch.int32
            and tuple(kv_lens.shape) == (batch, query_len)
            and kv_lens.is_contiguous()
        ):
            raise RuntimeError("qwen4_exp main target metadata owner/geometry changed")
    else:
        device_source = lengths_source
        if graph_paged:
            device_source = (
                self.attn_inputs.prefix_lengths_device
                if self.is_target_verify
                else self.attn_inputs.sequence_lengths_plus_1_device - 1
            )
        sequence_lengths = device_source.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        query_offsets = torch.arange(
            1, query_len + 1, device=qkv.device, dtype=torch.int32
        )
        kv_lens = sequence_lengths.view(batch, 1) + query_offsets.view(1, query_len)

    block_table = self.attn_inputs.kv_cache_kernel_block_id_device
    page_size, cache_blocks = _validate_main_cache_layout(
        self, kv_cache.kv_cache_base, block_table, batch, qkv.device
    )
    if self.attn_inputs.is_s_padded:
        kv_lens = torch.where(block_table.ne(0).any(dim=1, keepdim=True), kv_lens, 0)

    if graph_paged:
        if bool(
            torch.any(
                (lengths_source + query_len + page_size - 1) // page_size
                > int(block_table.shape[1])
            ).item()
        ):
            raise RuntimeError(
                "qwen4_exp sparse GQA main block table does not cover the visible KV"
            )
        return PagedInputs(
            **{
                "batch": batch,
                "query_len": query_len,
                "tokens": tokens,
                "selected_indices": selected_indices,
                "kv_lens": kv_lens,
                "block_table": block_table,
                "page_size": page_size,
            }
        )
    required_columns = (kv_lens[:, -1].to(torch.int64) + page_size - 1) // page_size
    columns = torch.arange(
        int(block_table.shape[1]), device=block_table.device
    ).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
    selected_indices = selected_indices.contiguous()
    # One host verdict for the complete plan, before the main cache writer.
    # The private FMHA consumer reuses this exact plan in the same invocation.
    verdict = (
        ((sequence_lengths < 0).any() | (kv_lens < 0).any()).to(torch.int32)
        | ((required_columns > int(block_table.shape[1])).any().to(torch.int32) << 1)
        | (invalid.any().to(torch.int32) << 2)
        | ((block_table >= cache_blocks).any().to(torch.int32) << 3)
        | ((selected_indices < -1).any().to(torch.int32) << 4)
        | (
            ((selected_indices >= 0) & (selected_indices >= kv_lens.unsqueeze(-1)))
            .any()
            .to(torch.int32)
            << 5
        )
    ).item()
    if verdict & 1:
        raise ValueError("qwen4_exp sparse GQA sequence lengths must be non-negative")
    if verdict & 2:
        raise RuntimeError(
            "qwen4_exp sparse GQA main block table does not cover the visible KV"
        )
    if verdict & 4:
        raise RuntimeError(
            "qwen4_exp sparse GQA visible KV resolves to an unallocated or "
            "out-of-range physical block"
        )

    if verdict & 8:
        raise RuntimeError(
            "qwen4_exp QSA paged main block table contains an out-of-range physical block"
        )
    if verdict & 16:
        raise ValueError("selected indices may only use -1 as padding")
    if verdict & 32:
        raise ValueError(
            "selected indices contain a token outside the row's visible KV"
        )

    return PagedInputs(
        **{
            "batch": batch,
            "query_len": query_len,
            "tokens": tokens,
            "selected_indices": selected_indices,
            "kv_lens": kv_lens,
            "block_table": block_table,
            "page_size": page_size,
            "validated_cache_blocks": cache_blocks,
        }
    )


def _prepare_incremental_prefill_inputs(
    self,
    qkv: torch.Tensor,
    kv_cache: Optional[LayerKVCache],
    selected_indices: torch.Tensor,
) -> IncrementalPrefillInputs:
    """Build ragged per-request paged plans before the production writer."""
    prefixes = self.attn_inputs.prefix_lengths
    if (
        not self.is_prefill
        or self.is_target_verify
        or (prefixes.device.type == "cpu" and not self._is_prefix_reuse_prefill())
    ):
        raise RuntimeError("qwen4_exp incremental prefill requires an explicit prefix")
    if qkv.dim() != 2:
        raise ValueError(
            f"qwen4_exp sparse GQA qkv must be rank 2, got {tuple(qkv.shape)}"
        )
    expected_qkv_width = self.q_width + 2 * self.kv_width
    if int(qkv.shape[1]) != expected_qkv_width:
        raise ValueError(
            "qwen4_exp sparse GQA qkv width must be "
            f"{expected_qkv_width}, got {qkv.shape[1]}"
        )
    if qkv.dtype != torch.bfloat16 or not qkv.is_cuda:
        raise ValueError("qwen4_exp sparse incremental prefill requires BF16 CUDA qkv")
    if kv_cache is None or kv_cache.kv_cache_base is None:
        raise RuntimeError(
            "qwen4_exp sparse incremental prefill requires the main KV cache"
        )

    input_lengths = self.attn_inputs.input_lengths
    prefixes = self.attn_inputs.prefix_lengths
    if input_lengths.dim() != 1 or prefixes.dim() != 1:
        raise ValueError(
            "qwen4_exp incremental prefill lengths and prefixes must be rank 1"
        )
    batch = int(input_lengths.numel())
    if batch <= 0 or int(prefixes.numel()) != batch:
        raise ValueError("qwen4_exp incremental prefill lengths and prefixes disagree")
    plan = _try_prepare_host_incremental_prefill(
        self, qkv, kv_cache, selected_indices, input_lengths, prefixes
    )
    if plan is not None:
        return plan

    from .fmha_validation import main_geometry_checks
    from .validation import DeviceChecks

    if input_lengths.dtype not in (torch.int32, torch.int64) or prefixes.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("qwen4_exp incremental prefill lengths must be int32 or int64")
    lengths_device = input_lengths.to(
        device=qkv.device, dtype=torch.int64, non_blocking=True
    )
    prefixes_wide = prefixes.to(device=qkv.device, dtype=torch.int64, non_blocking=True)
    checks = DeviceChecks()
    checks.add(
        ~(prefixes_wide != 0).any(),
        "qwen4_exp incremental prefill requires an explicit prefix",
    )
    checks.add(
        (lengths_device <= 0).any(),
        "qwen4_exp incremental prefill input lengths must be positive",
        ValueError,
    )
    checks.add(
        (prefixes_wide < 0).any(),
        "qwen4_exp incremental prefill prefixes must be non-negative",
        ValueError,
    )
    cu_seqlens = self.attn_inputs.cu_seqlens_device
    if (
        cu_seqlens is None
        or tuple(cu_seqlens.shape) != (batch + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != qkv.device
    ):
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
        )
    expected_cu = torch.cat((lengths_device.new_zeros(1), lengths_device.cumsum(0)))
    checks.add(
        (cu_seqlens != expected_cu).any() | (lengths_device > 2**31 - 1).any(),
        "qwen4_exp incremental prefill cu_seqlens do not match input lengths",
        ValueError,
    )
    tokens = int(qkv.shape[0])
    checks.add(
        expected_cu[-1] != tokens,
        "qwen4_exp incremental prefill cu_seqlens do not partition qkv",
        ValueError,
    )
    if selected_indices.dim() != 2 or int(selected_indices.shape[0]) != tokens:
        raise ValueError(
            "qwen4_exp incremental prefill selected_indices must be packed "
            f"[{tokens}, K], got {tuple(selected_indices.shape)}"
        )
    if selected_indices.dtype != torch.int32:
        raise ValueError("qwen4_exp incremental prefill selected_indices must be int32")
    if selected_indices.device != qkv.device:
        raise ValueError(
            "qwen4_exp incremental prefill selection and qkv must share CUDA"
        )
    block_table = self.attn_inputs.kv_cache_kernel_block_id_device
    page_size, cache_blocks = _validate_main_cache_layout(
        self, kv_cache.kv_cache_base, block_table, batch, qkv.device
    )
    main_geometry_checks(
        checks,
        prefixes_wide,
        lengths_device,
        block_table,
        page_size,
        cache_blocks,
        bool(self.attn_inputs.is_s_padded),
    )
    checks.add(
        (prefixes_wide > 2**31 - 1 - lengths_device.clamp(0, 2**31)).any(),
        "qwen4_exp incremental prefill visible KV exceeds int32 range",
        ValueError,
    )
    rows = torch.arange(tokens, device=qkv.device, dtype=torch.int64)
    # Clamp even malformed partitions: validation must never perform an unsafe
    # gather before its verdict has been checked. Actual CU is compared above.
    requests_for_rows = torch.searchsorted(expected_cu[1:], rows, right=True).clamp_max(
        batch - 1
    )
    visible = (
        prefixes_wide[requests_for_rows] + rows - expected_cu[requests_for_rows] + 1
    )
    checks.add(
        (selected_indices < -1).any(),
        "selected indices may only use -1 as padding",
        ValueError,
    )
    checks.add(
        ((selected_indices >= 0) & (selected_indices >= visible[:, None])).any(),
        "incremental prefill selected indices contain a token outside the row's visible KV",
        ValueError,
    )
    # Partition data and every verdict cross the device/host boundary together.
    cu_host = checks.verify(expected_cu)
    prefixes_device = prefixes_wide.to(torch.int32)
    prefix_host = prefixes.tolist() if prefixes.device.type == "cpu" else None
    requests = []
    for request_idx in range(batch):
        start, end = cu_host[request_idx : request_idx + 2]
        requests.append(
            PrefillRequest(
                start,
                end,
                visible[None, start:end].to(torch.int32).contiguous(),
                selected_indices[None, start:end].contiguous(),
                None if prefix_host is None else int(prefix_host[request_idx]),
            )
        )
    return IncrementalPrefillInputs(
        batch,
        tokens,
        block_table,
        page_size,
        requests,
        selected_indices,
        prefixes_device,
        cu_seqlens,
    )


def _try_prepare_host_incremental_prefill(
    self, qkv, cache, selected, lengths, prefixes
):
    """Derive request offsets on CPU and validate the device partition once.

    All device contents are checked for this invocation. Nothing is cached across
    layers or calls; in-place metadata updates still take effect before KV writes.
    """
    from .fmha_validation import _validate_main_cache_layout
    from .prefill_validation import validate_prefill_inputs

    cu = self.attn_inputs.cu_seqlens_device
    tokens = int(qkv.shape[0])
    batch = int(lengths.numel())
    if not (
        lengths.device.type == prefixes.device.type == "cpu"
        and lengths.dtype in (torch.int32, torch.int64)
        and prefixes.dtype in (torch.int32, torch.int64)
        and isinstance(cu, torch.Tensor)
        and cu.device == qkv.device
        and cu.dtype == torch.int32
        and cu.shape == (batch + 1,)
        and cu.is_contiguous()
        and selected.ndim == 2
        and selected.shape[0] == tokens
        and selected.shape[1] > 0
        and selected.shape[1] <= 8192
        and selected.dtype == torch.int32
        and selected.device == qkv.device
        and max(batch, tokens) < 2**30
        and batch <= 256
    ):
        return None
    lens = lengths.tolist()
    prefix_host = prefixes.tolist()
    if any(x <= 0 for x in lens):
        raise ValueError("qwen4_exp incremental prefill input lengths must be positive")
    if any(x < 0 for x in prefix_host):
        raise ValueError("qwen4_exp incremental prefill prefixes must be non-negative")
    if sum(lens) != tokens:
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not partition qkv"
        )
    # Keep int32 visible positions and device lengths inside their supported range.
    if any(p + s >= 2**31 for p, s in zip(prefix_host, lens)):
        raise ValueError("qwen4_exp incremental prefill visible KV exceeds int32 range")
    ld = lengths.to(device=qkv.device, dtype=torch.int32, non_blocking=True)
    pd = prefixes.to(device=qkv.device, dtype=torch.int32, non_blocking=True)
    table = self.attn_inputs.kv_cache_kernel_block_id_device
    page_size, cache_blocks = _validate_main_cache_layout(
        self, cache.kv_cache_base, table, batch, qkv.device
    )
    visible = validate_prefill_inputs(
        selected,
        cu,
        ld,
        pd,
        table,
        page_size,
        cache_blocks,
        bool(self.attn_inputs.is_s_padded),
    )
    requests = (
        [
            PrefillRequest(
                0, tokens, visible[None], selected[None].contiguous(), prefix_host[0]
            )
        ]
        if batch == 1
        else _PrefillRequests(lens, prefix_host, visible, selected)
    )
    return IncrementalPrefillInputs(
        batch, tokens, table, page_size, requests, selected, pd, cu
    )
