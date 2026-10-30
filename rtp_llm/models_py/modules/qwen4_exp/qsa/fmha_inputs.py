"""fmha_inputs; validation/layout planning does not own mutation state."""

from typing import Optional

import torch
from rtp_llm.ops.compute_ops import LayerKVCache

from .fmha_plans import IncrementalPrefillInputs, PagedInputs, PrefillRequest
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
        if not graph_paged and bool(torch.any(sequence_lengths < 0).item()):
            raise ValueError(
                "qwen4_exp sparse GQA sequence lengths must be non-negative"
            )
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

    if not graph_paged:
        required_columns = (kv_lens[:, -1] + page_size - 1) // page_size
    exceeds_capacity = (
        bool(
            torch.any(
                (lengths_source + query_len + page_size - 1) // page_size
                > int(block_table.shape[1])
            ).item()
        )
        if graph_paged
        else bool(torch.any(required_columns > int(block_table.shape[1])).item())
    )
    if exceeds_capacity:
        raise RuntimeError(
            "qwen4_exp sparse GQA main block table does not cover the visible KV"
        )
    if graph_paged:
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
    columns = torch.arange(
        int(block_table.shape[1]), device=block_table.device
    ).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
    if bool(torch.any(invalid).item()):
        raise RuntimeError(
            "qwen4_exp sparse GQA visible KV resolves to an unallocated or "
            "out-of-range physical block"
        )

    if selected_indices.numel():
        if int(selected_indices.min().item()) < -1:
            raise ValueError("selected indices may only use -1 as padding")
        invalid_visible = (selected_indices >= 0) & (
            selected_indices >= kv_lens.unsqueeze(-1)
        )
        if bool(torch.any(invalid_visible).item()):
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
        }
    )


def _prepare_incremental_prefill_inputs(
    self,
    qkv: torch.Tensor,
    kv_cache: Optional[LayerKVCache],
    selected_indices: torch.Tensor,
) -> IncrementalPrefillInputs:
    """Build ragged per-request paged plans before the production writer."""
    if not self._is_prefix_reuse_prefill():
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
    input_lengths_device = input_lengths.to(
        device=qkv.device, dtype=torch.int32, non_blocking=True
    )
    prefixes_device = prefixes.to(
        device=qkv.device, dtype=torch.int32, non_blocking=True
    )
    if bool(torch.any(input_lengths_device <= 0).item()):
        raise ValueError("qwen4_exp incremental prefill input lengths must be positive")
    if bool(torch.any(prefixes_device < 0).item()):
        raise ValueError("qwen4_exp incremental prefill prefixes must be non-negative")

    cu_seqlens = self.attn_inputs.cu_seqlens_device
    expected_cu = torch.zeros(batch + 1, dtype=torch.int32, device=qkv.device)
    expected_cu[1:] = input_lengths_device.cumsum(0)
    if (
        cu_seqlens is None
        or cu_seqlens.dim() != 1
        or tuple(cu_seqlens.shape) != (batch + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != qkv.device
        or not bool(torch.equal(cu_seqlens, expected_cu))
    ):
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
        )
    tokens = int(expected_cu[-1].item())
    if int(qkv.shape[0]) != tokens:
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not partition qkv: "
            f"tokens={qkv.shape[0]}, cu_end={tokens}"
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
    page_size = self._validate_main_paged_geometry(
        kv_cache.kv_cache_base,
        block_table,
        prefixes_device,
        input_lengths_device,
        qkv.device,
    )

    cu_host = expected_cu.cpu().tolist()
    requests = []
    for request_idx in range(batch):
        start, end = cu_host[request_idx : request_idx + 2]
        query_len = end - start
        kv_lens = prefixes_device[request_idx] + torch.arange(
            1, query_len + 1, dtype=torch.int32, device=qkv.device
        )
        request_selection = selected_indices[start:end]
        if request_selection.numel():
            if int(request_selection.min().item()) < -1:
                raise ValueError("selected indices may only use -1 as padding")
            invalid_visible = (request_selection >= 0) & (
                request_selection >= kv_lens.unsqueeze(-1)
            )
            if bool(torch.any(invalid_visible).item()):
                raise ValueError(
                    "incremental prefill selected indices contain a token "
                    "outside the row's visible KV"
                )
        requests.append(
            PrefillRequest(
                start,
                end,
                kv_lens.unsqueeze(0).contiguous(),
                request_selection.unsqueeze(0).contiguous(),
            )
        )

    return IncrementalPrefillInputs(
        **{
            "batch": batch,
            "tokens": tokens,
            "block_table": block_table,
            "page_size": page_size,
            "requests": requests,
            "selected_indices": selected_indices,
            "prefixes_device": prefixes_device,
        }
    )
