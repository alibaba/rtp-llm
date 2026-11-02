"""Page-by-page PLE oracle, outside the production execution path."""

from typing import Any

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import (
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    select_attention_inputs_for_tag,
)


def _apply_ple(
    self,
    layer_idx: int,
    hyper_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_inputs: Any,
    prefetch=None,
) -> torch.Tensor:
    ple = self.ple_layers[str(layer_idx)]
    state_inputs = select_attention_inputs_for_tag(attention_inputs, PLE_STATE_TAG)
    ctx_inputs = select_attention_inputs_for_tag(attention_inputs, PLE_NGRAM_CTX_TAG)
    self._validate_ple_mode(state_inputs, allow_target_verify=True)
    self._validate_ple_mode(ctx_inputs, allow_target_verify=True)
    is_target_verify = bool(state_inputs.is_target_verify)
    if (
        is_target_verify != bool(ctx_inputs.is_target_verify)
        or bool(state_inputs.is_prefill) != bool(ctx_inputs.is_prefill)
        or any(
            not self._same_metadata(state_inputs, ctx_inputs, name)
            for name in (
                "input_lengths",
                "prefix_lengths",
                "sequence_lengths",
                "cu_seqlens_device",
                "cu_kv_seqlens_device",
                "combo_position_ids",
            )
        )
    ):
        raise RuntimeError("qwen4_exp PLE state-region metadata is inconsistent")
    state_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_STATE_TAG)
    ctx_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_NGRAM_CTX_TAG)
    if state_cache.seq_size_per_block != ctx_cache.seq_size_per_block:
        raise RuntimeError("qwen4_exp PLE state-region page sizes are inconsistent")
    if hyper_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "qwen4_exp PLE currently requires BF16 activations, "
            f"got {hyper_states.dtype}"
        )
    state_pool = self._fixed_state_pool(
        state_cache,
        torch.bfloat16,
        ple.short_conv_state_len,
        ple.hc_hidden_size,
    )
    ctx_pool = self._fixed_state_pool(
        ctx_cache, torch.int64, ple.ple_embedding.context_len, 1
    ).squeeze(-1)
    ids = input_ids.reshape(-1).to(device=hyper_states.device, dtype=torch.long)
    if hyper_states.dim() != 2 or int(hyper_states.shape[0]) != int(ids.numel()):
        raise RuntimeError("qwen4_exp PLE expects packed [tokens, hc_hidden] input")
    eos = ple.ple_embedding.eos_token_id
    context_len = ple.ple_embedding.context_len
    page_size = int(state_cache.seq_size_per_block)
    if page_size <= 0:
        raise RuntimeError("qwen4_exp PLE cache page size must be positive")

    graph_capture = bool(state_inputs.is_cuda_graph)
    # Eager fallback must never reuse rows prepared for a captured graph.
    ple.ple_embedding._graph_gather_required = graph_capture
    local_rows = (
        [
            (ple.ple_embedding.consume_prefetch(handle) if handle is not None else None)
            for handle in prefetch
        ]
        if prefetch is not None
        else None
    )
    if is_target_verify:
        prefetched = (
            {"embeddings": ple.ple_embedding._reduce_rows(local_rows[0])}
            if local_rows is not None and local_rows[0] is not None
            else {}
        )
        return self._stage_ple_target_verify(
            layer_idx=layer_idx,
            ple=ple,
            hyper_states=hyper_states,
            ids=ids,
            state_inputs=state_inputs,
            ctx_inputs=ctx_inputs,
            state_pool=state_pool,
            ctx_pool=ctx_pool,
            page_size=page_size,
            **prefetched,
        )

    if state_inputs.is_prefill:
        lengths = [int(x) for x in state_inputs.input_lengths.tolist()]
        if any(length <= 0 for length in lengths):
            raise RuntimeError("qwen4_exp PLE requires positive prefill lengths")
        if sum(lengths) != int(ids.numel()):
            raise RuntimeError("qwen4_exp PLE packed prefill lengths are inconsistent")
        raw_prefixes = getattr(state_inputs, "prefix_lengths", None)
        prefixes = (
            [int(x) for x in raw_prefixes.tolist()]
            if raw_prefixes is not None and int(raw_prefixes.numel()) > 0
            else [0] * len(lengths)
        )
        if len(prefixes) != len(lengths) or any(p < 0 for p in prefixes):
            raise RuntimeError("qwen4_exp PLE prefill prefix lengths are inconsistent")
        # Requests carry per-request new-token lengths; prefix reuse resumes
        # the side state from the page holding the last prefix token, so
        # prefixes must land on page boundaries.
        if any(prefix % page_size for prefix in prefixes):
            raise RuntimeError(
                "qwen4_exp PLE prefix reuse requires page-aligned prefixes"
            )
        read_buffers = None
        read_contexts = None
        if any(prefixes):
            read_pages = torch.tensor(
                [(prefix - 1) // page_size if prefix else 0 for prefix in prefixes],
                dtype=torch.long,
            )
            read_buffers = state_pool.index_select(
                0,
                self._physical_blocks(
                    state_inputs,
                    PLE_STATE_TAG,
                    read_pages,
                    int(state_pool.shape[0]),
                    state_pool.device,
                ),
            )
            read_contexts = ctx_pool.index_select(
                0,
                self._physical_blocks(
                    ctx_inputs,
                    PLE_NGRAM_CTX_TAG,
                    read_pages,
                    int(ctx_pool.shape[0]),
                    ctx_pool.device,
                ),
            )
        # Checkpoint every page, not just the request's terminal page: a
        # later request can reuse a prefix that ends at any page boundary,
        # exactly like the GDN state snapshots written by
        # store_ssm_state_to_block_map.
        outputs = []
        write_rows = []
        write_pages = []
        write_states = []
        write_contexts = []
        offset = 0
        for idx, (prefix, length) in enumerate(zip(prefixes, lengths)):
            if prefix > 0:
                state = read_buffers[idx : idx + 1]
                context = read_contexts[idx : idx + 1]
            else:
                state = None
                context = ids.new_full((1, context_len), eos)
            chunk_outputs = []
            start = 0
            while start < length:
                end = min(start + page_size, length)
                chunk_ids = ids[offset + start : offset + end].view(1, end - start)
                history = torch.cat([context, chunk_ids], dim=1)
                chunk_hyper = hyper_states[offset + start : offset + end].view(
                    1, end - start, -1
                )
                if state is None:
                    output, state = ple.prefill(
                        chunk_hyper,
                        history,
                        **(
                            {
                                "embeddings": ple.ple_embedding._reduce_rows(
                                    local_rows[idx][:, start:end].contiguous()
                                )
                            }
                            if local_rows is not None and local_rows[idx] is not None
                            else {}
                        ),
                    )
                else:
                    output, state = ple.prefill_with_state(
                        chunk_hyper,
                        history,
                        state,
                        **(
                            {
                                "embeddings": ple.ple_embedding._reduce_rows(
                                    local_rows[idx][:, start:end].contiguous()
                                )
                            }
                            if local_rows is not None and local_rows[idx] is not None
                            else {}
                        ),
                    )
                chunk_outputs.append(output.squeeze(0))
                write_rows.append(idx)
                write_pages.append((prefix + end - 1) // page_size)
                write_states.append(state.squeeze(0))
                write_contexts.append(history[0, -context_len:])
                context = history[:, -context_len:]
                start = end
            outputs.append(torch.cat(chunk_outputs, dim=0))
            offset += length
        write_rows_t = torch.tensor(write_rows, dtype=torch.long)
        write_pages_t = torch.tensor(write_pages, dtype=torch.long)
        write_state_blocks, state_writable = self._physical_blocks_for_pairs(
            state_inputs,
            PLE_STATE_TAG,
            write_rows_t,
            write_pages_t,
            int(state_pool.shape[0]),
            state_pool.device,
        )
        write_ctx_blocks, ctx_writable = self._physical_blocks_for_pairs(
            ctx_inputs,
            PLE_NGRAM_CTX_TAG,
            write_rows_t,
            write_pages_t,
            int(ctx_pool.shape[0]),
            ctx_pool.device,
        )
        # Decode resumes from the page holding each request's last token, so
        # that page must exist even when the allocator dropped the rest of
        # the checkpoint chain.
        terminal_indices = []
        placed = 0
        for length in lengths:
            placed += (length + page_size - 1) // page_size
            terminal_indices.append(placed - 1)
        terminal_t = torch.tensor(
            terminal_indices, dtype=torch.long, device=state_writable.device
        )
        if not bool(
            (state_writable[terminal_t] & ctx_writable[terminal_t]).all().item()
        ):
            raise RuntimeError(
                "qwen4_exp PLE terminal page is unallocated: decode resumes "
                "from its checkpoint"
            )
        writable = state_writable & ctx_writable
        state_pool.index_copy_(
            0,
            write_state_blocks[writable],
            torch.stack(write_states)[writable].to(state_pool.dtype),
        )
        ctx_pool.index_copy_(
            0,
            write_ctx_blocks[writable],
            torch.stack(write_contexts)[writable].to(ctx_pool.dtype),
        )
        return hyper_states + torch.cat(outputs, dim=0)

    lengths = state_inputs.input_lengths
    sequence_lengths = state_inputs.sequence_lengths
    batch = int(sequence_lengths.numel())
    if batch == 0 or int(ids.numel()) != batch or int(lengths.numel()) != batch:
        raise RuntimeError("qwen4_exp PLE supports single-token decode only")
    if sequence_lengths.dim() != 1 or sequence_lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError(
            "qwen4_exp PLE decode sequence_lengths must be a 1-D integer tensor"
        )
    if graph_capture:
        sequence_plus_one_device = getattr(
            state_inputs, "sequence_lengths_plus_1_device", None
        )
        if (
            sequence_plus_one_device is None
            or sequence_plus_one_device.device != state_pool.device
            or sequence_plus_one_device.dtype != torch.int32
            or int(sequence_plus_one_device.numel()) != batch
        ):
            raise RuntimeError(
                "qwen4_exp PLE decode Graph requires device sequence lengths"
            )
        sequence_lengths = sequence_plus_one_device.to(torch.long) - 1
    else:
        sequence_lengths = sequence_lengths.to(dtype=torch.long)
    if not graph_capture and bool((sequence_lengths <= 0).any().item()):
        raise RuntimeError("qwen4_exp PLE decode requires a non-empty prefill history")
    read_pages = torch.div(sequence_lengths - 1, page_size, rounding_mode="floor")
    write_pages = torch.div(sequence_lengths, page_size, rounding_mode="floor")
    state_read_blocks = self._physical_blocks(
        state_inputs,
        PLE_STATE_TAG,
        read_pages,
        int(state_pool.shape[0]),
        state_pool.device,
        graph_capture=graph_capture,
    )
    ctx_read_blocks = self._physical_blocks(
        ctx_inputs,
        PLE_NGRAM_CTX_TAG,
        read_pages,
        int(ctx_pool.shape[0]),
        ctx_pool.device,
        graph_capture=graph_capture,
    )
    state_write_blocks = self._physical_blocks(
        state_inputs,
        PLE_STATE_TAG,
        write_pages,
        int(state_pool.shape[0]),
        state_pool.device,
        graph_capture=graph_capture,
    )
    ctx_write_blocks = self._physical_blocks(
        ctx_inputs,
        PLE_NGRAM_CTX_TAG,
        write_pages,
        int(ctx_pool.shape[0]),
        ctx_pool.device,
        graph_capture=graph_capture,
    )
    conv_buffer = state_pool.index_select(0, state_read_blocks)
    context = ctx_pool.index_select(0, ctx_read_blocks)
    history = torch.cat([context, ids.view(batch, 1)], dim=1)
    output, new_state = ple.decode_step(
        hyper_states.view(batch, 1, -1),
        history,
        conv_buffer,
        **(
            {"embeddings": ple.ple_embedding._reduce_rows(local_rows[0])}
            if local_rows is not None and local_rows[0] is not None
            else {}
        ),
    )
    state_pool.index_copy_(0, state_write_blocks, new_state.to(state_pool.dtype))
    ctx_pool.index_copy_(
        0,
        ctx_write_blocks,
        history[:, -context_len:].to(ctx_pool.dtype),
    )
    return hyper_states + output.squeeze(1)
