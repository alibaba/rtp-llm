"""PLE page planning, embedding lookup and execution for one model."""

from collections.abc import Mapping
from typing import Any

import torch
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.pd_publication import (
    validate_pd_cache_inputs,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs

from .ple_cache_views import PLECacheViews
from .ple_transaction import PLETransactionCoordinator


class PLEExecution:
    def __init__(
        self,
        layers: Mapping,
        cache,
        parallelism,
        eos_token_id,
        coordinator: PLETransactionCoordinator,
    ):
        self.ple_layers = layers
        self.kv_cache = cache
        self.parallelism_config = parallelism
        self.eos_token_id = eos_token_id
        self.coordinator = coordinator
        self.requires_cuda_graph_model_input_preparation = bool(layers)

    _fixed_state_pool = staticmethod(PLECacheViews._fixed_state_pool)
    _physical_blocks = staticmethod(PLECacheViews._physical_blocks)
    _physical_blocks_for_pairs = staticmethod(PLECacheViews._physical_blocks_for_pairs)
    _same_metadata = staticmethod(PLECacheViews._same_metadata)

    def _stage_ple_target_verify(self, **kwargs):
        return self.coordinator._stage_ple_target_verify(**kwargs)

    def _validate_ple_mode(
        self,
        attention_inputs: PyAttentionInputs,
        *,
        allow_target_verify: bool = False,
    ) -> None:
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        if is_target_verify and not allow_target_verify:
            raise RuntimeError("qwen4_exp PLE does not support target-verify/MTP yet")
        is_cuda_graph = bool(getattr(attention_inputs, "is_cuda_graph", False))
        if is_cuda_graph and bool(attention_inputs.is_prefill) and not is_target_verify:
            raise RuntimeError("qwen4_exp PLE prefill does not support CUDA Graph")
        if getattr(attention_inputs, "context_parallel_info", None) is not None:
            raise RuntimeError("qwen4_exp PLE does not support context parallelism yet")
        cp = self.parallelism_config.prefill_cp_config
        if cp.is_enabled() or cp.is_prefill_enabled():
            raise RuntimeError("qwen4_exp PLE does not support prefill CP yet")
        validate_pd_cache_inputs((attention_inputs,))

    def _start_eager_ple_prefetch(self, inputs: PyModelInputs, attention_inputs):
        """Read PLE contexts early; leave persistent state untouched."""
        enabled = [
            (int(key), ple)
            for key, ple in self.ple_layers.items()
            if getattr(getattr(ple, "ple_embedding", None), "supports_prefetch", False)
        ]
        if not enabled:
            return {}
        ctx_inputs = select_attention_inputs_for_tag(
            attention_inputs, PLE_NGRAM_CTX_TAG
        )
        if bool(ctx_inputs.is_cuda_graph):
            return {}
        ids = self._ple_input_ids(inputs).reshape(-1).long()
        plans = {}
        for layer_idx, ple in enabled:
            embedding = ple.ple_embedding
            self._validate_ple_mode(ctx_inputs, allow_target_verify=True)
            cache = self.kv_cache.get_layer_cache(layer_idx, PLE_NGRAM_CTX_TAG)
            page_size = int(cache.seq_size_per_block)
            if ctx_inputs.is_prefill:
                lengths = [int(x) for x in ctx_inputs.input_lengths.tolist()]
                raw_prefixes = ctx_inputs.prefix_lengths
                prefixes = (
                    [int(x) for x in raw_prefixes.tolist()]
                    if raw_prefixes is not None and raw_prefixes.numel()
                    else [0] * len(lengths)
                )
                if (
                    not lengths
                    or any(length <= 0 for length in lengths)
                    or sum(lengths) != ids.numel()
                    or len(prefixes) != len(lengths)
                    or any(prefix < 0 for prefix in prefixes)
                ):
                    raise RuntimeError("PLE prefetch prefill metadata is inconsistent")
                if len(lengths) > 128:
                    continue
                contexts = ids.new_full(
                    (len(lengths), embedding.context_len), embedding.eos_token_id
                )
                if any(prefixes):
                    pool = self._fixed_state_pool(
                        cache, torch.int64, embedding.context_len, 1
                    ).squeeze(-1)
                    pages = torch.tensor(
                        [
                            (prefix - 1) // page_size if prefix else 0
                            for prefix in prefixes
                        ],
                        dtype=torch.long,
                    )
                    blocks = self._physical_blocks(
                        ctx_inputs,
                        PLE_NGRAM_CTX_TAG,
                        pages,
                        int(pool.shape[0]),
                        pool.device,
                    )
                    cached = pool.index_select(0, blocks)
                    mask = torch.tensor(prefixes, device=ids.device).gt(0).unsqueeze(1)
                    contexts = torch.where(mask, cached, contexts)
                if ctx_inputs.is_target_verify:
                    width = lengths[0]
                    if any(length != width for length in lengths):
                        raise RuntimeError(
                            "PLE target prefetch requires uniform query rows"
                        )
                    history = torch.cat(
                        (contexts, ids.view(len(lengths), width)), dim=1
                    )
                    plans[layer_idx] = [
                        embedding.start_prefetch(embedding.hashed_ids(history, width))
                    ]
                else:
                    handles = []
                    offset = 0
                    for slot, length in enumerate(lengths):
                        history = torch.cat(
                            (
                                contexts[slot : slot + 1],
                                ids[offset : offset + length].view(1, length),
                            ),
                            dim=1,
                        )
                        handles.append(
                            embedding.start_prefetch(
                                embedding.hashed_ids(history, length), slot=slot
                            )
                        )
                        offset += length
                    plans[layer_idx] = handles
            else:
                lengths = ctx_inputs.sequence_lengths.to(
                    device=ids.device, dtype=torch.long
                )
                padded = bool(ctx_inputs.is_s_padded)
                if ids.numel() != lengths.numel() or bool(
                    (lengths < (0 if padded else 1)).any().item()
                ):
                    raise RuntimeError("PLE decode prefetch metadata is inconsistent")
                pool = self._fixed_state_pool(
                    cache, torch.int64, embedding.context_len, 1
                ).squeeze(-1)
                pages = torch.div(lengths - 1, page_size, rounding_mode="floor")
                blocks = self._physical_blocks(
                    ctx_inputs,
                    PLE_NGRAM_CTX_TAG,
                    pages,
                    int(pool.shape[0]),
                    pool.device,
                    graph_capture=padded,
                )
                history = torch.cat(
                    (pool.index_select(0, blocks), ids.view(-1, 1)), dim=1
                )
                plans[layer_idx] = [
                    embedding.start_prefetch(embedding.hashed_ids(history, 1))
                ]
        return plans

    def prepare_cuda_graph_model_inputs(self, inputs: PyModelInputs) -> None:
        """Prepare CPU-resident PLE rows before replay, without committing state."""
        if not self.requires_cuda_graph_model_input_preparation or not self.ple_layers:
            return
        groups = get_attention_inputs_value(inputs)
        ctx_inputs = select_attention_inputs_for_tag(groups, PLE_NGRAM_CTX_TAG)
        ids = self._ple_input_ids(inputs).reshape(-1).long()
        graph_key = inputs.input_ids.data_ptr()
        for key, ple in self.ple_layers.items():
            embedding = ple.ple_embedding
            if getattr(embedding, "inline_cpu_uva_graph", False):
                continue
            cache = self.kv_cache.get_layer_cache(int(key), PLE_NGRAM_CTX_TAG)
            pool = self._fixed_state_pool(
                cache, torch.int64, embedding.context_len, 1
            ).squeeze(-1)
            if ctx_inputs.is_prefill:
                lengths = ctx_inputs.input_lengths
                batch = int(lengths.numel())
                if batch <= 0:
                    raise RuntimeError("PLE CPU graph requires nonempty query rows")
                width = int(lengths[0].item())
                if (
                    width <= 0
                    or ids.numel() != batch * width
                    or bool((lengths != width).any().item())
                ):
                    raise RuntimeError(
                        "PLE CPU graph requires uniform nonempty query rows"
                    )
                prefixes = ctx_inputs.prefix_lengths_device.to(
                    device=ids.device, dtype=torch.long
                )
                pages = torch.div(
                    prefixes - 1, cache.seq_size_per_block, rounding_mode="floor"
                )
            else:
                batch, width = int(ids.numel()), 1
                sequence_lengths = ctx_inputs.sequence_lengths_plus_1_device.long() - 1
                pages = torch.div(
                    sequence_lengths - 1,
                    cache.seq_size_per_block,
                    rounding_mode="floor",
                )
            blocks = self._physical_blocks(
                ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                pages,
                int(pool.shape[0]),
                pool.device,
                graph_capture=True,
            )
            context = pool.index_select(0, blocks)
            active = (prefixes > 0) if ctx_inputs.is_prefill else (sequence_lengths > 0)
            context = torch.where(
                active.unsqueeze(1),
                context,
                torch.full_like(context, embedding.eos_token_id),
            )
            if ctx_inputs.is_prefill and not ctx_inputs.is_target_verify:
                context = torch.where(
                    (prefixes > 0).unsqueeze(1),
                    context,
                    torch.full_like(context, embedding.eos_token_id),
                )
            history = torch.cat((context, ids.reshape(batch, width)), dim=1)
            embedding.prepare_cpu_graph_rows(
                embedding.hashed_ids(history, width), graph_key
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
        ctx_inputs = select_attention_inputs_for_tag(
            attention_inputs, PLE_NGRAM_CTX_TAG
        )
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
        host_block_tables = (
            getattr(state_inputs, "kv_cache_block_id", None),
            getattr(ctx_inputs, "kv_cache_block_id", None),
        )
        device_block_tables = (
            getattr(state_inputs, "kv_cache_block_id_device", None),
            getattr(ctx_inputs, "kv_cache_block_id_device", None),
        )
        trusted_host_tables = (
            type(state_inputs) is PyAttentionInputs
            and type(ctx_inputs) is PyAttentionInputs
            and bool(state_inputs.is_prefill)
            and not is_target_verify
            and not graph_capture
            and all(
                isinstance(host, torch.Tensor)
                and isinstance(device_table, torch.Tensor)
                and host.device.type == "cpu"
                and host.dtype == torch.int32
                and host.ndim == 2
                and device_table.is_cuda
                and device_table.shape == host.shape
                for host, device_table in zip(host_block_tables, device_block_tables)
            )
        )
        pair_kwargs = {"trusted_host_table": True} if trusted_host_tables else {}
        # Eager fallback must never reuse rows prepared for a captured graph.
        with ple.ple_embedding.lookup_context(graph_capture):
            local_rows = (
                [
                    (
                        ple.ple_embedding.consume_prefetch(handle)
                        if handle is not None
                        else None
                    )
                    for handle in prefetch
                ]
                if prefetch is not None
                else None
            )
            if is_target_verify:
                prefetched = (
                    ple.ple_embedding._reduce_rows(local_rows[0])
                    if local_rows is not None and local_rows[0] is not None
                    else None
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
                    embeddings=prefetched,
                )

            if state_inputs.is_prefill:
                lengths = [int(x) for x in state_inputs.input_lengths.tolist()]
                if any(length <= 0 for length in lengths):
                    raise RuntimeError(
                        "qwen4_exp PLE requires positive prefill lengths"
                    )
                if sum(lengths) != int(ids.numel()):
                    raise RuntimeError(
                        "qwen4_exp PLE packed prefill lengths are inconsistent"
                    )
                raw_prefixes = getattr(state_inputs, "prefix_lengths", None)
                prefixes = (
                    [int(x) for x in raw_prefixes.tolist()]
                    if raw_prefixes is not None and int(raw_prefixes.numel()) > 0
                    else [0] * len(lengths)
                )
                if len(prefixes) != len(lengths) or any(p < 0 for p in prefixes):
                    raise RuntimeError(
                        "qwen4_exp PLE prefill prefix lengths are inconsistent"
                    )
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
                        [
                            (prefix - 1) // page_size if prefix else 0
                            for prefix in prefixes
                        ],
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
                    if length >= 2 * page_size:
                        request_ids = ids[offset : offset + length].view(1, length)
                        history = torch.cat((context, request_ids), dim=1)
                        output, snapshots = ple.prefill_page_snapshots(
                            hyper_states[offset : offset + length].view(1, length, -1),
                            history,
                            page_size,
                            state,
                            add_residual=True,
                            embeddings=(
                                ple.ple_embedding._reduce_rows(local_rows[idx])
                                if local_rows is not None
                                and local_rows[idx] is not None
                                else None
                            ),
                        )
                        for end, snapshot in zip(
                            range(page_size, length + page_size, page_size), snapshots
                        ):
                            end = min(end, length)
                            write_rows.append(idx)
                            write_pages.append((prefix + end - 1) // page_size)
                            write_states.append(snapshot.squeeze(0))
                            write_contexts.append(history[0, end : end + context_len])
                        outputs.append(output.squeeze(0))
                        offset += length
                        continue
                    chunk_outputs = []
                    start = 0
                    while start < length:
                        end = min(start + page_size, length)
                        chunk_ids = ids[offset + start : offset + end].view(
                            1, end - start
                        )
                        history = torch.cat([context, chunk_ids], dim=1)
                        chunk_hyper = hyper_states[offset + start : offset + end].view(
                            1, end - start, -1
                        )
                        if state is None:
                            output, state = ple.prefill(
                                chunk_hyper,
                                history,
                                embeddings=(
                                    ple.ple_embedding._reduce_rows(
                                        local_rows[idx][:, start:end].contiguous()
                                    )
                                    if local_rows is not None
                                    and local_rows[idx] is not None
                                    else None
                                ),
                            )
                        else:
                            output, state = ple.prefill_with_state(
                                chunk_hyper,
                                history,
                                state,
                                embeddings=(
                                    ple.ple_embedding._reduce_rows(
                                        local_rows[idx][:, start:end].contiguous()
                                    )
                                    if local_rows is not None
                                    and local_rows[idx] is not None
                                    else None
                                ),
                            )
                        chunk_outputs.append((chunk_hyper + output).squeeze(0))
                        write_rows.append(idx)
                        write_pages.append((prefix + end - 1) // page_size)
                        write_states.append(state.squeeze(0))
                        write_contexts.append(history[0, -context_len:])
                        context = history[:, -context_len:]
                        start = end
                    outputs.append(torch.cat(chunk_outputs, dim=0))
                    offset += length
                write_rows_t = torch.tensor(write_rows, dtype=torch.long, device="cpu")
                write_pages_t = torch.tensor(
                    write_pages, dtype=torch.long, device="cpu"
                )
                write_state_blocks, state_writable = self._physical_blocks_for_pairs(
                    state_inputs,
                    PLE_STATE_TAG,
                    write_rows_t,
                    write_pages_t,
                    int(state_pool.shape[0]),
                    state_pool.device,
                    **pair_kwargs,
                )
                write_ctx_blocks, ctx_writable = self._physical_blocks_for_pairs(
                    ctx_inputs,
                    PLE_NGRAM_CTX_TAG,
                    write_rows_t,
                    write_pages_t,
                    int(ctx_pool.shape[0]),
                    ctx_pool.device,
                    **pair_kwargs,
                )
                # Decode resumes from the page holding each request's last token, so
                # that page must exist even when the allocator dropped the rest of
                # the checkpoint chain.
                terminal_indices = []
                placed = 0
                for length in lengths:
                    placed += (length + page_size - 1) // page_size
                    terminal_indices.append(placed - 1)
                if trusted_host_tables:
                    terminal_t = torch.tensor(
                        terminal_indices, dtype=torch.long, device="cpu"
                    )
                    terminal_rows = write_rows_t[terminal_t]
                    terminal_pages = write_pages_t[terminal_t]
                    terminal_writable = bool(
                        (
                            (host_block_tables[0][terminal_rows, terminal_pages] > 0)
                            & (host_block_tables[1][terminal_rows, terminal_pages] > 0)
                        )
                        .all()
                        .item()
                    )
                else:
                    terminal_t = torch.tensor(
                        terminal_indices,
                        dtype=torch.long,
                        device=state_writable.device,
                    )
                    terminal_writable = bool(
                        (state_writable[terminal_t] & ctx_writable[terminal_t])
                        .all()
                        .item()
                    )
                if not terminal_writable:
                    raise RuntimeError(
                        "qwen4_exp PLE terminal page is unallocated: decode resumes "
                        "from its checkpoint"
                    )
                if trusted_host_tables:
                    # GPU boolean indexing calls nonzero to size its output and
                    # synchronizes the stream. The same C++-copied host tables
                    # already establish which checkpoint pages have storage.
                    writable_positions = [
                        index
                        for index, (row, page) in enumerate(
                            zip(write_rows, write_pages)
                        )
                        if host_block_tables[0][row, page] > 0
                        and host_block_tables[1][row, page] > 0
                    ]
                    host_positions = torch.tensor(
                        writable_positions,
                        dtype=torch.long,
                        device="cpu",
                        pin_memory=True,
                    )
                    writable = host_positions.to(
                        device=state_pool.device, non_blocking=True
                    )
                    ctx_writable_positions = (
                        writable
                        if ctx_pool.device == state_pool.device
                        else host_positions.to(
                            device=ctx_pool.device, non_blocking=True
                        )
                    )
                    state_blocks_to_write = write_state_blocks.index_select(0, writable)
                    ctx_blocks_to_write = write_ctx_blocks.index_select(
                        0, ctx_writable_positions
                    )
                    states_to_write = torch.stack(write_states).index_select(
                        0, writable
                    )
                    contexts_to_write = torch.stack(write_contexts).index_select(
                        0, ctx_writable_positions
                    )
                else:
                    writable = state_writable & ctx_writable
                    state_blocks_to_write = write_state_blocks[writable]
                    ctx_blocks_to_write = write_ctx_blocks[writable]
                    states_to_write = torch.stack(write_states)[writable]
                    contexts_to_write = torch.stack(write_contexts)[writable]
                state_pool.index_copy_(
                    0,
                    state_blocks_to_write,
                    states_to_write.to(state_pool.dtype),
                )
                ctx_pool.index_copy_(
                    0,
                    ctx_blocks_to_write,
                    contexts_to_write.to(ctx_pool.dtype),
                )
                return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

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
            padded = bool(state_inputs.is_s_padded)
            if not graph_capture and bool(
                (sequence_lengths < (0 if padded else 1)).any().item()
            ):
                raise RuntimeError(
                    "qwen4_exp PLE decode requires a non-empty prefill history"
                )
            read_pages = torch.div(
                sequence_lengths - 1, page_size, rounding_mode="floor"
            )
            write_pages = torch.div(sequence_lengths, page_size, rounding_mode="floor")
            state_read_blocks = self._physical_blocks(
                state_inputs,
                PLE_STATE_TAG,
                read_pages,
                int(state_pool.shape[0]),
                state_pool.device,
                graph_capture=graph_capture or padded,
            )
            ctx_read_blocks = self._physical_blocks(
                ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                read_pages,
                int(ctx_pool.shape[0]),
                ctx_pool.device,
                graph_capture=graph_capture or padded,
            )
            state_write_blocks = self._physical_blocks(
                state_inputs,
                PLE_STATE_TAG,
                write_pages,
                int(state_pool.shape[0]),
                state_pool.device,
                graph_capture=graph_capture or padded,
            )
            ctx_write_blocks = self._physical_blocks(
                ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                write_pages,
                int(ctx_pool.shape[0]),
                ctx_pool.device,
                graph_capture=graph_capture or padded,
            )
            conv_buffer = state_pool.index_select(0, state_read_blocks)
            context = ctx_pool.index_select(0, ctx_read_blocks)
            live = sequence_lengths.to(device=state_pool.device) > 0
            conv_buffer = torch.where(live[:, None, None], conv_buffer, 0)
            context = torch.where(live[:, None], context, eos)
            state_write_blocks = torch.where(live, state_write_blocks, 0)
            ctx_write_blocks = torch.where(live, ctx_write_blocks, 0)
            history = torch.cat([context, ids.view(batch, 1)], dim=1)
            output, new_state = ple.decode_step(
                hyper_states.view(batch, 1, -1),
                history,
                conv_buffer,
                embeddings=(
                    ple.ple_embedding._reduce_rows(local_rows[0])
                    if local_rows is not None and local_rows[0] is not None
                    else None
                ),
            )
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
                store_decode_rows,
            )

            store_decode_rows(state_pool, state_write_blocks, new_state)
            store_decode_rows(ctx_pool, ctx_write_blocks, history[:, -context_len:])
            return hyper_states + output.squeeze(1)

    def _ple_input_ids(self, inputs: PyModelInputs) -> torch.Tensor:
        return ple_input_ids(inputs, self.eos_token_id)


def ple_input_ids(inputs: PyModelInputs, eos_token_id) -> torch.Tensor:
    """Text ids for the n-gram hash; non-text positions hash as EOS.

    Mirrors upstream: ``ple_input_ids = where(text_mask, ids, eos_token_id)``,
    so image-placeholder positions contribute the EOS id to the n-gram
    history instead of an arbitrary vision token id.
    """
    input_ids = inputs.input_ids
    embedding_inputs = getattr(inputs, "embedding_inputs", None)
    text_mask = getattr(embedding_inputs, "text_tokens_mask", None)
    if text_mask is None or not text_mask.numel():
        return input_ids
    if text_mask.numel() != input_ids.numel():
        raise RuntimeError(
            "qwen4_exp PLE text token mask size does not match input ids: "
            f"{text_mask.numel()} != {input_ids.numel()}"
        )
    text_flags = text_mask.bool()
    if bool(text_flags.all().item()):
        return input_ids
    eos = eos_token_id
    if eos is None:
        raise RuntimeError(
            "qwen4_exp PLE needs an eos token id to hash multimodal positions"
        )
    return torch.where(text_flags, input_ids, torch.full_like(input_ids, eos))
