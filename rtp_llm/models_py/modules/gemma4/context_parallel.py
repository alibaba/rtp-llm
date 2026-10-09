"""Gemma4 prefill CP with owner-sharded persistent pages and temporary gathers."""

import torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context_for_forward,
    cp_all_gather_full,
    cp_gather_request_pool_blocks,
)
from rtp_llm.models_py.modules.factory.attention import common as attn_common
from rtp_llm.models_py.modules.gemma4.core import Gemma4RopeTable
from torch.nn import functional as F


class Gemma4ContextParallelFMHAImpl:
    def __init__(self, config, geometry, inputs, page_size, parallelism):
        if inputs.is_cuda_graph:
            raise ValueError("Gemma4 prefill CP requires eager attention planning")
        if not parallelism.prefill_cp_config.kv_cache_sharded:
            raise ValueError("Gemma4 prefill CP requires owner-sharded KV")
        self.attn_inputs = inputs
        self.geometry = geometry
        self.page_size = page_size
        self._graph_mode = False
        self.vision_group_ids = None
        self.rope_table = Gemma4RopeTable(
            geometry.head_dim, geometry.rope_theta, geometry.rope_partial_rotary_factor
        )
        local_lengths = inputs.input_lengths.detach().cpu().tolist()
        device = inputs.input_lengths_device.device
        if device.type != "cuda":
            device = inputs.context_parallel_info.prefill_qkv_restore_indice.device
        self.context = build_cp_context_for_forward(
            inputs.context_parallel_info,
            parallelism.tp_size,
            parallelism.tp_rank,
            sum(local_lengths),
            device,
            prefix_lengths=inputs.prefix_lengths_device,
            prefix_lengths_host=inputs.context_parallel_info.prefill_prefix_lengths_cpu,
            kv_cache_sharded=True,
        )
        self.local_lengths = self.context.chunk_lengths_per_req
        self.global_lengths = self.context.input_lengths_global_host
        self.prefix_lengths = self.context.prefix_lengths_host
        if self.global_lengths is None or self.prefix_lengths is None:
            raise ValueError("Gemma4 CP requires global input and prefix lengths")
        table = inputs.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            table = inputs.kv_cache_kernel_block_id
        self.block_table = table.to(device=device, dtype=torch.long)
        self.write_cache_store_impl = attn_common.create_write_cache_store_impl(inputs)

    def positions(self, num_tokens):
        if num_tokens != self.context.chunk_length:
            raise ValueError("Gemma4 CP query row count disagrees with zigzag metadata")
        return self.context.global_positions

    def rope_cos_sin(self, positions):
        return self.rope_table.cos_sin(positions)

    def rope_cos_sin_bf16(self, positions):
        return self.rope_table.cos_sin_bf16(positions)

    def _pool(self, cache):
        if cache is None:
            return None
        return (
            attn_common.reshape_paged_kv_cache(
                cache.kv_cache_base,
                self.geometry.kv_head_num,
                self.page_size,
                self.geometry.head_dim,
            )
            if cache.kv_cache_base.dim() == 2
            else cache.kv_cache_base
        )

    def _prefix(self, pool, request, prefix):
        if prefix == 0:
            return None
        total_blocks = (prefix + self.page_size - 1) // self.page_size
        local_blocks = (total_blocks + self.context.cp_size - 1) // self.context.cp_size
        table = self.block_table[request, :local_blocks]
        start = (
            max(0, prefix - self.geometry.sliding_window + 1)
            if self.geometry.sliding_window
            else 0
        )
        # Every rank validates before any rank packs pages for the gather. A
        # local exception would leave peers blocked in the next collective.
        incomplete = table.numel() != local_blocks or pool is None or pool.size(0) == 0
        status = torch.zeros(2, dtype=torch.int32, device=self.block_table.device)
        status[0] = int(incomplete)
        if not incomplete:
            logical = torch.arange(
                self.context.cp_rank,
                total_blocks,
                self.context.cp_size,
                device=pool.device,
            )
            needed = logical * self.page_size + self.page_size > start
            required = table[: logical.numel()][needed]
            status[1] = ((required < 0) | (required >= pool.size(0))).any()
        status_by_rank = all_gather(status, group=Group.TP).reshape(-1, 2)
        if bool(status_by_rank[:, 0].any()):
            raise ValueError(
                "Gemma4 CP prefix page table or persistent KV is incomplete"
            )
        if bool(status_by_rank[:, 1].any()):
            raise ValueError(
                "Gemma4 CP has a missing or invalid owned prefix page inside the attention window"
            )
        # Evicted pages and trailing padding do not enter attention. Pack a
        # valid dummy page for those slots so the collective has a fixed shape.
        safe_table = torch.where((table >= 0) & (table < pool.size(0)), table, 0)
        gathered = cp_gather_request_pool_blocks(
            pool,
            safe_table,
            self.context.cp_size,
            self.context.cp_rank,
            total_blocks,
        )
        # [blocks, 2, heads, page, dim] -> [2, tokens, heads, dim].
        return (
            gathered.permute(1, 0, 3, 2, 4).reshape(
                2, -1, self.geometry.kv_head_num, self.geometry.head_dim
            )[:, start:prefix],
            start,
        )

    def _store(self, pool, request, key, value, prefix):
        if pool is None:
            return
        positions = torch.arange(prefix, prefix + key.size(0), device=key.device)
        logical_blocks = positions // self.page_size
        owned = logical_blocks.remainder(self.context.cp_size) == self.context.cp_rank
        slots = logical_blocks[owned] // self.context.cp_size
        if slots.numel() == 0:
            return
        if int(slots.max()) >= self.block_table.size(1):
            raise ValueError("Gemma4 CP current token is outside its page table")
        physical = self.block_table[request].index_select(0, slots)
        valid = physical >= 0
        # Skipped SWA pages are never persisted. Current KV remains available
        # in the temporary gather for the complete current-query calculation.
        source_rows = torch.arange(key.size(0), device=key.device)[owned][valid]
        page_rows = positions[owned][valid].remainder(self.page_size)
        physical = physical[valid]
        pool[physical, 0, :, page_rows, :] = key.index_select(0, source_rows)
        pool[physical, 1, :, page_rows, :] = value.index_select(0, source_rows)

    def forward(self, query, key, value, cache):
        packed = torch.cat(
            [key.reshape(key.size(0), -1), value.reshape(value.size(0), -1)], dim=1
        )
        full = cp_all_gather_full(packed, self.context)
        full = full.reshape(-1, 2, self.geometry.kv_head_num, self.geometry.head_dim)
        return self._forward_with_current(query, full, cache)

    def _forward_with_current(self, query, full, cache):
        pool = self._pool(cache)
        output = torch.empty_like(query)
        q_offset = k_offset = 0
        for request, (local_length, global_length, prefix) in enumerate(
            zip(self.local_lengths, self.global_lengths, self.prefix_lengths)
        ):
            current = full[k_offset : k_offset + global_length]
            current_key, current_value = current[:, 0], current[:, 1]
            old = self._prefix(pool, request, prefix)
            key_start = old[1] if old is not None else prefix
            all_key = (
                torch.cat([old[0][0], current_key], dim=0)
                if old is not None
                else current_key
            )
            all_value = (
                torch.cat([old[0][1], current_value], dim=0)
                if old is not None
                else current_value
            )
            self._store(pool, request, current_key, current_value, prefix)
            repeat = self.geometry.head_num // self.geometry.kv_head_num
            all_key = all_key.repeat_interleave(repeat, dim=1).transpose(0, 1)[None]
            all_value = all_value.repeat_interleave(repeat, dim=1).transpose(0, 1)[None]
            key_positions = torch.arange(
                key_start, prefix + global_length, device=query.device
            )
            for offset in range(0, local_length, 256):
                count = min(256, local_length - offset)
                begin = q_offset + offset
                positions = self.context.global_positions[begin : begin + count]
                allowed = key_positions[None] <= positions[:, None]
                if self.geometry.sliding_window:
                    allowed &= (
                        key_positions[None]
                        > positions[:, None] - self.geometry.sliding_window
                    )
                q = query[begin : begin + count].transpose(0, 1)[None]
                attended = F.scaled_dot_product_attention(
                    q, all_key, all_value, attn_mask=allowed[None, None], scale=1.0
                )
                output[begin : begin + count] = attended[0].transpose(0, 1)
            q_offset += local_length
            k_offset += global_length
        attn_common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, cache
        )
        return output


class Gemma4ContextParallelDecodeFMHAImpl(Gemma4ContextParallelFMHAImpl):
    """Replicated decode queries consume temporary gathers of owner-sharded KV."""

    def __init__(self, config, geometry, inputs, page_size, parallelism):
        from types import SimpleNamespace

        if inputs.is_cuda_graph:
            raise ValueError(
                "Gemma4 owner-sharded CP decode graph planning is not implemented"
            )
        self.attn_inputs = inputs
        self.geometry = geometry
        self.page_size = page_size
        self._graph_mode = False
        self.vision_group_ids = None
        self.rope_table = Gemma4RopeTable(
            geometry.head_dim, geometry.rope_theta, geometry.rope_partial_rotary_factor
        )
        self.prefix_lengths = tuple(inputs.sequence_lengths.detach().cpu().tolist())
        self.context = SimpleNamespace(
            cp_size=int(parallelism.tp_size), cp_rank=int(parallelism.tp_rank)
        )
        table = inputs.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            table = inputs.kv_cache_kernel_block_id
        device = torch.device("cuda", int(parallelism.local_rank))
        self.block_table = table.to(device=device, dtype=torch.long)
        self.write_cache_store_impl = None

    def positions(self, num_tokens):
        batch = len(self.prefix_lengths)
        if batch == 0 or num_tokens % batch:
            raise ValueError(
                "Gemma4 CP decode requires a uniform query width per request"
            )
        width = num_tokens // batch
        self.local_lengths = self.global_lengths = (width,) * batch
        self.context.global_positions = torch.cat(
            [
                torch.arange(prefix, prefix + width, device=self.block_table.device)
                for prefix in self.prefix_lengths
            ]
        )
        return self.context.global_positions

    def forward(self, query, key, value, cache):
        self.positions(query.size(0))
        # Inputs are replicated on CP decode ranks. Persist only this rank's
        # owned pages and gather the resident prefix before attention.
        full = torch.stack([key, value], dim=1)
        return self._forward_with_current(query, full, cache)
