"""Paged-cache adapter for Kimi K3 delta attention."""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.kimi_k3.kda.state import KimiKDAState
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    kimi_k3_store_linear_cache_state,
    kimi_k3_store_linear_cache_states,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class KimiK3KDACache:
    """Translate RTP paged-cache metadata to canonical KDA state tensors."""

    def __init__(
        self,
        converter: LinearCacheConverter,
        *,
        local_heads: int,
        head_dim: int,
        projection_size: int,
        history_size: int,
    ) -> None:
        self.converter = converter
        self.local_heads = local_heads
        self.head_dim = head_dim
        self.projection_size = projection_size
        self.history_size = history_size

        conv_section_bytes = projection_size * converter.conv_state_item_size
        self.store_segment_sizes = (
            converter.ssm_state_size_bytes,
            *((conv_section_bytes,) * 3 * history_size),
        )
        if sum(self.store_segment_sizes) != converter.block_size_bytes:
            raise ValueError(
                "KDA cache-store segments do not cover the physical linear block: "
                f"segments={sum(self.store_segment_sizes)} "
                f"block={converter.block_size_bytes}"
            )

    def get_views(
        self, kv_cache: LayerKVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpret RTP's linear-cache block storage as SSM and conv states."""

        base = kv_cache.kv_cache_base
        if base is None or base.numel() == 0:
            raise ValueError("KDA LayerKVCache has no backing tensor")
        base = base.reshape(base.shape[0], -1)
        return (
            self.converter.get_ssm_state_tensor(base),
            self.converter.get_conv_state_tensor(base),
        )

    @staticmethod
    def prefix_lengths(
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
    ) -> list[int]:
        cu_host = getattr(attention_inputs, "cu_seqlens_host", None)
        offsets = [
            int(value)
            for value in (
                cu_host
                if cu_host is not None and cu_host.numel()
                else cu_seqlens.detach().cpu()
            ).tolist()
        ]
        sequence_count = max(0, len(offsets) - 1)
        source_host = getattr(attention_inputs, "prefix_lengths_host", None)
        source = (
            source_host
            if source_host is not None and source_host.numel()
            else attention_inputs.prefix_lengths
        )
        if source is None or source.numel() == 0:
            past_lengths = [0] * sequence_count
        else:
            past_lengths = [int(value) for value in source.detach().cpu().tolist()]
        if len(past_lengths) != sequence_count:
            raise ValueError(
                "KDA cache batch does not match packed sequence count: "
                f"past={len(past_lengths)} sequences={sequence_count}"
            )
        return past_lengths

    @staticmethod
    def block_map(attention_inputs: PyAttentionInputs) -> list[list[int]]:
        block_map = attention_inputs.kv_cache_block_id_host
        if block_map is None or block_map.numel() == 0:
            block_map = attention_inputs.kv_cache_kernel_block_id_host
        if block_map is None or block_map.numel() == 0 or block_map.ndim != 2:
            raise ValueError("KDA cache requires a two-dimensional kernel block map")
        return [
            [int(value) for value in row]
            for row in (
                block_map.tolist()
                if block_map.device.type == "cpu"
                else block_map.detach().cpu().tolist()
            )
        ]

    @staticmethod
    def _block_id_or_none(
        block_map: list[list[int]],
        sequence_idx: int,
        token_position: int,
        page_size: int,
    ) -> Optional[int]:
        block_position = token_position // page_size
        if block_position >= len(block_map[sequence_idx]):
            raise ValueError(
                f"linear cache block map is too short for token {token_position}"
            )
        block_id = block_map[sequence_idx][block_position]
        return None if block_id <= 0 else block_id

    @classmethod
    def _block_id(
        cls,
        block_map: list[list[int]],
        sequence_idx: int,
        token_position: int,
        page_size: int,
    ) -> int:
        block_id = cls._block_id_or_none(
            block_map, sequence_idx, token_position, page_size
        )
        if block_id is None:
            raise ValueError(
                "linear cache has no materialized block at position "
                f"{token_position // page_size}"
            )
        return block_id

    @staticmethod
    def _is_fake_block_row(block_row: list[int]) -> bool:
        return bool(block_row) and all(block_id == 0 for block_id in block_row)

    @staticmethod
    def _is_fake_stream(attention_inputs: PyAttentionInputs) -> bool:
        return bool(getattr(attention_inputs, "is_fake_stream", False))

    def load_state(
        self,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
    ) -> KimiKDAState:
        ssm_cache, conv_cache = self.get_views(kv_cache)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        past_lengths = self.prefix_lengths(attention_inputs, cu_seqlens)
        block_map = self.block_map(attention_inputs)

        if len(past_lengths) == 1:
            past_length = past_lengths[0]
            if (
                self._is_fake_stream(attention_inputs)
                or past_length == 0
                or self._is_fake_block_row(block_map[0])
            ):
                conv_states = conv_cache.new_zeros(
                    3, self.projection_size, self.history_size
                )
                recurrent = ssm_cache.new_zeros(
                    1, self.local_heads, self.head_dim, self.head_dim
                )
                return KimiKDAState(
                    q_conv_state=conv_states[0:1],
                    k_conv_state=conv_states[1:2],
                    v_conv_state=conv_states[2:3],
                    recurrent_state=recurrent,
                )
            block_id = self._block_id(block_map, 0, past_length - 1, page_size)
            packed_conv = conv_cache[block_id].transpose(0, 1)
            q_state, k_state, v_state = torch.split(
                packed_conv, self.projection_size, dim=0
            )
            return KimiKDAState(
                q_conv_state=q_state.unsqueeze(0),
                k_conv_state=k_state.unsqueeze(0),
                v_conv_state=v_state.unsqueeze(0),
                recurrent_state=ssm_cache[block_id].unsqueeze(0),
            )

        recurrent_states: list[torch.Tensor] = []
        q_states: list[torch.Tensor] = []
        k_states: list[torch.Tensor] = []
        v_states: list[torch.Tensor] = []
        is_fake_stream = self._is_fake_stream(attention_inputs)
        for sequence_idx, past_length in enumerate(past_lengths):
            if (
                is_fake_stream
                or past_length == 0
                or self._is_fake_block_row(block_map[sequence_idx])
            ):
                recurrent_states.append(
                    ssm_cache.new_zeros(self.local_heads, self.head_dim, self.head_dim)
                )
                empty_conv = conv_cache.new_zeros(
                    self.projection_size, self.history_size
                )
                q_states.append(empty_conv)
                k_states.append(empty_conv.clone())
                v_states.append(empty_conv.clone())
                continue
            block_id = self._block_id(
                block_map, sequence_idx, past_length - 1, page_size
            )
            recurrent_states.append(ssm_cache[block_id])
            packed_conv = conv_cache[block_id].transpose(0, 1)
            q_state, k_state, v_state = torch.split(
                packed_conv, self.projection_size, dim=0
            )
            q_states.append(q_state)
            k_states.append(k_state)
            v_states.append(v_state)
        return KimiKDAState(
            q_conv_state=torch.stack(q_states),
            k_conv_state=torch.stack(k_states),
            v_conv_state=torch.stack(v_states),
            recurrent_state=torch.stack(recurrent_states),
        )

    def store_position(
        self,
        state: KimiKDAState,
        state_index: int,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        sequence_idx: int,
        absolute_position: int,
        *,
        block_map: Optional[list[list[int]]] = None,
    ) -> None:
        if absolute_position < 0:
            raise ValueError("KDA cache position must be non-negative")
        if self._is_fake_stream(attention_inputs):
            return
        ssm_cache, conv_cache = self.get_views(kv_cache)
        block_map = self.block_map(attention_inputs) if block_map is None else block_map
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        block_id = self._block_id_or_none(
            block_map, sequence_idx, absolute_position, page_size
        )
        if block_id is None:
            return
        self._copy_state_to_block(
            state,
            state_index,
            block_id,
            ssm_cache,
            conv_cache,
        )

    def store_blocks(
        self,
        state: KimiKDAState,
        block_ids: list[int],
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
    ) -> None:
        """Store packed checkpoints with one cache-write kernel launch."""

        if self._is_fake_stream(attention_inputs):
            return
        state_count = int(state.recurrent_state.shape[0])
        if state_count != len(block_ids):
            raise ValueError(
                "packed KDA states/block IDs disagree: "
                f"states={state_count} blocks={len(block_ids)}"
            )
        if state_count == 0:
            return
        ssm_cache, conv_cache = self.get_views(kv_cache)
        block_ids_device = torch.tensor(
            block_ids,
            dtype=torch.int32,
            device=state.recurrent_state.device,
        )
        kimi_k3_store_linear_cache_states(
            state.recurrent_state,
            state.q_conv_state,
            state.k_conv_state,
            state.v_conv_state,
            block_ids_device,
            ssm_cache,
            conv_cache,
        )

    @staticmethod
    def _copy_state_to_block(
        state: KimiKDAState,
        state_index: int,
        block_id: int,
        ssm_cache: torch.Tensor,
        conv_cache: torch.Tensor,
    ) -> None:
        kimi_k3_store_linear_cache_state(
            state.recurrent_state[state_index],
            state.q_conv_state[state_index],
            state.k_conv_state[state_index],
            state.v_conv_state[state_index],
            ssm_cache[block_id],
            conv_cache[block_id],
        )


__all__ = ["KimiK3KDACache"]
