"""Paged-cache adapter for Kimi K3 delta attention."""

from __future__ import annotations

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    KimiKDARecurrentCheckpointMetadata,
    kimi_kda_load_recurrent_state,
    kimi_kda_store_recurrent_checkpoints,
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
    def linear_state_block_map_device(
        attention_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        """Return the selected LINEAR group's state-block map.

        LINEAR cache groups use one kernel block per physical state block, so
        their selected kernel table is also the physical index table for the
        KDA conv and recurrent cache tensors.
        """

        block_map = getattr(
            attention_inputs, "kv_cache_kernel_block_id_device", None
        )
        if (
            block_map is None
            or not block_map.numel()
            or not block_map.is_cuda
            or block_map.ndim != 2
        ):
            raise ValueError(
                "KDA cache requires a two-dimensional CUDA LINEAR block map"
            )
        if block_map.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"KDA LINEAR block map must be int32/int64, got {block_map.dtype}"
            )
        return block_map

    @staticmethod
    def _is_fake_stream(attention_inputs: PyAttentionInputs) -> bool:
        return bool(getattr(attention_inputs, "is_fake_stream", False))

    def load_recurrent_state(
        self,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        linear_block_map: torch.Tensor,
    ) -> torch.Tensor:
        """Gather cache-backed cuLA initial state without touching conv state."""

        ssm_cache, _ = self.get_views(kv_cache)
        return kimi_kda_load_recurrent_state(
            attention_inputs.prefix_lengths,
            linear_block_map,
            ssm_cache,
            int(kv_cache.seq_size_per_block),
        )

    def store_recurrent_checkpoints(
        self,
        checkpoints: torch.Tensor,
        metadata: KimiKDARecurrentCheckpointMetadata,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        linear_block_map: torch.Tensor,
    ) -> None:
        """Publish cuLA checkpoints into only the physical recurrent region."""

        if self._is_fake_stream(attention_inputs):
            return
        ssm_cache, _ = self.get_views(kv_cache)
        kimi_kda_store_recurrent_checkpoints(
            checkpoints,
            metadata,
            linear_block_map,
            ssm_cache,
        )

__all__ = ["KimiK3KDACache"]
