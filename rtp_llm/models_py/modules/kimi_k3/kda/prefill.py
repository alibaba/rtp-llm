"""Kimi K3 Prefill execution using short convolution and cuLA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda.cache import KimiK3KDACache
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    KimiKDARecurrentCheckpointMetadata,
    KimiKDAShortConvMetadata,
    kimi_kda_short_conv_paged_prefill,
    prepare_kimi_kda_recurrent_checkpoint_metadata,
    prepare_kimi_kda_short_conv_metadata,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


_CULA_LOGGED_DEVICES: set[int] = set()


@dataclass(frozen=True)
class KimiKDAPrefillMetadata:
    """Round-scoped KDA metadata and workspace shared by every KDA layer."""

    cu_seqlens_cpu: torch.Tensor
    sequence_count: int
    page_size: int
    required_pages: int
    conv: KimiKDAShortConvMetadata
    recurrent: KimiKDARecurrentCheckpointMetadata
    recurrent_checkpoints: torch.Tensor
    active_original_batch_indices: torch.Tensor
    continuation_mask: torch.Tensor
    active_original_batch_indices_host: tuple[int, ...]
    continuation_mask_host: tuple[bool, ...]


@dataclass
class KimiKDACurrentLayerState:
    """Invocation-scoped continuation state for one KDA layer."""

    conv: torch.Tensor
    recurrent: torch.Tensor
    valid_requests: set[int]


class KimiKDACurrentStateRegistry:
    """Temporary per-request KDA state shared by internal Prefill rounds."""

    def __init__(self, original_batch_size: int) -> None:
        self.original_batch_size = original_batch_size
        self._layers: dict[int, KimiKDACurrentLayerState] = {}

    def get_or_create(
        self,
        layer_idx: int,
        *,
        device: torch.device,
        conv_dtype: torch.dtype,
        history_size: int,
        projection_size: int,
        local_heads: int,
        head_dim: int,
    ) -> KimiKDACurrentLayerState:
        state = self._layers.get(layer_idx)
        if state is None:
            state = KimiKDACurrentLayerState(
                conv=torch.empty(
                    (self.original_batch_size, history_size, 3 * projection_size),
                    dtype=conv_dtype,
                    device=device,
                ),
                recurrent=torch.empty(
                    (
                        self.original_batch_size,
                        local_heads,
                        head_dim,
                        head_dim,
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                valid_requests=set(),
            )
            self._layers[layer_idx] = state
        return state


def prepare_kimi_kda_prefill_metadata(
    cu_seqlens_host: torch.Tensor,
    input_lengths_host: torch.Tensor,
    prefix_lengths_host: torch.Tensor,
    *,
    page_size: int,
    local_heads: int,
    head_dim: int,
    device: torch.device,
    active_original_batch_indices: Optional[Sequence[int]] = None,
    continuation_mask: Optional[Sequence[bool]] = None,
    materialized_block_maps_host: Optional[Sequence[torch.Tensor]] = None,
) -> KimiKDAPrefillMetadata:
    """Build the round-scoped metadata and checkpoint workspace once."""

    if page_size <= 0 or page_size % 64:
        raise ValueError(
            "KDA Prefill page size must be a positive multiple of 64, "
            f"got {page_size}"
        )
    sequence_count = int(cu_seqlens_host.numel()) - 1
    cu_seqlens_cpu = cu_seqlens_host.to(dtype=torch.int32).contiguous()
    input_lengths_cpu = input_lengths_host.to(dtype=torch.int64)
    prefix_lengths_cpu = prefix_lengths_host.to(dtype=torch.int64)
    if torch.any(prefix_lengths_cpu < 0) or torch.any(
        prefix_lengths_cpu % page_size != 0
    ):
        raise ValueError(
            "KDA Prefill prefixes must be non-negative and page-aligned: "
            f"prefixes={prefix_lengths_cpu.tolist()} page_size={page_size}"
        )

    required_pages = int(
        torch.max(
            torch.div(
                prefix_lengths_cpu + input_lengths_cpu + page_size - 1,
                page_size,
                rounding_mode="floor",
            )
        )
    )
    conv = prepare_kimi_kda_short_conv_metadata(cu_seqlens_cpu, device)
    recurrent = prepare_kimi_kda_recurrent_checkpoint_metadata(
        input_lengths_cpu,
        prefix_lengths_cpu,
        page_size,
        device,
        materialized_block_maps_host=materialized_block_maps_host,
    )
    recurrent_checkpoints = torch.empty(
        (1, recurrent.total_checkpoints, local_heads, head_dim, head_dim),
        dtype=torch.float32,
        device=device,
    )
    active_indices = list(
        range(sequence_count)
        if active_original_batch_indices is None
        else active_original_batch_indices
    )
    continuing = list(
        [False] * sequence_count
        if continuation_mask is None
        else continuation_mask
    )
    return KimiKDAPrefillMetadata(
        cu_seqlens_cpu=cu_seqlens_cpu,
        sequence_count=sequence_count,
        page_size=page_size,
        required_pages=required_pages,
        conv=conv,
        recurrent=recurrent,
        recurrent_checkpoints=recurrent_checkpoints,
        active_original_batch_indices=torch.tensor(
            active_indices, dtype=torch.int64, device=device
        ),
        continuation_mask=torch.tensor(
            continuing, dtype=torch.bool, device=device
        ),
        active_original_batch_indices_host=tuple(active_indices),
        continuation_mask_host=tuple(continuing),
    )


class KimiK3KDAPrefill(nn.Module):
    """Prefill-only KDA executor."""

    def __init__(
        self,
        *,
        weights: Dict[str, torch.Tensor],
        cache: KimiK3KDACache,
        local_heads: int,
        head_dim: int,
        projection_size: int,
        gate_lower_bound: Optional[float],
        fused_conv: torch.Tensor,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.cache = cache
        self.local_heads = local_heads
        self.head_dim = head_dim
        self.projection_size = projection_size
        self.gate_lower_bound = gate_lower_bound
        self.fused_conv = fused_conv

    def _cula_checkpoint_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        checkpoint_interval: int,
        checkpoint_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run the cache-backed cuLA checkpoint path."""

        if not q.is_cuda:
            raise RuntimeError("Kimi K3 cuLA Prefill requires CUDA")
        if checkpoint_interval <= 0:
            raise ValueError("K3 cuLA checkpoint interval must be positive")
        if (
            recurrent_state.dtype != torch.float32
            or not recurrent_state.is_contiguous()
        ):
            raise ValueError("K3 cuLA state must be contiguous FP32")
        if (
            checkpoint_states.dtype != torch.float32
            or not checkpoint_states.is_contiguous()
        ):
            raise ValueError("K3 cuLA checkpoints must be contiguous FP32")
        try:
            import cula
            from cula.kda import chunk_kda as cula_chunk_kda
        except Exception as error:
            raise RuntimeError(
                "Prefill requires cuLA but the cuda-linear-attention package "
                f"could not be imported: {type(error).__name__}: {error}"
            ) from error
        if self.gate_lower_bound is None:
            raise RuntimeError("cuLA requires K3's finite gate lower bound")

        device_index = q.device.index if q.device.index is not None else 0
        if device_index not in _CULA_LOGGED_DEVICES:
            logging.info(
                "[KimiK3 cuLA] enabled device=%s package=%s version=%s",
                q.device,
                getattr(cula, "__file__", "<unknown>"),
                getattr(cula, "__version__", "<unknown>"),
            )
            _CULA_LOGGED_DEVICES.add(device_index)
        single_sequence = int(cu_seqlens.numel()) == 2
        cula_cu_seqlens = None if single_sequence else cu_seqlens.contiguous()
        with torch.inference_mode():
            output, final_state, published_checkpoints = cula_chunk_kda(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                raw_gate.to(dtype=q.dtype).contiguous(),
                raw_beta.to(dtype=q.dtype).contiguous(),
                scale=self.head_dim**-0.5,
                initial_state=recurrent_state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=cula_cu_seqlens,
                cu_seqlens_cpu=(
                    None if cula_cu_seqlens is None else cu_seqlens_cpu
                ),
                safe_gate=True,
                lower_bound=float(self.gate_lower_bound),
                disable_recompute=False,
                use_intracard_cp=False,
                A_log=self.weights[W.linear_attn_alog].float().contiguous(),
                dt_bias=self.weights[W.linear_attn_dt_b_kda].float().contiguous(),
                checkpoint_interval=checkpoint_interval,
                checkpoint_states=checkpoint_states,
            )
            if (
                published_checkpoints is None
                or published_checkpoints.data_ptr() != checkpoint_states.data_ptr()
            ):
                raise RuntimeError(
                    "cuLA did not publish into the requested FP32 checkpoint buffer"
                )
            if final_state is not None:
                raise RuntimeError(
                    "cuLA returned final state when output_final_state=False"
                )
        return output.to(dtype=q.dtype)

    def _packed_checkpoint_prefill(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        linear_block_map: torch.Tensor,
        *,
        metadata: KimiKDAPrefillMetadata,
        current_state: Optional[KimiKDACurrentLayerState],
    ) -> torch.Tensor:
        """Run fused paged conv, packed cuLA, and recurrent-only state store."""

        token_count = int(mixed_qkv.shape[0])
        _, conv_cache = self.cache.get_views(kv_cache)
        current_conv = (
            current_state.conv.index_select(
                0, metadata.active_original_batch_indices
            )
            if current_state is not None
            else None
        )
        if current_state is not None:
            missing = [
                request_idx
                for request_idx, continuing in zip(
                    metadata.active_original_batch_indices_host,
                    metadata.continuation_mask_host,
                )
                if continuing and request_idx not in current_state.valid_requests
            ]
            if missing:
                raise RuntimeError(
                    "KDA continuation requested before current state was written: "
                    f"requests={missing}"
                )
        q_conv, k_conv, v_conv, final_conv = kimi_kda_short_conv_paged_prefill(
            mixed_qkv,
            self.fused_conv,
            conv_cache,
            linear_block_map,
            attention_inputs.prefix_lengths,
            cu_seqlens,
            metadata.page_size,
            metadata.conv,
            current_conv_state=current_conv,
            continuation_mask=(
                metadata.continuation_mask if current_state is not None else None
            ),
            return_final_state=current_state is not None,
        )
        physical_initial_state = self.cache.load_recurrent_state(
            kv_cache, attention_inputs, linear_block_map
        )
        if current_state is not None:
            registry_initial_state = current_state.recurrent.index_select(
                0, metadata.active_original_batch_indices
            )
            recurrent_state = torch.where(
                metadata.continuation_mask[:, None, None, None],
                registry_initial_state,
                physical_initial_state,
            ).contiguous()
        else:
            recurrent_state = physical_initial_state
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        output = self._cula_checkpoint_prefill(
            q_conv.reshape(head_shape),
            k_conv.reshape(head_shape),
            v_conv.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.reshape(1, token_count, self.local_heads),
            recurrent_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=metadata.cu_seqlens_cpu,
            checkpoint_interval=metadata.page_size,
            checkpoint_states=metadata.recurrent_checkpoints,
        )
        self.cache.store_recurrent_checkpoints(
            metadata.recurrent_checkpoints,
            metadata.recurrent,
            kv_cache,
            attention_inputs,
            linear_block_map,
        )
        if current_state is not None:
            current_state.conv.index_copy_(
                0, metadata.active_original_batch_indices, final_conv
            )
            final_recurrent = metadata.recurrent_checkpoints.squeeze(0).index_select(
                0, metadata.recurrent.final_checkpoint_indices
            )
            current_state.recurrent.index_copy_(
                0, metadata.active_original_batch_indices, final_recurrent
            )
            current_state.valid_requests.update(
                metadata.active_original_batch_indices_host
            )

        return output

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        metadata: KimiKDAPrefillMetadata,
        current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        linear_block_map = self.cache.linear_state_block_map_device(attention_inputs)
        if metadata.required_pages > linear_block_map.shape[1]:
            raise ValueError(
                "KDA LINEAR block table is too short for Prefill: "
                f"required_pages={metadata.required_pages}, "
                f"available_pages={linear_block_map.shape[1]}"
            )
        current_state = (
            current_state_registry.get_or_create(
                layer_idx,
                device=mixed_qkv.device,
                conv_dtype=mixed_qkv.dtype,
                history_size=int(self.fused_conv.shape[1]) - 1,
                projection_size=self.projection_size,
                local_heads=self.local_heads,
                head_dim=self.head_dim,
            )
            if current_state_registry is not None
            else None
        )
        return self._packed_checkpoint_prefill(
            mixed_qkv,
            raw_gate,
            raw_beta,
            cu_seqlens,
            kv_cache,
            attention_inputs,
            linear_block_map,
            metadata=metadata,
            current_state=current_state,
        )


__all__ = [
    "KimiK3KDAPrefill",
    "KimiKDACurrentLayerState",
    "KimiKDACurrentStateRegistry",
    "KimiKDAPrefillMetadata",
    "prepare_kimi_kda_prefill_metadata",
]
