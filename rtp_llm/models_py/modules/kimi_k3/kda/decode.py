"""Kimi K3 paged Decode and target-verification execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda.cache import KimiK3KDACache
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    fused_recurrent_kda,
    is_kimi_kda_short_conv_paged_decode_supported,
    kimi_kda_short_conv_paged_decode,
    kimi_kda_short_conv_paged_target_verify,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class _PagedDecodeCache:
    ssm: torch.Tensor
    conv: torch.Tensor
    block_map: torch.Tensor
    sequence_lengths_plus_one: torch.Tensor
    page_size: int


class KimiK3KDADecode(nn.Module):
    """Decode-only KDA executor backed by the physical paged cache."""

    def __init__(
        self,
        *,
        weights: Dict[str, torch.Tensor],
        cache: KimiK3KDACache,
        local_heads: int,
        head_dim: int,
        projection_size: int,
        history_size: int,
        gate_lower_bound: float | None,
        fused_conv: torch.Tensor,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.cache = cache
        self.local_heads = local_heads
        self.head_dim = head_dim
        self.projection_size = projection_size
        self.history_size = history_size
        self.gate_lower_bound = gate_lower_bound
        self.fused_conv = fused_conv

    def _cache_context(
        self,
        hidden_states: torch.Tensor,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
    ) -> _PagedDecodeCache:
        """Validate and return device-resident paged Decode state."""

        if not hidden_states.is_cuda:
            raise RuntimeError("Kimi K3 Decode requires CUDA inputs")
        sequence_lengths_plus_one = getattr(
            attention_inputs, "sequence_lengths_plus_1_d", None
        )
        # LINEAR cache groups use one kernel block per physical state block;
        # select_block_map_for_layer therefore exposes the physical KDA IDs
        # through the existing kernel-map field.
        block_map = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
        if (
            sequence_lengths_plus_one is None
            or block_map is None
            or not sequence_lengths_plus_one.is_cuda
            or not block_map.is_cuda
            or sequence_lengths_plus_one.ndim != 1
            or block_map.ndim != 2
            or sequence_lengths_plus_one.numel() != block_map.shape[0]
            or hidden_states.shape[0] % block_map.shape[0] != 0
            or block_map.shape[1] == 0
        ):
            raise RuntimeError(
                "KDA paged decode requires CUDA sequence lengths and a "
                "two-dimensional LINEAR block map matching the request batch"
            )

        ssm_cache, conv_cache = self.cache.get_views(kv_cache)
        page_size = int(kv_cache.seq_size_per_block)
        if (
            page_size <= 0
            or ssm_cache.dtype != torch.float32
            or ssm_cache.ndim != 4
            or tuple(ssm_cache.shape[1:])
            != (self.local_heads, self.head_dim, self.head_dim)
            or conv_cache.ndim != 3
            or tuple(conv_cache.shape[1:])
            != (self.history_size, 3 * self.projection_size)
            or ssm_cache.device != hidden_states.device
            or conv_cache.device != hidden_states.device
        ):
            raise RuntimeError(
                "KDA paged decode cache layout does not match the model "
                "state and convolution dimensions"
            )
        return _PagedDecodeCache(
            ssm=ssm_cache,
            conv=conv_cache,
            block_map=block_map,
            sequence_lengths_plus_one=sequence_lengths_plus_one,
            page_size=page_size,
        )

    def _recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        ssm_cache: torch.Tensor,
        block_map: torch.Tensor,
        sequence_lengths_plus_one: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """Run one indexed recurrence and update physical SSM pages."""

        token_count = q.shape[0]
        batch = block_map.shape[0]
        if token_count % batch != 0:
            raise ValueError(
                f"KDA indexed token count {token_count} is not divisible by "
                f"batch {batch}"
            )
        sequence_length = token_count // batch
        indexed_shape = (
            batch,
            sequence_length,
            self.local_heads,
            self.head_dim,
        )
        output, _ = fused_recurrent_kda(
            q.reshape(indexed_shape),
            k.reshape(indexed_shape),
            v.reshape(indexed_shape),
            raw_gate.reshape(indexed_shape),
            raw_beta.reshape(batch, sequence_length, self.local_heads),
            initial_state=ssm_cache,
            A_log=self.weights[W.linear_attn_alog],
            dt_bias=self.weights[W.linear_attn_dt_b_kda],
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            # RTP stores [H,K,V]; the kernel translates its V-first tile.
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=sequence_lengths_plus_one,
        )
        return output.reshape(1, token_count, self.local_heads, self.head_dim).to(
            dtype=q.dtype
        )

    def _short_conv(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        cache: _PagedDecodeCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not is_kimi_kda_short_conv_paged_decode_supported(
            q_projected,
            k_projected,
            v_projected,
            self.fused_conv,
            cache.conv,
            cache.block_map,
            cache.sequence_lengths_plus_one,
            cache.page_size,
        ):
            raise RuntimeError(
                "KDA paged short-conv does not support this Decode input"
            )
        return kimi_kda_short_conv_paged_decode(
            q_projected,
            k_projected,
            v_projected,
            self.fused_conv,
            cache.conv,
            cache.block_map,
            cache.sequence_lengths_plus_one,
            cache.page_size,
        )

    def _target_verify(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cache: _PagedDecodeCache,
    ) -> torch.Tensor:
        """Replay speculative positions through the exact paged Decode kernels."""

        token_count = q_projected.shape[0]
        batch = cache.block_map.shape[0]
        if token_count % batch != 0:
            raise ValueError(
                f"KDA target token count {token_count} is not divisible by "
                f"batch {batch}"
            )
        sequence_length = token_count // batch
        q_steps = q_projected.reshape(batch, sequence_length, -1)
        k_steps = k_projected.reshape(batch, sequence_length, -1)
        v_steps = v_projected.reshape(batch, sequence_length, -1)
        q, k, v = kimi_kda_short_conv_paged_target_verify(
            q_steps,
            k_steps,
            v_steps,
            self.fused_conv,
            cache.conv,
            cache.block_map,
            cache.sequence_lengths_plus_one,
            cache.page_size,
        )
        # Both target-verify kernels consume the original request block map.
        # They derive read/write pages in-kernel and publish every speculative
        # checkpoint directly, so no layer-local page-index tensors, cloned
        # block maps, cache copies, or per-step packing launches are required.
        return self._recurrent(
            q.reshape(token_count, self.projection_size),
            k.reshape(token_count, self.projection_size),
            v.reshape(token_count, self.projection_size),
            raw_gate,
            raw_beta,
            cu_seqlens,
            cache.ssm,
            cache.block_map,
            cache.sequence_lengths_plus_one,
            cache.page_size,
        )

    def forward(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        cache = self._cache_context(q_projected, kv_cache, attention_inputs)
        if is_target_verify:
            return self._target_verify(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                cache,
            )
        q, k, v = self._short_conv(q_projected, k_projected, v_projected, cache)
        return self._recurrent(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            cu_seqlens,
            cache.ssm,
            cache.block_map,
            cache.sequence_lengths_plus_one,
            cache.page_size,
        )


__all__ = ["KimiK3KDADecode"]
