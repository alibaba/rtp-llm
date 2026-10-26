"""Shared KDA convolution, recurrence and paged-cache lifecycle.

Model descriptions supply projections; this layer owns the recurrent execution
contract for both chunked prefill and Graph-compatible decode/verification.
"""

from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from rtp_llm.device.device_type import is_cuda
from rtp_llm.models_py.triton_kernels.linear_replay import linear_serial_replay
from rtp_llm.models_py.triton_kernels.kda import fused_recurrent_kda
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import LinearAttentionConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype


class KdaMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
        prefill_paged_conv_meta=None,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        self.prefill_paged_conv_meta = prefill_paged_conv_meta
        self.prefill_cuda_inputs = {}

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


class KdaBase(nn.Module):
    """Base class for KDA (Kimi Delta Attention) prefill/decode.

    KDA differs from GDN in:
      - qkv_size = q + k + v (no z component, since output gate is separate LoRA)
      - beta is [B, H] scalar per head (from b_proj), not packed with a
      - forget gate g is [B, H, D] per-dim vector (from the model gate projection)
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.gate_lower_bound = None
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // parallelism_config.tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        # KDA: qkv only (no z), since output gate is separate LoRA
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        self.ssm_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.ssm_state_dtype
        )
        self.conv_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.conv_state_dtype
        )
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b_kda]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KdaMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class KdaPrefill(KdaBase):
    """Generic KDA Prefill using the existing in-tree chunk implementation."""

    def __init__(self, config, parallelism, weights, use_paged_conv=False):
        super().__init__(config, parallelism, weights)
        self.use_paged_conv = use_paged_conv

    @staticmethod
    def _validate_prefix_state(attn_inputs, seq_size_per_block):
        # Validate cached-prefix ownership before a kernel can read the state.
        prefixes = attn_inputs.prefix_lengths
        table = attn_inputs.kv_cache_kernel_block_id
        if prefixes.device.type != "cpu" or table.device.type != "cpu":
            raise ValueError("KDA convolution requires host cache metadata mirrors")
        for row, prefix in enumerate(prefixes.tolist()):
            if prefix > 0:
                position = (prefix - 1) // seq_size_per_block
                if position >= table.shape[1] or int(table[row, position]) <= 0:
                    raise ValueError("KDA convolution is missing a cached prefix state")

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
    ) -> torch.Tensor:
        cu_seqlen_without_padding = attn_inputs.cu_seqlens_device
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_device,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        attn_meta: Optional[KdaMetadata] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.kda.chunk import chunk_kda
        from rtp_llm.models_py.triton_kernels.fla.block import (
            load_initial_state_from_block_map, store_ssm_state_to_block_map,
        )

        # Compute gate: [T, local_H*D] -> [1, T, local_H, D]
        g = forget_gate.view(
            1, forget_gate.shape[0], self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        # beta: [token, H] -> apply sigmoid in float32 -> [1, token, H]
        beta_reshaped = beta.float().sigmoid().unsqueeze(0)

        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens_device
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        query = query.view(
            1, query.shape[0], self.local_num_k_heads, self.head_k_dim
        ).contiguous()
        key = key.view(
            1, key.shape[0], self.local_num_k_heads, self.head_k_dim
        ).contiguous()
        value = value.view(
            1, value.shape[0], self.local_num_v_heads, self.head_v_dim
        ).contiguous()

        # Prefill: use chunk_kda with fused gate (use_gate_in_kernel=True)
        attn_out, final_state, h = chunk_kda(
            query,
            key,
            value,
            g,
            beta_reshaped,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=self.alog,
            dt_bias=self.dt_bias,
        )
        h_from_chunk = h

        if ssm_states is not None and final_state is not None:
            h_for_store = (
                h_from_chunk
                if h_from_chunk is not None
                else final_state.unsqueeze(0).unsqueeze(0)
            )
            # store_ssm_state_to_block_map requires h and final_state in float32
            if h_for_store.dtype != torch.float32:
                h_for_store = h_for_store.float()
            if final_state.dtype != torch.float32:
                final_state = final_state.float()

            store_ssm_state_to_block_map(
                h_for_store,
                final_state,
                attn_inputs.prefix_lengths_device,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )

        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KdaMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 64
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        attn_out = self._compute_attention(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            attn_meta,
        )
        cache_store_inputs = attn_inputs.cache_store_inputs
        cache_store_writer = attn_inputs.cache_store_writer
        if (
            kv_cache is not None
            and cache_store_inputs is not None
            and cache_store_writer is not None
        ):
            cache_store_writer.write(cache_store_inputs, kv_cache)
        return attn_out

    def _compute_attention(
        self, mixed_qkv, forget_gate, beta, kv_cache_tensor,
        seq_size_per_block, attn_inputs, attn_meta,
    ):
        if self.use_paged_conv:
            from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
                paged_short_conv_prefill,
            )

            metadata = attn_meta.prefill_paged_conv_meta
            if metadata is None:
                raise ValueError("paged convolution metadata is required for prefill")
            conv_states = block_map = None
            if kv_cache_tensor is not None:
                self._validate_prefix_state(attn_inputs, seq_size_per_block)
                conv_states = self._get_conv_states(kv_cache_tensor)
                block_map = attn_inputs.kv_cache_kernel_block_id_device
            elif any(attn_inputs.prefix_lengths.tolist()):
                raise ValueError("A cached KDA prefix requires recurrent cache storage")
            q, k, v, _, beta = paged_short_conv_prefill(
                mixed_qkv, self.conv_weights, conv_states, block_map,
                attn_inputs.prefix_lengths_device, attn_inputs.cu_seqlens_device,
                seq_size_per_block, metadata, aux=beta,
            )
            mixed_qkv = (q, k, v)
        else:
            mixed_qkv = self._conv1d(
                mixed_qkv, kv_cache_tensor, seq_size_per_block, attn_inputs,
                metadata=attn_meta.get_prefill_conv1d_meta(),
            )
        return self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            attn_meta,
        )


class KdaPagedPrefill(KdaPrefill):
    """Paged checkpoint prefill for callers using the pinned cuLA operator."""

    def __init__(self, config, parallelism, weights, use_paged_conv=False):
        super().__init__(config, parallelism, weights, use_paged_conv=use_paged_conv)
        if self.head_k_dim != 128 or self.local_num_k_heads != self.local_num_v_heads:
            raise ValueError("cuLA paged KDA requires equal Q/K/V heads with head_dim=128")
        self.alog = self.alog.float().contiguous()
        self.dt_bias = self.dt_bias.float().contiguous()

    def _fla(
        self,
        mixed_qkv,
        forget_gate,
        beta,
        kv_cache_tensor,
        seq_size_per_block,
        attn_inputs,
        attn_meta=None,
    ):
        from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
            cula_kda_paged_prefill,
            prepare_cula_kda_prefill_metadata,
        )

        shape = (-1, self.local_num_v_heads, self.head_k_dim)
        if isinstance(mixed_qkv, tuple):
            q, k, v = (x.reshape(shape) for x in mixed_qkv)
            tokens, device = mixed_qkv[0].shape[0], mixed_qkv[0].device
        else:
            q, k, v = (x.reshape(shape) for x in mixed_qkv.chunk(3, dim=-1))
            tokens, device = mixed_qkv.shape[0], mixed_qkv.device
        states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        # Inputs and page IDs are immutable within this round. Keep each
        # attention group's mapping separate while sharing it across layers.
        key = (
            attn_inputs,
            seq_size_per_block,
            tokens,
            self.local_num_v_heads,
            device,
            None if states is None else (states.shape[0], states.dtype),
        )
        cached = None if attn_meta is None else attn_meta.prefill_cuda_inputs.get(key)
        if cached is not None:
            cu, prefixes, block_table, metadata, scratch_states = cached
            seq_size_per_block = metadata.block_size
            if states is None:
                states = scratch_states
        else:
            cu = attn_inputs.cu_seqlens
            prefixes = attn_inputs.prefix_lengths
            if cu.device.type != "cpu" or prefixes.device.type != "cpu":
                raise ValueError("cuLA KDA prefill requires RTP host metadata mirrors")
            cu, prefixes = cu.tolist(), prefixes.tolist()
            if len(prefixes) != len(cu) - 1:
                raise ValueError("cuLA KDA prefix metadata does not match the batch")
            if states is None:
                if any(prefixes):
                    raise ValueError(
                        "A cached KDA prefix requires recurrent cache storage"
                    )
                count = len(prefixes)
                logical = attn_inputs.logical_request_count or count
                block_table = [[i + 1] if i < logical else [0] for i in range(count)]
                max_length = max([1] + [b - a for a, b in zip(cu, cu[1:])])
                seq_size_per_block = (max_length + 63) // 64 * 64
                states = torch.zeros(
                    (
                        count + 1,
                        self.local_num_v_heads,
                        self.head_k_dim,
                        self.head_v_dim,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                table = attn_inputs.kv_cache_kernel_block_id
                if table.device.type != "cpu":
                    raise ValueError("cuLA KDA block planning requires host block IDs")
                block_table = table.tolist()
            metadata = prepare_cula_kda_prefill_metadata(
                cu, prefixes, block_table, seq_size_per_block, states
            )
            if attn_meta is not None:
                attn_meta.prefill_cuda_inputs[key] = (
                    cu,
                    prefixes,
                    block_table,
                    metadata,
                    states if kv_cache_tensor is None else None,
                )

        return cula_kda_paged_prefill(
            q,
            k,
            v,
            forget_gate.reshape(shape),
            beta,
            self.alog,
            self.dt_bias,
            self.gate_lower_bound,
            states,
            cu,
            prefixes,
            block_table,
            seq_size_per_block,
            metadata=metadata,
        ).reshape(tokens, -1)


class KdaDecode(KdaBase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        transpose_state_layout: bool = True,
    ):
        super().__init__(linear_attn_config, parallelism_config, weights)
        self.transpose_state_layout = transpose_state_layout

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )

        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
        )
        out = out.transpose(1, 2).reshape(origin_shape)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # Split qkv
        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads,
                self.local_num_k_heads,
                self.local_num_v_heads,
            ],
            dim=2,
        )

        # Compute gate: reshape to [B, S, local_H, D] for fused_recurrent
        forget_gate_2d = forget_gate.reshape(batch * seq, -1)
        g = forget_gate_2d.view(
            batch, seq, self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        # Preserve Kimi-Linear's FP32 producer; K3 fuses sigmoid into recurrence.
        beta_out = beta.reshape(batch, seq, self.local_num_v_heads)
        if not self.transpose_state_layout:
            beta_out = beta_out.float().sigmoid()

        ssm_states = self._get_ssm_states(kv_cache_tensor)

        core_attn_out, _ = fused_recurrent_kda(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            g=g,
            beta=beta_out,
            scale=None,
            initial_state=ssm_states,
            A_log=self.alog,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=self.transpose_state_layout,
            state_v_first=False,
            transpose_state_layout=self.transpose_state_layout,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
        )

        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KdaMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify

        if (
            is_target_verify
            and attn_inputs.linear_replay is not None
            and mixed_qkv.is_cuda
            and is_cuda()
        ):
            query, key, value = torch.split(
                mixed_qkv,
                [
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_v_heads * self.head_v_dim,
                ],
                dim=-1,
            )
            return linear_serial_replay(
                query,
                key,
                value,
                forget_gate,
                beta,
                self.conv_weights,
                self.alog,
                self.dt_bias,
                self._get_ssm_states(kv_cache_tensor),
                self._get_conv_states(kv_cache_tensor),
                kv_cache.linear_replay,
                attn_inputs.linear_replay,
                group_id=kv_cache.group_id,
                vector_gate=True,
                state_v_first=False,
                lower_bound=self.gate_lower_bound,
            )

        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )

        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )
        return attn_out

    def _get_bs_from_attention_input(
        self,
        mixed_qkv: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), f"prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} <=0 when target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), f"token: {token} is not divisible by prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} when target verify"
        b = attention_inputs.prefix_lengths.size(0)
        s = token // b
        return b, s
