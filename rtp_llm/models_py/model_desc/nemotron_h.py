"""NemotronH model descriptor: hybrid Mamba-2 / GQA attention / MoE.

Layer kinds (from ``layers_block_type``): ``mamba`` blocks run a selective
state-space mixer with a depthwise causal conv; ``attention`` blocks are
standard GQA attention; ``moe`` blocks are attention-free routed+shared
experts.  Mamba blocks use the LINEAR cache group (SSM state + conv state
per block); attention blocks use the ``full`` MHA group; moe blocks carry no
cache descriptor.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSResNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import layer_norm_fwd
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.mamba2 import (
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype

logger = logging.getLogger(__name__)


def _cal_block_idx(token_position: int, seq_size_per_block: int) -> int:
    """Block index holding the state *after* processing token_position."""
    if token_position < 0:
        return 0
    return token_position // seq_size_per_block


@dataclass
class MambaChunkMetadata:
    """Per-forward chunk-scan metadata shared by all mamba layers.

    Chunks are cut inside each sequence at ``chunk_size`` granularity in the
    flattened token stream; sequences change at chunk boundaries only
    (pseudo chunks), matching the vLLM/mamba-ssm SSD convention.
    """

    cu_chunk_seqlens: torch.Tensor  # [nchunks + 1] int32, device
    last_chunk_indices: torch.Tensor  # [batch] int32, device
    seq_idx: torch.Tensor  # [nchunks] int32, device


@dataclass
class NemotronHMetadata:
    prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None
    is_target_verify: bool = False
    is_cuda_graph: bool = False
    chunk_meta_cache: Dict[int, MambaChunkMetadata] = field(default_factory=dict)

    def get_chunk_meta(
        self, cu_seqlens_device: torch.Tensor, chunk_size: int
    ) -> MambaChunkMetadata:
        key = (id(cu_seqlens_device), chunk_size)
        meta = self.chunk_meta_cache.get(key)
        if meta is None:
            meta = build_mamba_chunk_metadata(cu_seqlens_device, chunk_size)
            self.chunk_meta_cache[key] = meta
        return meta


def build_mamba_chunk_metadata(
    cu_seqlens_device: torch.Tensor, chunk_size: int
) -> MambaChunkMetadata:
    cu_seqlens = cu_seqlens_device.detach().cpu().tolist()
    cu_chunk = [0]
    last_chunk_indices = []
    seq_idx = []
    for seq_id in range(len(cu_seqlens) - 1):
        start = cu_seqlens[seq_id]
        length = cu_seqlens[seq_id + 1] - start
        if length == 0:
            # Padding rows without tokens: point at the running last chunk.
            last_chunk_indices.append(max(len(cu_chunk) - 2, 0))
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            cu_chunk.append(start + min((i + 1) * chunk_size, length))
            seq_idx.append(seq_id)
        # Index of this sequence's last chunk (cu_chunk has nchunks + 1
        # entries; the last appended chunk sits at len - 2).
        last_chunk_indices.append(len(cu_chunk) - 2)
    device = cu_seqlens_device.device
    return MambaChunkMetadata(
        cu_chunk_seqlens=torch.tensor(cu_chunk, dtype=torch.int32, device=device),
        last_chunk_indices=torch.tensor(
            last_chunk_indices, dtype=torch.int32, device=device
        ),
        seq_idx=torch.tensor(seq_idx, dtype=torch.int32, device=device),
    )


class Mamba2Mixer(nn.Module):
    """Mamba-2 SSM mixer.

    in_proj emits [B; C; x; gate; dt] (rows reordered at load time from the
    HF [gate; x; B; C; dt] layout) so the conv inputs [B; C; x] match the
    framework's [q; k; v] linear-attention cache convention:
    q == B and k == C use the key heads (SSM groups), v == x uses the value
    heads (SSM heads).  The SSM state is (heads, headdim, dstate) ==
    (value heads, value dim, key dim).
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.chunk_size = chunk_size
        self.head_dim: int = linear_attn_config.linear_value_head_dim
        self.dstate: int = linear_attn_config.linear_key_head_dim
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.nheads: int = linear_attn_config.linear_num_value_heads // attn_tp_size
        self.ngroups: int = linear_attn_config.linear_num_key_heads // attn_tp_size
        self.d_inner: int = self.nheads * self.head_dim
        self.bc_size: int = self.ngroups * self.dstate
        self.qkv_size: int = self.bc_size * 2 + self.d_inner
        self.ssm_state_size: int = self.nheads * self.head_dim * self.dstate
        self.ssm_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.ssm_state_dtype
        )
        self.conv_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.conv_state_dtype
        )
        self.linear_conv_kernel_dim: int = linear_attn_config.linear_conv_kernel_dim
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.nheads,
            head_v_dim=self.head_dim,
            head_k_dim=self.dstate,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )
        self.in_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_in_proj_w,
            W.linear_attn_in_proj_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_out_w,
            W.linear_attn_out_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.conv_bias = weights.get(W.linear_attn_conv1d_b, None)
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.A_log = weights[W.linear_attn_alog]
        self.A = -torch.exp(self.A_log.float())
        self.D = weights[W.linear_attn_d]
        # selective_state_update kernel operand views: A/dt_bias/D are
        # per-head scalars expanded to (heads, headdim[, dstate]) with zero
        # strides, which the kernel detects as the tied-dim (TIE_HDIM) path.
        self._A_view = self.A.view(self.nheads, 1, 1).expand(
            self.nheads, self.head_dim, self.dstate
        )
        self._dt_bias_view = self.dt_bias.view(self.nheads, 1).expand(
            self.nheads, self.head_dim
        )
        self._D_view = self.D.view(self.nheads, 1).expand(self.nheads, self.head_dim)
        self.norm_weight = weights[W.linear_attn_norm_w]
        # Mamba-2 gated norm groups the d_inner features into ngroups blocks.
        self.norm_group_size = self.d_inner // max(self.ngroups, 1)
        self.eps = layernorm_eps

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: NemotronHMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor = kv_cache.kv_cache_base.reshape(kv_cache.kv_cache_base.shape[0], -1) if kv_cache is not None else None
        seq_size_per_block = (
            kv_cache.seq_size_per_block if kv_cache is not None else 1
        )
        projected = self.in_proj(hidden_states)
        conv_part, gate, dt = torch.split(
            projected,
            [self.qkv_size, self.d_inner, self.nheads],
            dim=-1,
        )
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            scan_output = self._prefill(
                conv_part, dt, attention_inputs, kv_cache_tensor, seq_size_per_block, attn_meta
            )
        else:
            scan_output = self._decode(
                conv_part, dt, attention_inputs, kv_cache_tensor, seq_size_per_block
            )
        # Mamba-2 gated group RMSNorm: fp32(x * silu(gate)) -> per-group RMS
        # -> * weight (gate applied BEFORE the norm, unlike GDN).
        out = layer_norm_fwd(
            scan_output,
            self.norm_weight,
            None,
            self.eps,
            z=gate,
            out=None,
            group_size=self.norm_group_size,
            norm_before_gate=False,
            is_rms_norm=True,
            activation="silu",
        )[0]
        return self.out_proj(out)

    def _prefill(
        self,
        conv_part: torch.Tensor,
        dt: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_meta: NemotronHMetadata,
    ) -> torch.Tensor:
        cu_seqlens = attn_inputs.cu_seqlens_device
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        conv_out = causal_conv1d_fn(
            x=conv_part.transpose(0, 1),
            weight=self.conv_weights,
            bias=self.conv_bias,
            conv_states=conv_states,
            query_start_loc=cu_seqlens,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_device,
            metadata=attn_meta.prefill_conv1d_meta,
        ).transpose(0, 1)
        b, c, x = torch.split(
            conv_out, [self.bc_size, self.bc_size, self.d_inner], dim=-1
        )
        # [T, ngroups, dstate] and [T, nheads, headdim]
        b = b.view(-1, self.ngroups, self.dstate)
        c = c.view(-1, self.ngroups, self.dstate)
        x = x.view(-1, self.nheads, self.head_dim)

        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        batch = attn_inputs.input_lengths.shape[0]
        initial_states = None
        if ssm_states is not None:
            initial_states = torch.empty(
                batch,
                self.nheads,
                self.head_dim,
                self.dstate,
                device=x.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )
        chunk_meta = attn_meta.get_chunk_meta(cu_seqlens, self.chunk_size)
        out = torch.empty_like(x)
        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=self.A,
            B=b,
            C=c,
            chunk_size=self.chunk_size,
            cu_seqlens=cu_seqlens,
            cu_chunk_seqlens=chunk_meta.cu_chunk_seqlens,
            last_chunk_indices=chunk_meta.last_chunk_indices,
            seq_idx=chunk_meta.seq_idx,
            out=out,
            D=self.D,
            z=None,
            dt_bias=self.dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            return_intermediate_states=ssm_states is not None,
            state_dtype=self.ssm_state_dtype,
        )
        if ssm_states is not None:
            # varlen_states carries per-chunk states (nchunks, H, V, K);
            # the last chunk of each sequence holds its final state.
            h = varlen_states
            final_state = varlen_states[chunk_meta.last_chunk_indices]
            store_ssm_state_to_block_map(
                h,
                final_state,
                attn_inputs.prefix_lengths,
                cu_seqlens,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=self.chunk_size,
            )
        return out.view(-1, self.d_inner)

    def _decode(
        self,
        conv_part: torch.Tensor,
        dt: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        # Decode input is [num_tokens, qkv]: one token per request in normal
        # decode; target verify carries several tokens per request.
        num_requests = (
            attn_inputs.sequence_lengths_plus_1_device.shape[0]
        )
        seq = max(1, conv_part.shape[0] // max(num_requests, 1))
        batch = num_requests
        origin_shape = conv_part.shape
        conv_out = causal_conv1d_update(
            conv_part.reshape(batch, seq, -1).transpose(1, 2),
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=self.conv_bias,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
        )
        conv_out = conv_out.transpose(1, 2).reshape(origin_shape)
        b, c, x = torch.split(
            conv_out, [self.bc_size, self.bc_size, self.d_inner], dim=-1
        )

        ssm_states = self._get_ssm_states(kv_cache_tensor)
        # Read the state of the block that holds the last processed token;
        # write the advanced state to the current token's block.
        sequence_lengths_plus_1 = attn_inputs.sequence_lengths_plus_1_device
        block_map = attn_inputs.kv_cache_kernel_block_id_device
        num_requests = sequence_lengths_plus_1.shape[0]
        lengths = (sequence_lengths_plus_1 - 1).long()  # tokens processed before
        read_positions = lengths - 1
        read_ids = block_map.gather(
            1,
            (read_positions.clamp(min=0) // seq_size_per_block)
            .clamp(max=block_map.shape[1] - 1)
            .unsqueeze(1),
        ).squeeze(1)
        write_positions = lengths + torch.arange(
            seq, device=x.device, dtype=torch.long
        ).unsqueeze(0).expand(num_requests, -1)
        write_ids = block_map.gather(
            1,
            (write_positions // seq_size_per_block)
            .clamp(max=block_map.shape[1] - 1)
        )
        valid_read = (read_positions >= 0) & (read_ids >= 0)
        read_ids = torch.where(valid_read, read_ids, torch.zeros_like(read_ids))

        if seq == 1:
            x_view = x.view(num_requests, self.nheads, self.head_dim)
            dt_view = dt.view(num_requests, self.nheads, 1).expand_as(x_view)
            out = torch.empty_like(x_view)
            selective_state_update(
                state=ssm_states,
                x=x_view,
                dt=dt_view,
                A=self._A_view,
                B=b.view(num_requests, self.ngroups, self.dstate),
                C=c.view(num_requests, self.ngroups, self.dstate),
                D=self._D_view,
                dt_bias=self._dt_bias_view,
                z=None,
                dt_softplus=True,
                state_batch_indices=read_ids,
                dst_state_batch_indices=write_ids.squeeze(1),
                out=out,
                cu_seqlens=None,
            )
            return out.reshape(-1, self.d_inner)
        # Target verify: varlen decode with per-token destination blocks.
        # The kernel's spec-decode mode writes each token's post-state to its
        # own dst slot; num_accepted_tokens=zeros selects the head state.
        cu_seqlens = torch.arange(
            0, num_requests * seq + 1, seq, device=x.device, dtype=torch.int32
        )
        x_view = x.view(num_requests, seq, self.nheads, self.head_dim).reshape(
            num_requests * seq, self.nheads, self.head_dim
        )
        dt_view = (
            dt.view(num_requests, seq, self.nheads, 1)
            .expand(num_requests, seq, self.nheads, self.head_dim)
            .reshape(num_requests * seq, self.nheads, self.head_dim)
        )
        out = torch.empty_like(x_view)
        selective_state_update(
            state=ssm_states,
            x=x_view,
            dt=dt_view,
            A=self._A_view,
            B=b.view(num_requests, seq, self.ngroups, self.dstate).reshape(
                num_requests * seq, self.ngroups, self.dstate
            ),
            C=c.view(num_requests, seq, self.ngroups, self.dstate).reshape(
                num_requests * seq, self.ngroups, self.dstate
            ),
            D=self._D_view,
            dt_bias=self._dt_bias_view,
            z=None,
            dt_softplus=True,
            state_batch_indices=read_ids.unsqueeze(1),
            dst_state_batch_indices=write_ids,
            out=out,
            num_accepted_tokens=torch.zeros(
                num_requests, dtype=torch.int32, device=x.device
            ),
            cu_seqlens=cu_seqlens,
        )
        return out.reshape(-1, self.d_inner)


class NemotronHAttention(CausalAttention):
    """GQA attention without output gate; weights loaded via AttnAtomicWeight."""


class SharedExpertMLP(nn.Module):
    """Ungated shared expert: up_proj -> ReLU^2 -> down_proj."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.up_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_w1,
            W.ffn_s1,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_w2,
            W.ffn_s2,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = torch.relu(hidden_states).square()
        return self.down_proj(hidden_states)


class NemotronHMoEBlock(nn.Module):
    """Routed experts (up/down + ReLU^2, no gate) plus a shared expert."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        # GenericMoeLayer runs routed experts only for up-only layouts (the
        # framework DenseMLP shared expert assumes a gated w13): the
        # ungated up/down + ReLU^2 shared expert runs locally below.
        self.routed_moe = GenericMoeLayer(
            config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph,
            hw_kernel_config=hw_kernel_config,
        )
        self.shared_expert = SharedExpertMLP(
            config, parallelism_config, weights, config.quant_config, hw_kernel_config
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.routed_moe(hidden_states) + self.shared_expert(hidden_states)


class NemotronHDecoderLayer(nn.Module):
    """Pre-norm block with exactly one mixer (mamba / attention / moe)."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = _block_type(config, layer_idx)
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if self.block_type == "mamba":
            self.mixer = Mamba2Mixer(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                chunk_size=config.mamba_chunk_size
                if hasattr(config, "mamba_chunk_size")
                else 128,
            )
        elif self.block_type == "attention":
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.mixer = NemotronHAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )
        else:
            self.mixer = NemotronHMoEBlock(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Optional[FMHAImplBase],
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: NemotronHMetadata = NemotronHMetadata(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states,
                attention_inputs=attention_inputs,
                kv_cache=kv_cache,
                attn_meta=attn_meta,
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )
        else:
            hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


def _block_type(config: ModelConfig, layer_idx: int) -> str:
    if config.hybrid_attention_config.hybrid_attention_types[layer_idx] == (
        HybridAttentionType.LINEAR
    ):
        return "mamba"
    if layer_idx in set(config.moe_layer_index):
        return "moe"
    return "attention"


def _is_cuda_graph_forward(inputs: PyModelInputs, fmha_impl: Any) -> bool:
    attention_inputs = inputs.attention_inputs
    if attention_inputs is None and inputs.attention_inputs_by_tag is not None:
        attention_inputs = inputs.attention_inputs_by_tag
    if isinstance(attention_inputs, dict):
        return any(
            group_inputs.is_cuda_graph
            for group_inputs in attention_inputs.values()
        )
    if attention_inputs is not None:
        return attention_inputs.is_cuda_graph
    return fmha_impl is not None and getattr(fmha_impl, "is_cuda_graph", False)


class NemotronHModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                NemotronHDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(model_config.num_layers)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma),
            eps=model_config.layernorm_eps,
        )
        self._full_attention_layers = [
            i
            for i in range(model_config.num_layers)
            if _block_type(model_config, i) == "attention"
        ]

    def _get_fmha_group_tags(self):
        if self.kv_cache is None:
            return None
        return get_group_tags_for_layers(self.kv_cache, self._full_attention_layers)

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        return self.embed_tokens(inputs.input_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.word_embedding(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)

        is_cuda_graph = _is_cuda_graph_forward(inputs, fmha_impl)
        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        mamba_layer_idx = next(
            (
                layer_idx
                for layer_idx, layer in enumerate(self.layers)
                if layer.block_type == "mamba"
            ),
            None,
        )
        if mamba_layer_idx is not None and self.kv_cache is not None:
            mamba_attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, mamba_layer_idx
            )
            attention_inputs = mamba_attention_inputs
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            cu_seqlens = attention_inputs.cu_seqlens_device
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlens,
                device=hidden_states.device,
            )
        attn_meta = NemotronHMetadata(
            prefill_conv1d_meta=prefill_conv1d_meta,
            is_target_verify=is_target_verify,
            is_cuda_graph=is_cuda_graph,
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers):
            # Mamba blocks use the linear cache group but no FMHA; moe blocks
            # carry no cache group tags at all.  Only attention blocks select
            # an FMHA implementation.
            layer_is_attention = decoder_layer.block_type == "attention"
            layer_attention_inputs = (
                select_attention_inputs_for_layer(inputs, self.kv_cache, i)
                if self.kv_cache is not None and decoder_layer.block_type != "moe"
                else None
            )
            layer_fmha_impl = (
                select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
                if self.kv_cache is not None and layer_is_attention
                else None
            )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i)
                if self.kv_cache and decoder_layer.block_type != "moe"
                else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )
            capture.capture_layer(i, hidden_states, residual)

        hidden_states, residual = self.norm(hidden_states, residual)
        outputs = capture.finalize(hidden_states)
        return outputs
