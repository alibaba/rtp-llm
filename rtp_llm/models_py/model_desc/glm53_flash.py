"""Kimi Linear model description.

Hybrid architecture:
  - KDA (Kimi Delta Attention) linear attention layers
  - MLA (Multi-head Latent Attention) full attention layers
  - MoE FFN with sigmoid routing (layer 1+), Dense FFN (layer 0)
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.glm53_prefill_parallel import (
    MlaCPParallelismView,
    glm_prefill_moe_chunk_plan,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.distributed.glm53_collective_gemm import (
    all_gather_trim,
    reduce_scatter_glm53,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    TokenShardLayout,
    shard_tokens,
    token_shard_layout,
)
from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.glm53_moe import DecodeLayerOutput, GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.dsv4.forward_metadata import (
    cached_prefill_prefix_summary,
    metadata_cache,
    scoped_forward_metadata,
    tensor_key,
)
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.glm53.norm import Glm53RMSNorm as RMSNorm
from rtp_llm.models_py.modules.glm53.norm import Glm53RMSResNorm as RMSResNorm
from rtp_llm.models_py.modules.glm53_mega_mhc import (
    can_use_mega_mhc,
    fused_post_pre_rms,
    prepare_mega_mhc_stream,
)
from rtp_llm.models_py.modules.glm53_mhc import fused_post_pre as glm53_fused_post_pre
from rtp_llm.models_py.modules.hybrid.glm53_mla_attention import MlaAttention
from rtp_llm.models_py.triton_kernels.causal_conv1d.glm53_causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.kda_checkpoint import (
    get_cula_checkpoint_kernel,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_kda_chunk_states,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state import (
    DEFAULT_KDA_CHUNK_SIZE,
    KDARecurrentCheckpointMetadata,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
    prepare_kda_recurrent_checkpoint_metadata,
    store_kda_recurrent_checkpoints,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    AttentionConfigs,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype


class KimiLinearMetadata(object):

    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        self.kda_checkpoint_meta: Optional[KDARecurrentCheckpointMetadata] = None
        self.kda_checkpoint_states: Optional[torch.Tensor] = None
        self.kda_cu_seqlens_host: Optional[torch.Tensor] = None
        self.kda_checkpoint_page_size = 0
        self.token_shard: Optional[TokenShardLayout] = None
        self.mla_cp_layout: Optional[ZigzagTokenLayout] = None

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


def _linear_attn_tp_size(parallelism_config: ParallelismConfig) -> int:
    # Context parallelism reuses the TP process group for sequence sharding,
    # so linear-attention heads and their state cache remain unsharded.
    return int(parallelism_config.get_attn_tp_size())


class KimiLinearKDABase(nn.Module):
    """Base class for KDA (Kimi Delta Attention) prefill/decode.

    KDA differs from GDN in:
      - qkv_size = q + k + v (no z component, since output gate is separate LoRA)
      - beta is [B, H] scalar per head (from b_proj), not packed with a
      - forget gate g is [B, H, D] per-dim vector (from LoRA + fused_kda_gate)
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.gate_lower_bound = gate_lower_bound
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        attn_tp_size = _linear_attn_tp_size(parallelism_config)
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // attn_tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
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
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class KimiLinearKDAPrefill(KimiLinearKDABase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__(
            linear_attn_config, parallelism_config, weights, gate_lower_bound
        )
        self.conv_output_groups = (
            3
            if gate_lower_bound is not None
            and self.local_num_k_heads == self.local_num_v_heads
            and self.head_k_dim == self.head_v_dim
            else 1
        )

    @staticmethod
    def _host_lengths(
        attn_inputs: PyAttentionInputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_count = int(attn_inputs.input_lengths.numel())
        cu_seqlens_host = getattr(attn_inputs, "cu_seqlens", None)
        if cu_seqlens_host is None or cu_seqlens_host.numel() < sequence_count + 1:
            input_lengths_host = attn_inputs.input_lengths.cpu().to(dtype=torch.int32)
            cu_seqlens_host = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32),
                    torch.cumsum(input_lengths_host, dim=0, dtype=torch.int32),
                )
            ).contiguous()
        else:
            cu_seqlens_host = (
                cu_seqlens_host[: sequence_count + 1].to(dtype=torch.int32).contiguous()
            )
            input_lengths_host = torch.diff(cu_seqlens_host)
        prefix_lengths_host = (
            attn_inputs.prefix_lengths.cpu().to(dtype=torch.int32).contiguous()
        )
        return cu_seqlens_host, input_lengths_host, prefix_lengths_host

    def _cula_checkpoint_prefill(
        self,
        kernel,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        initial_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: torch.Tensor,
        checkpoint_interval: int,
        checkpoint_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run cuLA once and publish page-aligned recurrent checkpoints."""

        if not query.is_cuda:
            raise RuntimeError("GLM5 cuLA KDA prefill requires CUDA")
        if checkpoint_interval <= 0:
            raise ValueError("cuLA KDA checkpoint interval must be positive")
        if initial_states.dtype != torch.float32 or not initial_states.is_contiguous():
            raise ValueError("cuLA KDA initial state must be contiguous FP32")
        if (
            checkpoint_states.dtype != torch.float32
            or not checkpoint_states.is_contiguous()
        ):
            raise ValueError("cuLA KDA checkpoints must be contiguous FP32")
        packed_cu_seqlens = (
            None if int(cu_seqlens.numel()) == 2 else cu_seqlens.contiguous()
        )
        with torch.inference_mode():
            output, final_state, published_checkpoints = kernel(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                raw_gate.to(dtype=query.dtype).contiguous(),
                raw_beta.to(dtype=query.dtype).contiguous(),
                scale=self.head_k_dim**-0.5,
                initial_state=initial_states,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=packed_cu_seqlens,
                cu_seqlens_cpu=(None if packed_cu_seqlens is None else cu_seqlens_host),
                safe_gate=True,
                lower_bound=float(self.gate_lower_bound),
                disable_recompute=False,
                use_intracard_cp=False,
                A_log=self.alog.float().contiguous(),
                dt_bias=self.dt_bias.float().contiguous(),
                checkpoint_interval=checkpoint_interval,
                checkpoint_states=checkpoint_states,
            )
        if (
            published_checkpoints is None
            or published_checkpoints.data_ptr() != checkpoint_states.data_ptr()
        ):
            raise RuntimeError("cuLA did not publish into the checkpoint buffer")
        if final_state is not None:
            raise RuntimeError("cuLA unexpectedly returned a final state")
        return output.to(dtype=query.dtype)

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
            output_groups=self.conv_output_groups,
        )
        return out if self.conv_output_groups > 1 else out.transpose(0, 1)

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        # Compute gate: [T, local_H*D] -> [1, T, local_H, D]
        g = forget_gate.view(
            1, forget_gate.shape[0], self.local_num_v_heads, self.head_k_dim
        ).contiguous()
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
                dtype=torch.float32,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        if self.conv_output_groups == 3:
            query, key, value = mixed_qkv.unbind(0)
        else:
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

        cula_kernel = get_cula_checkpoint_kernel(
            query.device,
            query.dtype,
            self.head_k_dim,
            self.ssm_state_dtype,
            self.gate_lower_bound,
            seq_size_per_block,
        )
        if cula_kernel is not None:
            if initial_states is None:
                initial_states = torch.zeros(
                    context_batch_size,
                    self.local_num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    device=mixed_qkv.device,
                    dtype=torch.float32,
                )
            if (
                attn_meta.kda_checkpoint_meta is None
                or attn_meta.kda_checkpoint_page_size != seq_size_per_block
            ):
                cu_seqlens_host, input_lengths_host, prefix_lengths_host = (
                    self._host_lengths(attn_inputs)
                )
                attn_meta.kda_checkpoint_meta = (
                    prepare_kda_recurrent_checkpoint_metadata(
                        input_lengths_host,
                        prefix_lengths_host,
                        seq_size_per_block,
                        mixed_qkv.device,
                    )
                )
                attn_meta.kda_checkpoint_states = torch.empty(
                    1,
                    attn_meta.kda_checkpoint_meta.total_checkpoints,
                    self.local_num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    device=mixed_qkv.device,
                    dtype=torch.float32,
                )
                attn_meta.kda_cu_seqlens_host = cu_seqlens_host
                attn_meta.kda_checkpoint_page_size = seq_size_per_block
            assert attn_meta.kda_checkpoint_states is not None
            assert attn_meta.kda_cu_seqlens_host is not None
            attn_out = self._cula_checkpoint_prefill(
                cula_kernel,
                query,
                key,
                value,
                g,
                beta.unsqueeze(0),
                initial_states.contiguous(),
                cu_seqlens=cu_seqlens_without_padding,
                cu_seqlens_host=attn_meta.kda_cu_seqlens_host,
                checkpoint_interval=seq_size_per_block,
                checkpoint_states=attn_meta.kda_checkpoint_states,
            )
            if ssm_states is not None:
                store_kda_recurrent_checkpoints(
                    attn_meta.kda_checkpoint_states,
                    attn_meta.kda_checkpoint_meta,
                    attn_inputs.kv_cache_kernel_block_id_device,
                    ssm_states,
                )
            return attn_out.squeeze(0)

        if ssm_states is not None:
            cache_alignment = seq_size_per_block
            if self.parallelism_config.role_type in (RoleType.PREFILL, RoleType.DECODE):
                cache_alignment *= int(self.parallelism_config.get_attn_tp_size())
            if cache_alignment % DEFAULT_KDA_CHUNK_SIZE:
                raise ValueError(
                    f"KDA chunk size {DEFAULT_KDA_CHUNK_SIZE} must divide "
                    f"the linear cache alignment {cache_alignment}"
                )
        attn_out, final_state, h_from_chunk = chunk_kda(
            query,
            key,
            value,
            g,
            beta.float().sigmoid().unsqueeze(0),
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=self.alog,
            dt_bias=self.dt_bias,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            chunk_size=DEFAULT_KDA_CHUNK_SIZE,
        )

        store_kda_chunk_states(
            h_from_chunk,
            final_state,
            ssm_states,
            attn_inputs,
            seq_size_per_block,
            chunk_size=DEFAULT_KDA_CHUNK_SIZE,
        )
        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            attn_meta,
        )
        return attn_out


class KimiLinearKDADecode(KimiLinearKDABase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__(
            linear_attn_config, parallelism_config, weights, gate_lower_bound
        )
        self.fuse_decode = gate_lower_bound == -5.0
        self._verify_fusion_logged = False

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
        qkv: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
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
        query, key, value = (
            qkv
            if qkv is not None
            else torch.split(
                mixed_qkv,
                [
                    self.local_num_k_heads,
                    self.local_num_k_heads,
                    self.local_num_v_heads,
                ],
                dim=2,
            )
        )

        # Compute gate: reshape to [B, S, local_H, D] for fused_recurrent
        forget_gate_2d = forget_gate.reshape(batch * seq, -1)
        g = forget_gate_2d.view(
            batch, seq, self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        # beta: [batch*seq, H] -> sigmoid in float32 -> [batch, seq, H]
        beta_out = beta.reshape(batch * seq, -1)
        if qkv is None:
            beta_out = beta_out.float().sigmoid()
        beta_out = beta_out.view(batch, seq, self.local_num_v_heads)

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
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=qkv is not None,
            decode_low_warps=True,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
            lower_bound=self.gate_lower_bound,
            store_states=True,
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
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify

        if is_target_verify and self.fuse_decode:
            from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state.short_conv import (
                kda_short_conv_verify,
                kda_short_conv_verify_supported,
            )

            batch, tokens = self._get_bs_from_attention_input(
                mixed_qkv, attn_inputs, True
            )
            conv_states = self._get_conv_states(kv_cache_tensor)
            supported = kda_short_conv_verify_supported(
                mixed_qkv,
                self.conv_weights,
                conv_states,
                kv_cache.seq_size_per_block,
                batch,
                tokens,
            )
            if supported:
                if not self._verify_fusion_logged:
                    logging.info(
                        "[KDA fusion] verify convolution enabled width=%d",
                        tokens,
                    )
                    self._verify_fusion_logged = True
                qkv = kda_short_conv_verify(
                    mixed_qkv,
                    self.conv_weights,
                    self._get_conv_states(kv_cache_tensor),
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.sequence_lengths_plus_1_device,
                    kv_cache.seq_size_per_block,
                    batch,
                    tokens,
                )
                qkv = tuple(
                    t.view(batch, tokens, self.local_num_k_heads, self.head_k_dim)
                    for t in qkv
                )
                return self._fla(
                    mixed_qkv,
                    forget_gate,
                    beta,
                    kv_cache_tensor,
                    kv_cache.seq_size_per_block,
                    attn_inputs,
                    True,
                    qkv=qkv,
                )

        if (
            self.fuse_decode
            and not is_target_verify
            and self.linear_conv_kernel_dim == 4
            and self.local_num_k_heads == self.local_num_v_heads
            and mixed_qkv.stride(-1) == 1
            and mixed_qkv.dtype == torch.bfloat16
            and self.conv_weights.dtype in (torch.bfloat16, torch.float32)
            and self.conv_state_dtype in (torch.bfloat16, torch.float32)
        ):
            from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state.short_conv import (
                kda_short_conv_decode,
            )

            qkv = kda_short_conv_decode(
                mixed_qkv,
                self.conv_weights,
                self._get_conv_states(kv_cache_tensor),
                attn_inputs.kv_cache_kernel_block_id_device,
                attn_inputs.sequence_lengths_plus_1_device,
                kv_cache.seq_size_per_block,
            )
            qkv = tuple(
                t.view(-1, 1, self.local_num_k_heads, self.head_k_dim) for t in qkv
            )
            return self._fla(
                mixed_qkv,
                forget_gate,
                beta,
                kv_cache_tensor,
                kv_cache.seq_size_per_block,
                attn_inputs,
                False,
                qkv=qkv,
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


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) module.

    Forward flow:
      1. qkv = in_proj_qkv(hidden)           # merged q+k+v projection
      2. beta_input = in_proj_b(hidden)       # beta gate input [token, H]
      3. forget_gate = f_b_proj(f_a_proj(hidden))  # LoRA forget gate [token, H*D]
      4. g_proj = g_b_proj(g_a_proj(hidden))  # LoRA output gate [token, H*D]
      5. conv1d on qkv
      6. FLA attention (chunk_kda for prefill, fused_recurrent_kda for decode)
         - fused_kda_gate applies: g = -exp(A_log) * softplus(forget_gate + dt_bias)
         - beta = sigmoid(beta_input)
      7. o_norm(attn_out, g_proj) with sigmoid activation
      8. o_proj(result) + all_reduce
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config

        # Projections
        self.in_proj_qkv = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkv_w, None, None, quant_config
        )
        self.in_proj_b = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_b_w, None, None, quant_config
        )
        # LoRA forget gate
        self.f_a_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_f_a_w, None, None, quant_config
        )
        self.f_b_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_f_b_w, None, None, quant_config
        )
        # LoRA output gate
        self.g_a_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_g_a_w, None, None, quant_config
        )
        self.g_b_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_g_b_w, None, None, quant_config
        )

        self.register_buffer("packed_input_weight", None, persistent=False)
        self.packed_input_widths = None
        self.fuse_input_projection = gate_lower_bound == -5.0
        if gate_lower_bound is not None and (
            parallelism_config.role_type == RoleType.PREFILL
            or self.fuse_input_projection
        ):
            from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                canonical_bf16_weight,
            )

            packed_parts = [
                canonical_bf16_weight(proj) for proj in self.input_projections()
            ]
            if all(part is not None for part in packed_parts):
                self.packed_input_widths = tuple(part.shape[1] for part in packed_parts)
                # A 128-column row pitch keeps each gathered GEMM tile aligned.
                width = sum(self.packed_input_widths)
                padded_width = (width + 127) // 128 * 128
                packed = packed_parts[0].new_zeros(
                    (packed_parts[0].shape[0], padded_width)
                )
                offset = 0
                for part in packed_parts:
                    packed[:, offset : offset + part.shape[1]].copy_(part)
                    offset += part.shape[1]
                self.packed_input_weight = packed

        self.fuse_input_projection = (
            self.fuse_input_projection and self.packed_input_weight is not None
        )

        self.head_k_dim = linear_attn_config.linear_key_head_dim
        if self.fuse_input_projection:
            logging.info(
                "[KDA fusion] merged QKV/beta/F_A/G_A input projection enabled"
            )
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads
            // _linear_attn_tp_size(parallelism_config)
        )

        self.prefill_kda = KimiLinearKDAPrefill(
            linear_attn_config, parallelism_config, weights, gate_lower_bound
        )
        self.decode_kda = KimiLinearKDADecode(
            linear_attn_config, parallelism_config, weights, gate_lower_bound
        )
        # o_norm with sigmoid activation (not SwiGLU)
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
            activation="sigmoid",
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_out_w, None, None, quant_config
        )

    def input_projections(self):
        return (self.in_proj_qkv, self.in_proj_b, self.f_a_proj, self.g_a_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
        input_is_sharded: bool = False,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
        ), "prefill_conv1d_meta is required for prefill"

        # 1. Projections. The SP caller may leave the input rank-local for AG/GEMM.
        is_context = attention_inputs.is_prefill and not attn_meta.is_target_verify
        use_packed_projection = self.packed_input_weight is not None and (
            (self.parallelism_config.role_type == RoleType.PREFILL and is_context)
            or (self.fuse_input_projection and not is_context)
        )
        if input_is_sharded:
            if use_packed_projection:
                from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                    all_gather_packed_kda_projections,
                )

                projected_qkv, beta_input, forget_low, gate_low = (
                    all_gather_packed_kda_projections(
                        hidden_states,
                        self.input_projections(),
                        self.packed_input_weight,
                        self.packed_input_widths,
                        attn_meta.token_shard.logical_tokens,
                    )
                )
            else:
                from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                    all_gather_projections,
                )

                projected_qkv, beta_input, forget_low, gate_low = (
                    all_gather_projections(
                        hidden_states,
                        self.input_projections(),
                        attn_meta.token_shard.logical_tokens,
                    )
                )
            forget_gate = self.f_b_proj(forget_low)
            g_proj = self.g_b_proj(gate_low)
            del forget_low, gate_low
        elif use_packed_projection:
            # The caller already completed AG when communication/GEMM overlap
            # is unsuitable. Packing the four GEMMs is independent of that
            # choice. Keep combined-role context projection on its original
            # path; this switch targets decode/verify only.
            from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                packed_kda_projections,
            )

            projected_qkv, beta_input, forget_low, gate_low = packed_kda_projections(
                hidden_states, self.packed_input_weight, self.packed_input_widths
            )
            forget_gate = self.f_b_proj(forget_low)
            g_proj = self.g_b_proj(gate_low)
            del forget_low, gate_low
        else:
            projected_qkv = self.in_proj_qkv(hidden_states)
            beta_input = self.in_proj_b(hidden_states)
            forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
            g_proj = self.g_b_proj(self.g_a_proj(hidden_states))

        # 2. Prefill or decode
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            attn_output = self.prefill_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )
        else:
            attn_output = self.decode_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )

        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            del projected_qkv, beta_input, forget_gate

        # 3. o_norm with sigmoid gating: y = RMSNorm(attn_out) * sigmoid(g_proj)
        if self.decode_kda.fuse_decode and (
            not attention_inputs.is_prefill or attn_meta.is_target_verify
        ):
            from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state.rms_norm_gate import (
                kimi_kda_rms_norm_sigmoid_gate,
            )

            attn_output = kimi_kda_rms_norm_sigmoid_gate(
                attn_output.reshape(-1, self.head_v_dim),
                g_proj.reshape(-1, self.head_v_dim),
                self.norm.weight,
                self.norm.eps,
            )
        else:
            attn_output = self.norm(
                attn_output.reshape(-1, self.head_v_dim),
                g_proj.reshape(-1, self.head_v_dim),
            )

        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)

        # 4. Output projection and TP sum, optionally writing directly to RS peers.
        if attn_meta.token_shard is not None:
            from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                project_reduce_scatter,
            )

            scattered = project_reduce_scatter(attn_output, self.out_proj)
            if scattered is not None:
                return scattered
        attn_output = self.out_proj(attn_output)

        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = (
                reduce_scatter_glm53(attn_output)
                if attn_meta.token_shard is not None
                else all_reduce(attn_output, group=Group.TP)
            )

        return attn_output


class KimiLinearDecoderLayer(nn.Module):
    """Kimi Linear decoder layer.

    Hybrid KDA/MLA attention + MoE/Dense FFN.
    Uses RMSResNorm (fused residual add + layernorm) like GenericMoeDecoderLayer.
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._drain_pinned_mla_before_moe = (
            layer_idx in config.moe_layer_index and config.mega_moe_policy is not None
        )
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]

        quant_config = config.quant_config

        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = KimiLinearKDA(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                getattr(config, "kda_gate_lower_bound", None),
            )
        else:
            # Full MLA attention layer
            self.self_attn = MlaAttention(
                config.attn_config,
                (
                    MlaCPParallelismView(parallelism_config)
                    if config.prefill_mla_cp
                    else parallelism_config
                ),
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
                global_weights=global_weights,
                indexer_layernorm_eps=1e-6,
            )

        # FFN: Dense (layer 0) or MoE (layer 1+)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )

        self.hc_enabled = config.glm53_hc_mult > 1
        if self.hc_enabled:
            self.input_layernorm = RMSNorm(
                weights[W.pre_ln_gamma], eps=config.layernorm_eps
            )
            self.post_attention_layernorm = RMSNorm(
                weights[W.post_ln_gamma], eps=config.layernorm_eps
            )
            hc_args = dict(
                dim=config.hidden_size,
                hc_mult=config.glm53_hc_mult,
                hc_sinkhorn_iters=config.hc_sinkhorn_iters,
                norm_eps=config.layernorm_eps,
                hc_eps=config.hc_eps,
                layer_id=layer_idx,
            )
            self.attn_hc = build_hc_unit(
                weights[W.v4_hc_attn_fn],
                weights[W.v4_hc_attn_base],
                weights[W.v4_hc_attn_scale],
                name="attn",
                **hc_args,
            )
            self.ffn_hc = build_hc_unit(
                weights[W.v4_hc_ffn_fn],
                weights[W.v4_hc_ffn_base],
                weights[W.v4_hc_ffn_scale],
                name="ffn",
                **hc_args,
            )
        else:
            self.input_layernorm = RMSResNorm(
                weights[W.pre_ln_gamma], eps=config.layernorm_eps
            )
            self.post_attention_layernorm = RMSResNorm(
                weights[W.post_ln_gamma], eps=config.layernorm_eps
            )

    def _wait_pinned_mla_before_moe(self, fmha_impl: FMHAImplBase) -> None:
        if not getattr(self, "_drain_pinned_mla_before_moe", False):
            return
        implementation = fmha_impl
        groups = getattr(implementation, "pinned_mla_groups", None)
        if groups is None:
            implementation = getattr(implementation, "fmha_impl", None)
            groups = getattr(implementation, "pinned_mla_groups", {})
        entry = groups.get(self.layer_idx)
        if entry is not None:
            # Do not restrict this to group_layer == 0: a shared-index group
            # can begin with a dense layer followed by its first MoE layer.
            working, _ = entry
            if getattr(working, "started", True):
                working.wait_prefetch_complete()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: KimiLinearMetadata = KimiLinearMetadata(),
        global_kv_cache: Optional[KVCache] = None,
    ) -> DecodeLayerOutput:
        if self.hc_enabled:
            return self._forward_hc(
                hidden_states,
                fmha_impl,
                kv_cache,
                attention_inputs,
                attn_meta,
                global_kv_cache,
            )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention (KDA or MLA)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                global_kv_cache=global_kv_cache,
            )

        self._wait_pinned_mla_before_moe(fmha_impl)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return DecodeLayerOutput(hidden_states, residual)

    def _forward_hc(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
        global_kv_cache: Optional[KVCache],
    ) -> DecodeLayerOutput:
        if attn_meta.token_shard is not None:
            return self._forward_hc_sequence_parallel(
                hidden_states,
                fmha_impl,
                kv_cache,
                attention_inputs,
                attn_meta,
                global_kv_cache,
            )
        residual = hidden_states
        hidden_states, post, comb = self.attn_hc.pre(residual)
        hidden_states = self.input_layernorm(hidden_states, None)

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                global_kv_cache=global_kv_cache,
            )

        hidden_states = self.attn_hc.post(hidden_states, residual, post, comb)
        residual = hidden_states
        hidden_states, post, comb = self.ffn_hc.pre(residual)
        self._wait_pinned_mla_before_moe(fmha_impl)
        hidden_states = self.post_attention_layernorm(hidden_states, None)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_hc.post(hidden_states, residual, post, comb)
        return DecodeLayerOutput(hidden_states, hidden_states)

    def _forward_hc_sequence_parallel(
        self,
        hidden_states,
        fmha_impl,
        kv_cache,
        attention_inputs,
        attn_meta,
        global_kv_cache,
    ) -> DecodeLayerOutput:
        layout = attn_meta.token_shard
        cp_layout = attn_meta.mla_cp_layout
        residual = hidden_states
        hidden_states, post, comb = self.attn_hc.pre(residual)
        hidden_states = self.input_layernorm(hidden_states, None)
        if self.layer_type == HybridAttentionType.LINEAR:
            from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                can_fuse_input,
            )

            fuse_ag = can_fuse_input(
                self.self_attn.input_projections(), layout.logical_tokens
            )
            if not fuse_ag:
                hidden_states = all_gather_trim(
                    hidden_states, layout.logical_tokens, Group.TP
                )
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
                input_is_sharded=fuse_ag,
            )
        elif cp_layout is not None:
            hidden_states = cp_layout.sp_to_cp(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                global_kv_cache=global_kv_cache,
            )
            hidden_states = cp_layout.cp_to_sp(hidden_states)
        else:
            hidden_states = all_gather_trim(
                hidden_states, layout.logical_tokens, Group.TP
            )
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                global_kv_cache=global_kv_cache,
                reduce_scatter_output=True,
            )
        hidden_states = self.attn_hc.post(hidden_states, residual, post, comb)
        residual = hidden_states
        hidden_states, post, comb = self.ffn_hc.pre(residual)
        self._wait_pinned_mla_before_moe(fmha_impl)
        hidden_states = self.post_attention_layernorm(hidden_states, None)
        if isinstance(self.mlp, GenericMoeLayer):
            hidden_states = self.mlp(hidden_states, sequence_parallel_layout=layout)
        else:
            hidden_states = all_gather_trim(
                hidden_states, layout.logical_tokens, Group.TP
            )
            hidden_states = self.mlp(hidden_states, skip_allreduce=True)
            hidden_states = reduce_scatter_glm53(hidden_states)
        hidden_states = self.ffn_hc.post(hidden_states, residual, post, comb)
        return DecodeLayerOutput(hidden_states, hidden_states)

    def forward_hc_deferred(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
        global_kv_cache: Optional[KVCache],
        residual: Optional[torch.Tensor],
        post: Optional[torch.Tensor],
        comb: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode HC path with the final FFN writeback deferred.

        The previous layer's FFN output is fused with this layer's attention
        pre. Within the layer, attention writeback is fused with FFN pre. The
        model loop materializes only the first attention pre and final FFN
        post as standalone kernels.
        """

        norm_done = False
        if post is None:
            residual = hidden_states
            hidden_states, post, comb = self.attn_hc.pre(residual)
        else:
            if residual is None or comb is None:
                raise RuntimeError(
                    "deferred mHC state requires residual/post/comb together"
                )
            if can_use_mega_mhc(
                hidden_states, residual, post, comb, self.attn_hc, self.input_layernorm
            ):
                residual, hidden_states, post, comb = fused_post_pre_rms(
                    hidden_states,
                    residual,
                    post,
                    comb,
                    self.attn_hc,
                    self.input_layernorm,
                )
                norm_done = True
            else:
                residual, hidden_states, post, comb = glm53_fused_post_pre(
                    hidden_states, residual, post, comb, self.attn_hc
                )
        if not norm_done:
            hidden_states = self.input_layernorm(hidden_states, None)

        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                global_kv_cache=global_kv_cache,
            )

        assert residual is not None and post is not None and comb is not None
        prepacked_mega = None
        if can_use_mega_mhc(
            hidden_states,
            residual,
            post,
            comb,
            self.ffn_hc,
            self.post_attention_layernorm,
        ):
            if hasattr(self.mlp, "prepare_fused_mhc_inputs"):
                prepacked_mega = self.mlp.prepare_fused_mhc_inputs(hidden_states)
            residual, hidden_states, post, comb = fused_post_pre_rms(
                hidden_states,
                residual,
                post,
                comb,
                self.ffn_hc,
                self.post_attention_layernorm,
                mega_inputs=prepacked_mega,
            )
            self._wait_pinned_mla_before_moe(fmha_impl)
        else:
            residual, hidden_states, post, comb = glm53_fused_post_pre(
                hidden_states, residual, post, comb, self.ffn_hc
            )
            self._wait_pinned_mla_before_moe(fmha_impl)
            hidden_states = self.post_attention_layernorm(hidden_states, None)
        if prepacked_mega is not None:
            hidden_states = self.mlp(hidden_states, prepacked_mega=prepacked_mega)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, post, comb


class Glm53FlashModel(GptModelBase):
    prepare_cuda_graph_stream = staticmethod(prepare_mega_mhc_stream)

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
        )
        self.prefill_mla_cp = model_config.prefill_mla_cp
        self.prefill_sequence_parallel = self.prefill_mla_cp
        if self.prefill_sequence_parallel and model_config.glm53_hc_mult <= 1:
            raise ValueError("GLM53 Prefill sequence parallel requires mHC")
        self.mla_parallelism = (
            MlaCPParallelismView(parallelism_config)
            if self.prefill_mla_cp
            else parallelism_config
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = (
            MultimodalEmbeddingInjector()
            if model_config.mm_model_config.is_multimodal
            else None
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        # PDFUSION can mix eager prefill and decode on EP peers in one round.
        # Both phases must take the same path at the graph/eager boundary.
        self.requires_ep_sync = any(
            getattr(layer.mlp, "requires_ep_sync", False) for layer in self.layers
        )
        self.synchronize_cuda_graph_choice = (
            parallelism_config.role_type == RoleType.PDFUSION
            and parallelism_config.dp_size > 1
            and self.requires_ep_sync
        )
        self.hc_enabled = model_config.glm53_hc_mult > 1
        if self.hc_enabled:
            self.norm = RMSNorm(
                weights.get_global_weight(W.final_ln_gamma),
                eps=model_config.layernorm_eps,
            )
        else:
            self.norm = RMSResNorm(
                weights.get_global_weight(W.final_ln_gamma),
                eps=model_config.layernorm_eps,
            )

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        if not super().initialize(init_resource):
            return False
        if self.prefill_sequence_parallel:
            from rtp_llm.models_py.distributed.collective_torch import get_process_group
            from rtp_llm.models_py.distributed.glm53_collective_gemm import (
                configure_glm53_collective_gemm,
            )

            configure_glm53_collective_gemm(
                get_process_group(Group.TP),
                self.config.hidden_size,
                self.parallelism_config,
                max_batch_tokens=getattr(init_resource, "max_batch_tokens_size", 0),
                enable_cuda_graph=(
                    self.py_hw_kernel_config.enable_cuda_graph
                    if self.py_hw_kernel_config is not None
                    else True
                ),
            )
        return True

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        from rtp_llm.models_py.distributed.glm53_mla_cp import prepare_glm53_fmha

        return prepare_glm53_fmha(self, inputs, is_cuda_graph)

    @scoped_forward_metadata
    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        group_inputs = get_attention_inputs_value(inputs)
        attention_inputs = select_attention_inputs_for_tag(group_inputs, "full")
        linear_inputs = select_attention_inputs_for_tag(group_inputs, "linear")
        if attention_inputs.is_prefill and not attention_inputs.is_target_verify:
            cached_prefill_prefix_summary(
                attention_inputs.prefix_lengths,
            )
        input_ids: torch.Tensor = inputs.input_ids
        multimodal_embedding_injector = getattr(
            self, "multimodal_embedding_injector", None
        )
        if multimodal_embedding_injector is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = self.embed_tokens(
                input_ids,
                inputs.combo_position_ids,
                inputs.embedding_inputs.combo_tokens_type_ids,
                inputs.embedding_inputs.text_tokens_mask,
            )
            inputs_embeds = multimodal_embedding_injector(
                inputs_embeds,
                inputs.multimodal_inputs.multimodal_features,
                inputs.multimodal_inputs.mm_features_locs,
            )
        hidden_states = inputs_embeds

        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            cu_seqlen_without_padding = linear_inputs.cu_seqlens_device
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
                query_start_loc_cpu=linear_inputs.cu_seqlens,
            )
            cache = metadata_cache()
            if cache is not None:
                boundaries = prefill_conv1d_meta.sequence_boundaries_host
                cache[("prefill_cu_host", tensor_key(cu_seqlen_without_padding))] = (
                    boundaries
                )
                cache[
                    (
                        "kda_sequence_boundaries",
                        tensor_key(cu_seqlen_without_padding),
                        boundaries[-1],
                    )
                ] = boundaries

        attn_meta = KimiLinearMetadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        fmha_impl.glm53_group_inputs = group_inputs

        if (
            self.prefill_sequence_parallel
            and attention_inputs.is_prefill
            and not is_target_verify
        ):
            attn_meta.token_shard = token_shard_layout(
                hidden_states.shape[0],
                self.parallelism_config.tp_size,
                self.parallelism_config.tp_rank,
            )
            if self.prefill_mla_cp:
                attn_meta.mla_cp_layout = fmha_impl.glm53_cp_layout
            hidden_states = shard_tokens(hidden_states, attn_meta.token_shard)
        del inputs_embeds

        if self.hc_enabled:
            hidden_states = (
                hidden_states.unsqueeze(1)
                .expand(-1, self.config.glm53_hc_mult, -1)
                .contiguous()
            )
            residual = hidden_states
        else:
            residual = torch.zeros_like(hidden_states)

        use_deferred_mhc = (
            self.hc_enabled
            and (not attention_inputs.is_prefill or is_target_verify)
            and len(self.layers) > 0
        )
        deferred_post = deferred_comb = None
        moe_chunk_plan = (
            glm_prefill_moe_chunk_plan(
                self.layers,
                hidden_states.shape[0],
                hidden_states.device,
                token_sharded=attn_meta.token_shard is not None,
            )
            if self.synchronize_cuda_graph_choice
            or (attention_inputs.is_prefill and not is_target_verify)
            else nullcontext()
        )
        with moe_chunk_plan:
            for i, decoder_layer in enumerate(self.layers):
                is_linear = decoder_layer.layer_type == HybridAttentionType.LINEAR
                tag = "linear" if is_linear else "full"
                layer_inputs = linear_inputs if is_linear else attention_inputs
                layer_kv_cache = (
                    self.kv_cache.get_layer_cache(i, tag) if self.kv_cache else None
                )
                if use_deferred_mhc:
                    hidden_states, residual, deferred_post, deferred_comb = (
                        decoder_layer.forward_hc_deferred(
                            hidden_states,
                            fmha_impl,
                            layer_kv_cache,
                            layer_inputs,
                            attn_meta,
                            self.kv_cache,
                            residual,
                            deferred_post,
                            deferred_comb,
                        )
                    )
                else:
                    output = decoder_layer(
                        hidden_states,
                        residual,
                        fmha_impl,
                        kv_cache=layer_kv_cache,
                        attention_inputs=layer_inputs,
                        attn_meta=attn_meta,
                        global_kv_cache=self.kv_cache,
                    )
                    hidden_states = output.hidden_states
                    residual = output.residual
                if (
                    self.kv_cache is not None
                    and layer_inputs.is_prefill
                    and not is_target_verify
                ):
                    for layer_cache in self.kv_cache.get_layer_cache_groups(i):
                        if layer_cache.tag == "full":
                            continue  # Full MLA publishes its own cache after the write.
                        group = select_attention_inputs_for_tag(
                            group_inputs, layer_cache.tag
                        )
                        if group.cache_store_writer and group.cache_store_inputs:
                            group.cache_store_writer.write(
                                group.cache_store_inputs, layer_cache
                            )

        if use_deferred_mhc:
            assert deferred_post is not None and deferred_comb is not None
            hidden_states = self.layers[-1].ffn_hc.post(
                hidden_states, residual, deferred_post, deferred_comb
            )
            residual = hidden_states

        if self.hc_enabled:
            hidden_states = hidden_states.mean(dim=-2)
            hidden_states = self.norm(hidden_states, None)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if attn_meta.token_shard is not None:
            hidden_states = all_gather_trim(
                hidden_states, attn_meta.token_shard.logical_tokens, Group.TP
            )
        return PyModelOutputs(hidden_states)
