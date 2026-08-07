from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


class MlaAttention(nn.Module):
    """MLA attention. Supports both dense and sparse (indexer/top-k) modes.
    Whether to use Indexer is determined by attn_config.is_sparse.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        global_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.attn_config = attn_config
        self.parallelism_config = parallelism_config
        self.num_heads = (
            attn_config.head_num // self.parallelism_config.get_attn_tp_size()
        )
        self.qk_nope_head_dim = attn_config.nope_head_dim
        self.qk_rope_head_dim = attn_config.rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = attn_config.kv_lora_rank
        self.v_head_dim = attn_config.v_head_dim
        self.q_lora_rank = attn_config.q_lora_rank
        self.softmax_scale = self.q_head_dim ** (-0.5)
        self.layer_idx = layer_idx
        self.token_per_block = attn_config.kernel_tokens_per_block

        if attn_config.is_sparse:
            self.indexer = Indexer(
                attn_config,
                weights,
                global_weights,
                layer_idx,
                layernorm_eps,
                quant_config,
                hw_kernel_config,
                parallelism_config,
            )
        else:
            self.indexer = None

        if self.q_lora_rank > 0:
            self.fused_qkv_a_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_w,
                W.mla_fusedqkrope_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            self.q_a_layernorm = RMSNorm(
                weights.get(W.mla_q_a_ln_gamma, None), eps=layernorm_eps
            )
            self.q_b_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_q_b_w,
                W.mla_q_b_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.fused_qkv_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_no_lora_w,
                W.mla_fusedqkrope_no_lora_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        self.kv_a_layernorm = RMSNorm(
            weights.get(W.mla_kv_a_ln_gamma, None), eps=layernorm_eps
        )

        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def _run_sparse_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: Optional[torch.Tensor],
        q_view: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_impl: MlaImplBase,
    ) -> Optional[torch.Tensor]:
        if self.indexer is None:
            return None
        q_for_indexer = q_c if self.q_lora_rank > 0 else q_view
        return self.indexer(
            hidden_states,
            q_for_indexer,
            kv_cache,
            fmha_impl.fmha_params,
            fmha_impl.attn_inputs,
            use_fast_path=not fmha_impl.is_sparse(),
            cp_params=fmha_impl.cp_params,
        )

    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project the shared MLA input and optionally return an output gate."""

        return self.fused_qkv_a_proj(hidden_states), None

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        output_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Hook applied to the attention context after reshape and before o_proj.

        Identity by default (standard MLA has no output gate). Subclasses that
        need a per-element output gate (e.g. Kimi-K3's sigmoid gate) override
        this. The projected gate must follow the local attention-head layout.
        """
        return attn_output

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Apply the row-parallel output projection and TP reduction.

        This hook preserves the existing AllReduce behavior by default. Hybrid
        attention subclasses can override only this boundary when their
        layer-to-layer activation layout uses Sequence Parallel.
        """

        attn_output = self.o_proj(attn_output)
        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output

    def _profile_stage(self, stage: str, tensor: torch.Tensor):
        prefix = getattr(self, "_perf_profile_prefix", None)
        if prefix is None:
            return nullcontext()
        shape = "x".join(str(dim) for dim in tensor.shape)
        return torch.autograd.profiler.record_function(
            f"{prefix}.{stage}[shape={shape},dtype={tensor.dtype}]"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        output_gate = None
        q_c = None
        if self.q_lora_rank > 0:
            with self._profile_stage("q_kv_down_projection", hidden_states):
                fused_qkv, output_gate = self._project_qkv_a_input(hidden_states)
            kv_offset = self.q_lora_rank
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            with self._profile_stage("q_latent_rmsnorm", q):
                q_c = self.q_a_layernorm(
                    q
                    if getattr(self, "_perf_accepts_strided_latent", False)
                    else q.contiguous()
                )
            with self._profile_stage("q_up_projection_local_heads", q_c):
                q = self.q_b_proj(q_c)
        else:
            with self._profile_stage("q_kv_projection", hidden_states):
                fused_qkv = self.fused_qkv_proj(hidden_states)
            kv_offset = self.num_heads * self.attn_config.size_per_head
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
        input_shape = q.shape[:-1]
        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        with self._profile_stage("kv_latent_rmsnorm", compressed_kv):
            compressed_kv = self.kv_a_layernorm(
                compressed_kv
                if getattr(self, "_perf_accepts_strided_latent", False)
                else compressed_kv.contiguous()
            )

        with self._profile_stage("sparse_indexer_or_dense_noop", q_view):
            topk_indices = self._run_sparse_indexer(
                hidden_states, q_c, q_view, kv_cache, fmha_impl
            )
        with self._profile_stage("native_mla_and_cache_pipeline", q_view):
            attn_output = fmha_impl.forward(
                q_view, compressed_kv, k_pe, kv_cache, self.layer_idx, topk_indices
            )

        if attn_output is not None:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        with self._profile_stage("sigmoid_output_gate", attn_output):
            attn_output = self._apply_output_gate(attn_output, output_gate)
        with self._profile_stage("o_projection_then_token_reduce_scatter", attn_output):
            return self._project_output(attn_output)
