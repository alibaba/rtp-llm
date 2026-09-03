"""Public Kimi K3 delta-attention module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional

import torch
from torch import nn

from rtp_llm.model_loader.linear_attn_weight import split_kda_qkvg_fa_beta_sections
from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
from rtp_llm.models_py.modules.kimi_k3.all_gather_gemm import all_gather_gemm
from rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter import gemm_reduce_scatter
from rtp_llm.models_py.modules.kimi_k3.kda.cache import KimiK3KDACache
from rtp_llm.models_py.modules.kimi_k3.kda.decode import KimiK3KDADecode
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
    KimiK3KDAPrefill,
    KimiKDACurrentStateRegistry,
    KimiKDAPrefillMetadata,
)
from rtp_llm.models_py.triton_kernels.kimi_kda import kimi_kda_rms_norm_sigmoid_gate
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig


KDAExecutionMode = Literal["prefill", "decode"]


class KimiK3KDA(nn.Module):
    """Project KDA inputs, delegate role-specific execution, and project output."""

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.weights = weights
        runtime = config.k3_runtime_config
        self.head_dim = int(config.linear_attention_config.linear_key_head_dim)
        self.attn_tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.total_heads = int(config.linear_attention_config.linear_num_key_heads)

        self.local_heads = self.total_heads // self.attn_tp_size
        self.projection_size = self.local_heads * self.head_dim
        self.history_size = (
            int(config.linear_attention_config.linear_conv_kernel_dim) - 1
        )
        self.eps = float(config.layernorm_eps)
        self.gate_lower_bound = runtime.kda_gate_lower_bound

        if parallelism_config.role_type not in (
            RoleType.PREFILL,
            RoleType.DECODE,
            RoleType.PDFUSION,
        ):
            raise RuntimeError(
                "Kimi K3 supports only PREFILL, DECODE, or PDFUSION roles, got "
                f"{parallelism_config.role_type}"
            )
        self._role_type = parallelism_config.role_type

        converter = LinearCacheConverter(
            local_num_v_heads=self.local_heads,
            head_v_dim=self.head_dim,
            head_k_dim=self.head_dim,
            ssm_state_dtype=to_torch_dtype(
                config.linear_attention_config.ssm_state_dtype
            ),
            linear_conv_kernel_dim=int(
                config.linear_attention_config.linear_conv_kernel_dim
            ),
            qkv_size=3 * self.projection_size,
            conv_state_dtype=to_torch_dtype(
                config.linear_attention_config.conv_state_dtype
            ),
        )
        self.cache = KimiK3KDACache(
            converter,
            local_heads=self.local_heads,
            head_dim=self.head_dim,
            projection_size=self.projection_size,
            history_size=self.history_size,
        )
        self.cache_store_segment_sizes = self.cache.store_segment_sizes

        fused_projection = weights[W.linear_attn_qkvg_fa_beta_w]
        self.forget_latent_size = int(weights[W.linear_attn_f_b_w].shape[0])

        self.kda_fused_w = fused_projection

        fused_conv = weights[W.linear_attn_conv1d_w].squeeze(1)     # 这里为什么要squeeze(1)？？

        self.prefill_executor: Optional[KimiK3KDAPrefill]
        self.decode_executor: Optional[KimiK3KDADecode]
        self.prefill_executor = KimiK3KDAPrefill(
            weights=weights,
            cache=self.cache,
            local_heads=self.local_heads,
            head_dim=self.head_dim,
            projection_size=self.projection_size,
            gate_lower_bound=self.gate_lower_bound,
            fused_conv=fused_conv,
        )

        self.decode_executor = KimiK3KDADecode(
            weights=weights,
            cache=self.cache,
            local_heads=self.local_heads,
            head_dim=self.head_dim,
            projection_size=self.projection_size,
            history_size=self.history_size,
            gate_lower_bound=self.gate_lower_bound,
            fused_conv=fused_conv,
        )

    def _project_fused_kda_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run and unpack the loader-provided Q/K/V/G/F_A/beta projection."""

        projected_fused = all_gather_gemm(
            hidden_states,
            [self.kda_fused_w],
        )[0]
        (
            q_projected,
            k_projected,
            v_projected,
            output_gate,
            forget_latent,
            full_raw_beta,
        ) = split_kda_qkvg_fa_beta_sections(
            projected_fused,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.forget_latent_size,
            self.total_heads,
            dim=1,
        )
        raw_gate = torch.matmul(forget_latent, self.weights[W.linear_attn_f_b_w])
        beta_begin = self.attn_tp_rank * self.local_heads
        raw_beta = full_raw_beta.narrow(1, beta_begin, self.local_heads)
        mixed_qkv_projected = projected_fused.narrow(1, 0, 3 * self.projection_size)
        return (
            mixed_qkv_projected,
            q_projected,
            k_projected,
            v_projected,
            raw_gate,
            raw_beta,
            output_gate,
        )

    def _paged_decode_core(
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
        """Compatibility shim for the paged-cache ABI unit test."""

        return KimiK3KDADecode._recurrent(
            self,
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            cu_seqlens,
            ssm_cache,
            block_map,
            sequence_lengths_plus_one,
            page_size,
        )

    def _project_output(
        self,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        token_count = output_gate.shape[1]
        # Prefill, Decode and target verify share one normalization/gate path.
        output = kimi_kda_rms_norm_sigmoid_gate(
            output,
            output_gate,
            self.weights[W.linear_attn_norm_w],
            self.eps,
        )

        projection_input = output.reshape(token_count, self.projection_size)
        return gemm_reduce_scatter(
            projection_input,
            self.weights[W.linear_attn_out_w],
            get_process_group(Group.TP),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        prefill_metadata: Optional[KimiKDAPrefillMetadata] = None,
        current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
    ) -> torch.Tensor:
        is_target_verify = bool(
            attention_inputs is not None
            and getattr(attention_inputs, "is_target_verify", False)
        )
        (
            mixed_qkv_projected,
            q_projected,
            k_projected,
            v_projected,
            raw_gate,
            raw_beta,
            output_gate_projected,
        ) = self._project_fused_kda_inputs(
            hidden_states,
        )
        token_count = q_projected.shape[0]
        output_gate = output_gate_projected.reshape(
            1, token_count, self.local_heads, self.head_dim
        )

        if mode == "prefill":
            assert self.prefill_executor is not None
            output = self.prefill_executor(
                mixed_qkv_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                metadata=prefill_metadata,
                current_state_registry=current_state_registry,
                layer_idx=self.layer_idx,
            )
        else:
            assert kv_cache is not None and attention_inputs is not None
            assert self.decode_executor is not None
            output = self.decode_executor(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                is_target_verify=is_target_verify,
            )
        output = self._project_output(output, output_gate)
        return output


__all__ = ["KDAExecutionMode", "KimiK3KDA"]
