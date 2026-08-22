"""Public Kimi K3 delta-attention module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional

import torch
from torch import nn

from rtp_llm.model_loader.linear_attn_weight import split_kda_qkvg_fa_beta_sections
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_reduce,
    get_process_group,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    TokenShardLayout,
    shard_tokens_with_padding,
)
from rtp_llm.models_py.modules.factory.linear.parallel import row_parallel_linear
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
        collective_group: Group = Group.TP,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.collective_group = collective_group
        self.weights = weights
        runtime = config.k3_runtime_config
        self.head_dim = int(config.linear_attention_config.linear_key_head_dim)
        self.attn_tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.total_heads = int(config.linear_attention_config.linear_num_key_heads)
        if self.total_heads % self.attn_tp_size:
            raise ValueError(
                f"KDA heads {self.total_heads} must be divisible by "
                f"attention TP {self.attn_tp_size}"
            )
        self.local_heads = self.total_heads // self.attn_tp_size
        self.projection_size = self.local_heads * self.head_dim
        self.history_size = (
            int(config.linear_attention_config.linear_conv_kernel_dim) - 1
        )
        self.eps = float(config.layernorm_eps)
        self.gate_lower_bound = runtime.kda_gate_lower_bound
        if not runtime.kda_use_full_rank_gate:
            raise NotImplementedError(
                "K3 checkpoint manifest currently requires full-rank KDA output gate"
            )
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
        expected_fused_width = (
            4 * self.projection_size + self.forget_latent_size + self.total_heads
        )
        if fused_projection.shape[1] != expected_fused_width:
            raise ValueError(
                "fused KDA QKVG/F_A/beta width "
                f"{fused_projection.shape[1]} != {expected_fused_width}"
            )
        self.kda_fused_w = fused_projection

        fused_conv = weights[W.linear_attn_conv1d_w].squeeze(1)
        if fused_conv.shape[0] != 3 * self.projection_size:
            raise ValueError(
                "fused KDA conv channels "
                f"{fused_conv.shape[0]} != 3*{self.projection_size}"
            )

        self.prefill_executor: Optional[KimiK3KDAPrefill]
        self.decode_executor: Optional[KimiK3KDADecode]
        if self._role_type in (RoleType.PREFILL, RoleType.PDFUSION):
            self.prefill_executor = KimiK3KDAPrefill(
                weights=weights,
                cache=self.cache,
                local_heads=self.local_heads,
                head_dim=self.head_dim,
                projection_size=self.projection_size,
                gate_lower_bound=self.gate_lower_bound,
                fused_conv=fused_conv,
            )
        else:
            self.prefill_executor = None

        if self._role_type in (RoleType.DECODE, RoleType.PDFUSION):
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
        else:
            self.decode_executor = None

    def _project_fused_kda_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        prefill_sp_layout: Optional[TokenShardLayout],
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

        if prefill_sp_layout is not None:
            projected_fused = all_gather_gemm(
                hidden_states,
                [self.kda_fused_w],
                logical_m=prefill_sp_layout.logical_tokens,
            )[0]
        else:
            projected_fused = torch.matmul(hidden_states, self.kda_fused_w)
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
        *,
        is_target_verify: bool,
        sequence_parallel: bool,
        hidden_states: torch.Tensor,
        mode: KDAExecutionMode,
    ) -> torch.Tensor:
        token_count = output_gate.shape[1]
        # Decode and target-verify must use the same numerics. Mixing the fused
        # projection path with this explicit path can change near-tied logits.
        use_explicit_output = mode == "decode"
        if use_explicit_output:
            output_dtype = output.dtype
            norm_weight = self.weights[W.linear_attn_norm_w]
            output_float = output.float()
            rms = torch.rsqrt(
                output_float.square().mean(dim=-1, keepdim=True) + self.eps
            )
            output = output_float * rms
            output = output * norm_weight.float()
            output = output * torch.sigmoid(output_gate.float())
            output = output.to(dtype=output_dtype)
        else:
            output = kimi_kda_rms_norm_sigmoid_gate(
                output,
                output_gate,
                self.weights[W.linear_attn_norm_w],
                self.eps,
            )

        projection_input = output.reshape(token_count, self.projection_size)
        if use_explicit_output:
            output = torch.matmul(
                projection_input,
                self.weights[W.linear_attn_out_w],
            )
            if self.attn_tp_size > 1:
                output = all_reduce(output, group=self.collective_group)
                decode_sp = (
                    sequence_parallel and not is_target_verify and hidden_states.is_cuda
                )
                if decode_sp:
                    output, _ = shard_tokens_with_padding(
                        output,
                        token_count,
                        self.attn_tp_size,
                        self.attn_tp_rank,
                    )
            return output
        use_reduce_scatter = (
            sequence_parallel and self.attn_tp_size > 1 and hidden_states.is_cuda
        )
        pad_reduce_scatter = use_reduce_scatter and (
            mode == "decode" or token_count % self.attn_tp_size != 0
        )
        if mode == "prefill" and use_reduce_scatter:
            fused = gemm_reduce_scatter(
                projection_input,
                self.weights[W.linear_attn_out_w],
                get_process_group(self.collective_group),
                pad_rows=pad_reduce_scatter,
            )
            if fused is not None:
                return fused
        return row_parallel_linear(
            projection_input,
            self.weights[W.linear_attn_out_w],
            self.attn_tp_size,
            reduce_scatter_tokens=use_reduce_scatter,
            pad_reduce_scatter_tokens=pad_reduce_scatter,
            use_input_dtype_reduce_scatter=(mode == "prefill"),
            group=self.collective_group,
        )

    def _validate_request(
        self,
        hidden_states: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        sequence_parallel: bool,
        prefill_sp_layout: Optional[TokenShardLayout],
    ) -> bool:
        """Validate the role-specific contract and return target-verify mode."""

        is_target_verify = bool(
            attention_inputs is not None
            and getattr(attention_inputs, "is_target_verify", False)
        )
        if is_target_verify and self._role_type == RoleType.PREFILL:
            raise RuntimeError(
                "Kimi K3 target verify requires the direct paged Decode path"
            )
        if self._role_type == RoleType.PREFILL and mode != "prefill":
            raise RuntimeError("Kimi K3 Prefill role cannot execute Decode")
        if self._role_type == RoleType.DECODE and mode != "decode":
            raise RuntimeError("Kimi K3 Decode role cannot execute Prefill")
        if kv_cache is None or attention_inputs is None:
            raise RuntimeError(
                "Kimi K3 Prefill, Decode, and target verify require direct paged cache"
            )
        if prefill_sp_layout is not None and (
            mode != "prefill"
            or not sequence_parallel
            or self.attn_tp_size <= 1
            or not hidden_states.is_cuda
        ):
            raise ValueError(
                "prefill_sp_layout requires CUDA Prefill Sequence Parallel with TP>1"
            )
        return is_target_verify

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        sequence_parallel: bool = False,
        prefill_sp_layout: Optional[TokenShardLayout] = None,
        prefill_metadata: Optional[KimiKDAPrefillMetadata] = None,
        current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
    ) -> torch.Tensor:
        is_target_verify = self._validate_request(
            hidden_states,
            mode=mode,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            sequence_parallel=sequence_parallel,
            prefill_sp_layout=prefill_sp_layout,
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
            prefill_sp_layout=prefill_sp_layout,
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
        output = self._project_output(
            output,
            output_gate,
            is_target_verify=is_target_verify,
            sequence_parallel=sequence_parallel,
            hidden_states=hidden_states,
            mode=mode,
        )
        return output


__all__ = ["KDAExecutionMode", "KimiK3KDA"]
