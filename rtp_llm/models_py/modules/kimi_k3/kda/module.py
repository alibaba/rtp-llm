"""Public Kimi K3 delta-attention module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Literal, Optional

import torch
from torch import nn

from rtp_llm.model_loader.linear_attn_weight import split_kda_qkvg_fa_beta_sections
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    get_process_group,
)
from rtp_llm.models_py.distributed.sequence_parallel import SequenceParallelLayout
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.kimi_k3.all_gather_gemm import all_gather_gemm
from rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter import gemm_reduce_scatter
from rtp_llm.models_py.modules.kimi_k3.projection_ktp import (
    project_kda_inputs_ktp,
    resolve_projection_local_heads,
)
from rtp_llm.models_py.modules.kimi_k3.kda.cache import KimiK3KDACache
from rtp_llm.models_py.modules.kimi_k3.kda.decode import KimiK3KDADecode
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
    KimiK3KDAPrefill,
    KimiKDACurrentStateRegistry,
    KimiKDAPrefillMetadata,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_quant import (
    quantize_forget_latent_fp8,
)
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
        self.ktp_size = int(getattr(parallelism_config, "ktp_size", 1))
        self.ktp_rank = int(getattr(parallelism_config, "ktp_rank", 0))
        self.total_heads = int(config.linear_attention_config.linear_num_key_heads)
        if self.total_heads % self.attn_tp_size:
            raise ValueError(
                f"KDA heads {self.total_heads} must be divisible by "
                f"attention TP {self.attn_tp_size}"
            )
        self.local_heads = self.total_heads // self.attn_tp_size
        self.projection_size = self.local_heads * self.head_dim
        if self.ktp_size > 1:
            if parallelism_config.role_type != RoleType.DECODE:
                raise RuntimeError(
                    "Projection KTP is Decode-only; Prefill and PDFUSION must use ktp_size=1"
                )
            if self.attn_tp_size != 1:
                raise RuntimeError(
                    f"Projection KTP requires attention TP=1, got {self.attn_tp_size}"
                )
            if self.ktp_size not in (8, 16):
                raise RuntimeError(
                    f"Projection KTP supports only sizes 8 and 16, got {self.ktp_size}"
                )
            if self.total_heads % self.ktp_size:
                raise ValueError(
                    f"KDA heads {self.total_heads} must be divisible by KTP {self.ktp_size}"
                )
        self.projection_local_heads = resolve_projection_local_heads(
            total_heads=self.total_heads,
            attention_tp_size=self.attn_tp_size,
            ktp_size=self.ktp_size,
        )
        self.projection_local_size = self.projection_local_heads * self.head_dim
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

        quant_config = getattr(config, "k3_attention_quant_config", None)
        self._fp8_enabled = quant_config is not None
        self._fp8_projections = {}
        if self._fp8_enabled:
            for name, scale_name in (
                (W.linear_attn_qkvg_fa_beta_w, W.linear_attn_qkvg_fa_beta_s),
                (W.linear_attn_f_b_w, W.linear_attn_f_b_s),
                (W.linear_attn_out_w, W.linear_attn_out_s),
            ):
                self._fp8_projections[name] = LinearFactory.create_linear_from_weights(
                    weights,
                    name,
                    scale_name,
                    None,
                    quant_config=quant_config,
                )
                self.add_module(
                    "fp8_" + name.replace(".", "_"), self._fp8_projections[name]
                )
        from rtp_llm.models_py.modules.kimi_k3.fp8_producers import (
            Fp8KdaOutputNorm,
            KdaOutputNorm,
        )

        output_norm_impl = Fp8KdaOutputNorm if self._fp8_enabled else KdaOutputNorm
        self.output_norm = output_norm_impl(weights[W.linear_attn_norm_w], self.eps)
        fused_projection = weights[W.linear_attn_qkvg_fa_beta_w]
        self.forget_latent_size = (
            self._fp8_projections[W.linear_attn_f_b_w].K
            if self._fp8_enabled
            else int(weights[W.linear_attn_f_b_w].shape[0])
        )
        self._fp8_strided_forget = (
            self._fp8_enabled
            and self.forget_latent_size == 128
            and getattr(
                self._fp8_projections[W.linear_attn_f_b_w], "scale_ue8m0", False
            )
        )
        if self._fp8_strided_forget:
            logging.info(
                "K3_FP8_EXECUTION layer=%d projection=f_b "
                "activation_quant=strided_group128_ue8m0 compute=fp8 output=bf16",
                layer_idx,
            )
        expected_fused_width = (
            4 * self.projection_local_size
            + self.forget_latent_size
            + self.total_heads
        )
        actual_fused_width = (
            self._fp8_projections[W.linear_attn_qkvg_fa_beta_w].N
            if self._fp8_enabled
            else fused_projection.shape[1]
        )
        if actual_fused_width != expected_fused_width:
            raise ValueError(
                "fused KDA QKVG/F_A/beta width "
                f"{actual_fused_width} != {expected_fused_width}"
            )
        self.kda_fused_w = self._fp8_projections.get(
            W.linear_attn_qkvg_fa_beta_w, fused_projection
        )

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
        sp_layout: Optional[SequenceParallelLayout],
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

        if self.ktp_size > 1:
            result = project_kda_inputs_ktp(
                hidden_states,
                self.kda_fused_w,
                self._fp8_projections.get(
                    W.linear_attn_f_b_w,
                    self.weights[W.linear_attn_f_b_w],
                ),
                total_heads=self.total_heads,
                head_dim=self.head_dim,
                forget_latent_size=self.forget_latent_size,
                ktp_size=self.ktp_size,
                ktp_rank=self.ktp_rank,
            )
            mixed_qkv_projected = torch.cat(
                (result.q, result.k, result.v), dim=-1
            )
            return (
                mixed_qkv_projected,
                result.q,
                result.k,
                result.v,
                result.raw_gate,
                result.raw_beta,
                result.output_gate,
            )

        if self.attn_tp_size > 1 and sp_layout is not None:
            projected_fused = all_gather_gemm(
                hidden_states,
                [self.kda_fused_w],
                logical_m=sp_layout.tokens.physical_tokens,
            )[0]
        else:
            projected_fused = (
                self.kda_fused_w(hidden_states)
                if self._fp8_enabled
                else torch.matmul(hidden_states, self.kda_fused_w)
            )
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
        if self._fp8_enabled:
            forget_projection = self._fp8_projections[W.linear_attn_f_b_w]
            if self._fp8_strided_forget:
                # F_A is a strided slice of the fused output. Read it directly
                # into FP8 instead of launching a BF16 staging copy first.
                raw_gate = forget_projection.forward_quantized(
                    *quantize_forget_latent_fp8(forget_latent)
                )
            else:
                raw_gate = forget_projection(forget_latent.contiguous())
        else:
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
        sequence_parallel: bool,
        mode: KDAExecutionMode,
    ) -> torch.Tensor:
        token_count = output_gate.shape[1]
        # Prefill, Decode, and target verify share one normalization and
        # projection path. Physical rows are padded once before modeling, so
        # the fused ReduceScatter never allocates an intermediate pad buffer.
        output = self.output_norm(output, output_gate, mode)

        projection_input = output.reshape(token_count, self.projection_size)
        output_weight = self._fp8_projections.get(
            W.linear_attn_out_w, self.weights[W.linear_attn_out_w]
        )
        if sequence_parallel and self.attn_tp_size > 1:
            return gemm_reduce_scatter(
                projection_input,
                output_weight,
                get_process_group(Group.TP),
                pad_rows=False,
            )
        return (
            output_weight(projection_input)
            if self._fp8_enabled
            else torch.matmul(projection_input, output_weight)
        )

    def _validate_request(
        self,
        hidden_states: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        sequence_parallel: bool,
        sp_layout: SequenceParallelLayout,
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
        if sequence_parallel and (
            self.attn_tp_size <= 1
            or not hidden_states.is_cuda
            or int(hidden_states.shape[0]) != sp_layout.tokens.local_tokens
        ):
            raise ValueError(
                "K3 Sequence Parallel requires a CUDA physical token shard "
                "whose layout matches attention TP"
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
        sp_layout: Optional[SequenceParallelLayout] = None,
        prefill_metadata: Optional[KimiKDAPrefillMetadata] = None,
        current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
    ) -> torch.Tensor:
        if sp_layout is None:
            raise ValueError("K3 attention requires a physical token layout")
        is_target_verify = self._validate_request(
            hidden_states,
            mode=mode,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            sequence_parallel=sequence_parallel,
            sp_layout=sp_layout,
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
            sp_layout=sp_layout,
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
            sequence_parallel=sequence_parallel,
            mode=mode,
        )
        return output


__all__ = ["KDAExecutionMode", "KimiK3KDA"]
