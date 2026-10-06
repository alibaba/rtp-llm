"""MiniMax-M3.1 runtime model specializations."""

from typing import Any

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3Model,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import RoleType
from rtp_llm.ops.compute_ops import PyModelInputs


class MiniMaxM31MoeLayer(GenericMoeLayer):
    def prepare_prefill_router(self):
        """Expand FP32 gates; BF16 gates retain their ordinary linear path."""
        if not isinstance(self.gate, CudaF16Linear) or self.gate.bias is not None:
            raise ValueError("M3.1 Prefill router requires a bias-free linear gate")
        part_names = ("_prefill_gate_high", "_prefill_gate_middle", "_prefill_gate_low")
        if self.gate.weight.dtype == torch.bfloat16:
            # Do not retain FP32 expansion buffers across a gate reload.
            for name in part_names:
                self._buffers.pop(name, None)
            return
        from rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router import (
            expand_fp32_router_weight,
        )

        parts = expand_fp32_router_weight(self.gate.weight)
        for name, tensor in zip(part_names, parts):
            self.register_buffer(name, tensor, persistent=False)

    def clone_for_cuda_graph(self):
        clone = super().clone_for_cuda_graph()
        # Decode/verify clones share the raw FP32 gate, not Prefill-only parts.
        clone._prefill_router_active = False
        return clone

    def _compute_router_logits(self, hidden_states):
        if (
            getattr(self, "_prefill_router_active", False)
            and hasattr(self, "_prefill_gate_high")
            and hidden_states.dtype == torch.bfloat16
        ):
            from rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router import (
                minimax_m31_prefill_router_logits,
            )

            return minimax_m31_prefill_router_logits(
                hidden_states.contiguous(),
                (
                    self._prefill_gate_high,
                    self._prefill_gate_middle,
                    self._prefill_gate_low,
                ),
            )
        if (
            getattr(self, "_batch_invariant_router", False)
            and isinstance(self.gate, CudaF16Linear)
            and self.gate.weight.dtype == torch.float32
            and self.gate.bias is None
        ):
            from rtp_llm.models_py.triton_kernels.minimax_m31_router import (
                minimax_m31_router_logits,
            )

            return minimax_m31_router_logits(
                hidden_states.float().contiguous(), self.gate.weight
            )
        return super()._compute_router_logits(hidden_states)


class _MiniMaxM31MSAQueryContext:
    """No-op FMHA context for the all-sparse MSA model.

    MiniMax-M3.1 has an MSA attention layer in every transformer block.  The
    regular ``GenericMoeModel`` still asks the model for an FMHA context, but
    MSA performs its own paged-cache/indexer path and must not instantiate a
    CP FlashInfer/MLA implementation.  Returning this context keeps the
    generic model output contract (``fmha_params is None``) without entering
    ``fill_mla_params``.
    """

    fmha_params = None

    def prepare_cuda_graph(self, _attn_inputs) -> None:
        return None

    def support_cuda_graph(self) -> bool:
        return True


class MiniMaxM31DecoderLayer(MiniMaxM3DecoderLayer):
    """M3.1 retains M3 experts, with fixed row routing for decode/verify."""

    def _create_mlp(
        self,
        config,
        parallelism_config,
        weights,
        moe_config,
        max_generate_batch_size,
        enable_cuda_graph,
        hw_kernel_config,
        layer_idx,
    ):
        if layer_idx not in config.moe_layer_index:
            return super()._create_mlp(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
                hw_kernel_config,
                layer_idx,
            )
        return MiniMaxM31MoeLayer(
            config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph=enable_cuda_graph,
            hw_kernel_config=hw_kernel_config,
            layer_idx=layer_idx,
        )

    def _forward_attention(
        self,
        hidden_states,
        fmha_impl,
        kv_cache,
        prev_topk_indices,
        force_reuse_topk_indices,
        attn_inputs,
        x_fp8=None,
        x_scale=None,
    ):
        # Set on this layer's actual MLP (including Graph clones), never on the
        # shared gate module. Capture records fixed kernels; replay needs no
        # Python flag mutation. Only an explicitly prepared Prefill model uses
        # its exact FP32-weight expansion; Decode/PDFUSION allocate no parts.
        if isinstance(self.mlp, MiniMaxM31MoeLayer):
            self.mlp._prefill_router_active = (
                attn_inputs is not None
                and attn_inputs.is_prefill
                and not getattr(attn_inputs, "is_target_verify", False)
            )
            self.mlp._batch_invariant_router = attn_inputs is not None and (
                not attn_inputs.is_prefill
                or getattr(attn_inputs, "is_target_verify", False)
            )
        return super()._forward_attention(
            hidden_states,
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            force_reuse_topk_indices,
            attn_inputs,
            x_fp8,
            x_scale,
        )


class MiniMaxM31Model(MiniMaxM3Model):
    decoder_layer_cls = MiniMaxM31DecoderLayer

    def initialize(self, init_resource):
        result = super().initialize(init_resource)
        if self.parallelism_config.role_type == RoleType.PREFILL:
            for layer in self.layers:
                if isinstance(layer.mlp, MiniMaxM31MoeLayer) and getattr(
                    layer.self_attn, "nvfp4_kv_cache", False
                ):
                    layer.mlp.prepare_prefill_router()
        return result

    def _prepare_prefill_moe_chunk_plan(self, inputs, hidden_states, layers):
        from rtp_llm.models_py.model_desc.generic_moe import (
            cuda_graph_capture_forward_enabled,
            cuda_graph_warmup_forward_enabled,
        )

        attn = inputs.attention_inputs
        if (
            not attn.is_prefill
            or getattr(attn, "is_target_verify", False)
            or cuda_graph_capture_forward_enabled()
            or cuda_graph_warmup_forward_enabled()
            or (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            return None

        import torch.distributed as dist

        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_nvfp4_wrapper import (
            MegaMoeNvfp4Wrapper,
        )
        from rtp_llm.models_py.modules.glm5_mega_moe.prefill_chunk_plan import (
            PrefillChunkPlan,
            local_chunk_count,
        )

        routed = [
            getattr(layer.mlp, "fused_moe", None) for layer in layers[: self.layer_num]
        ]
        nvfp4 = [m.mega_moe for m in routed if isinstance(m, MegaMoeNvfp4Wrapper)]
        if not nvfp4:
            return None
        first = nvfp4[0]
        group = first._mega_group
        if dist.get_world_size(group) == 1:
            return None
        capacity = int(first._mega_buf.num_max_tokens_per_rank)
        if any(
            m._mega_group is not group
            or int(m._mega_buf.num_max_tokens_per_rank) != capacity
            for m in nvfp4
        ):
            raise ValueError(
                "NVFP4 prefill layers must share EP group and chunk capacity"
            )
        if any(
            m is not None and not isinstance(m, MegaMoeNvfp4Wrapper) for m in routed
        ):
            raise ValueError(
                "mixed routed MoE strategies cannot use NVFP4 prefill plan"
            )
        # One forward-local collective, including fake/short ranks. Use actual
        # local embedding rows (already CP-local), never divide by CP again.
        # +/- capacity also detects inconsistent buffer limits across ranks.
        agreed = torch.tensor(
            [local_chunk_count(hidden_states.shape[0], capacity), capacity, -capacity],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group)
        chunks, max_capacity, neg_min_capacity = agreed.tolist()
        if max_capacity != -neg_min_capacity:
            raise ValueError("NVFP4 prefill chunk capacity differs across EP ranks")
        return PrefillChunkPlan(capacity, chunks)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        # Every M3.1 layer is MSAAttention, including target verification.  MSA
        # owns its paged cache, indexer and speculative metadata and does not
        # consume GenericMoeModel's FMHA implementation.  Constructing the
        # inherited M3 dense/FlashInfer target-verify context is both redundant
        # and incorrect for the all-sparse checkpoint (it can enter
        # fill_mla_params despite there being no full-attention layer).
        del inputs, is_cuda_graph
        return _MiniMaxM31MSAQueryContext()


__all__ = ["MiniMaxM31DecoderLayer", "MiniMaxM31Model"]
