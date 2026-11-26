"""Overlap shared experts with routed experts on SM120.

The BF16 TP-sharded path follows SGLang's Qwen2MoE dual-stream schedule and
combines the local shared and routed contributions before their TP reduction.
The FP8 path retains its local shared-expert schedule.
"""

from copy import deepcopy
from functools import lru_cache

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.qwen4_exp.dense_mlp import Qwen4DenseMLP


@lru_cache(maxsize=None)
def _shared_stream_for_device(device_index: int):
    """Reuse an auxiliary stream across layers, as vLLM's aux_stream does."""
    return torch.cuda.Stream(device=torch.device("cuda", device_index))


class Qwen4MoeLayer(GenericMoeLayer):
    _shared_mlp_cls = Qwen4DenseMLP

    _shared_stream = None
    _reused_shared_stream = None
    _bf16_shared_stream = None
    _bf16_shared_tp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            getattr(self.config, "_qwen4_dense_storage", None) != "source_bf16"
            or self.shared_expert is None
        ):
            return
        pc = deepcopy(self.parallelism_config)
        pc.ffn_tp_size = pc.get_attn_tp_size()
        pc.ffn_tp_rank = pc.tp_rank
        self.shared_expert.parallelism_config = pc
        router = self.fused_moe.router
        self._bf16_shared_tp = (
            router.supports_skip_tp_allreduce
            and router.tp_collective_size == pc.ffn_tp_size
        )
        self.use_unified_tp_allreduce = self._bf16_shared_tp
        self.use_ep_shared_allreduce = not self._bf16_shared_tp and pc.ffn_tp_size > 1

    def prepare_shared_expert_overlap(self) -> None:
        """Allocate streams only for the explicitly qualified Qwen4 layout."""
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
            CutlassNvfp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor import (
            Nvfp4HybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNvfp4Cutlass,
        )
        from rtp_llm.models_py.modules.qwen4_exp.dense_mlp import (
            Qwen4DenseMLP as DenseMLP,
        )

        pc = self.parallelism_config
        role = str(getattr(pc, "role_type", "PDFUSION")).rsplit(".", 1)[-1]
        fp8_decode = role == "DECODE" and getattr(
            self.config, "_qwen4_full_fp8_nvfp4", False
        )
        bf16_target = (
            getattr(self.config, "_qwen4_dense_storage", None) == "source_bf16"
        )
        if not (fp8_decode or bf16_target):
            return
        shared = self.shared_expert
        if bf16_target:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
                CudaF16Linear,
            )

            if (
                self._bf16_shared_tp
                and torch.version.hip is None
                and shared.up_proj.weight.is_cuda
                and all(
                    isinstance(projection, CudaF16Linear)
                    and projection.weight.dtype == torch.bfloat16
                    and projection.weight.device == shared.up_proj.weight.device
                    for projection in (
                        shared.up_proj,
                        shared.down_proj,
                        self.shared_expert_gate,
                    )
                )
            ):
                self._bf16_shared_stream = _shared_stream_for_device(
                    shared.up_proj.weight.device.index
                )
                self._shared_input_ready = torch.cuda.Event()
                self._shared_output_ready = torch.cuda.Event()
            return
        if not (
            pc.tp_size == 2
            and pc.ep_size == 2
            and pc.dp_size == 1
            and self.ffn_tp_size == 1
            and not self.use_ep_shared_allreduce
            and (not self.use_unified_tp_allreduce or self._bf16_shared_tp)
            and not self.config.eplb_config.enable_eplb()
            and type(shared) is DenseMLP
            and type(self.fused_moe) is FusedMoe
            and type(self.fused_moe.router) is PureTpRouterNvfp4Cutlass
            and type(self.fused_moe.fused_experts)
            in (CutlassNvfp4Executor, Nvfp4HybridExecutor)
            and shared.up_proj.weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(shared.up_proj.weight.device)
            == (12, 0)
        ):
            return

        device = shared.up_proj.weight.device
        if (
            fp8_decode
            and type(shared.up_proj) is Fp8Linear
            and type(shared.down_proj) is Fp8Linear
            and type(self.shared_expert_gate) is Fp8Linear
            and tuple(shared.up_proj.weight.shape) == (1280, 2560)
            and tuple(shared.down_proj.weight.shape) == (2560, 640)
            and tuple(self.shared_expert_gate.weight.shape) == (1, 2560)
        ):
            self._shared_stream = torch.cuda.Stream(device=device)
            self._reused_shared_stream = _shared_stream_for_device(device.index)
        else:
            return
        self._shared_input_ready = torch.cuda.Event()
        self._shared_output_ready = torch.cuda.Event()

    def forward(
        self, hidden_states: torch.Tensor, *, attention_inputs=None, is_cuda_graph=False
    ) -> torch.Tensor:
        if self._bf16_shared_tp:
            return self._forward_bf16_shared_tp(
                hidden_states, is_cuda_graph=is_cuda_graph
            )
        reduce_shared = getattr(self, "_reduce_shared_gate", None)
        if reduce_shared is not None and reduce_shared.supports(hidden_states):
            output = reduce_shared.prepare_output(hidden_states)
            experts = self._forward_routed(hidden_states, skip_tp_allreduce=True)
            shared = self.shared_expert(hidden_states, skip_allreduce=True)
            gate = self.shared_expert_gate(hidden_states)
            return reduce_shared(experts, shared, gate, output)

        stream = self._shared_stream
        # Larger shapes did not improve in the complete TP2 consumer. Keep
        # their established schedule and each layer's distinct event pair.
        if (
            self._reused_shared_stream is not None
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in (1, 4)
        ):
            stream = self._reused_shared_stream
        if not (
            stream is not None
            and hidden_states.is_cuda
            and hidden_states.device == stream.device
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and 1 <= hidden_states.shape[0] <= 32
            and hidden_states.shape[1] == 2560
            and hidden_states.is_contiguous()
        ):
            return super().forward(hidden_states)

        experts, shared, gate = self._overlap_shared_branch(hidden_states, stream)
        self.sigmoid_gate_scale_add(gate, shared, experts)
        return experts

    def _forward_bf16_shared_tp(self, hidden_states, *, is_cuda_graph):
        from rtp_llm.models_py.modules.qwen4_exp.moe.shared_gate import (
            fused_gate_sigmoid_mul_add,
        )

        main = torch.cuda.current_stream(hidden_states.device)
        stream = self._bf16_shared_stream
        capturing = torch.cuda.is_current_stream_capturing()
        overlap = (
            hidden_states.shape[0] > 0
            and stream is not None
            and is_cuda_graph
            and not torch.compiler.is_compiling()
        )
        shared = None
        if hidden_states.shape[0] > 0:
            if overlap:
                self._shared_input_ready.record(main)
            shared = self.shared_expert(
                hidden_states.clone() if overlap else hidden_states, skip_allreduce=True
            )
        if overlap:
            try:
                with torch.cuda.stream(stream):
                    self._shared_input_ready.wait(stream)
                    try:
                        experts = self._forward_routed(
                            hidden_states, skip_tp_allreduce=True
                        )
                    finally:
                        self._shared_output_ready.record(stream)
            finally:
                self._shared_output_ready.wait(main)
            if not capturing:
                experts.record_stream(main)
                hidden_states.record_stream(stream)
        else:
            experts = self._forward_routed(hidden_states, skip_tp_allreduce=True)
        if shared is not None:
            fused_gate_sigmoid_mul_add(
                hidden_states, self.shared_expert_gate.weight, shared, experts
            )
        if self.fused_moe.router.tp_collective_size > 1:
            return getattr(self, "tp_all_reduce", all_reduce)(experts, group=Group.TP)
        return experts

    def _overlap_shared_branch(
        self,
        hidden_states: torch.Tensor,
        stream,
        *,
        skip_tp_allreduce: bool = False,
    ):
        main = torch.cuda.current_stream(hidden_states.device)
        capturing = torch.cuda.is_current_stream_capturing()
        self._shared_input_ready.record(main)
        try:
            with torch.cuda.stream(stream):
                self._shared_input_ready.wait(stream)
                try:
                    # No collective is allowed in this branch. DenseMLP's
                    # only collective is suppressed by this existing argument.
                    shared = self.shared_expert(hidden_states, skip_allreduce=True)
                    gate = self.shared_expert_gate(hidden_states)
                finally:
                    self._shared_output_ready.record(stream)
            if skip_tp_allreduce:
                experts = self._forward_routed(hidden_states, skip_tp_allreduce=True)
            else:
                experts = self._forward_routed(hidden_states)
        finally:
            # Also join on an exception: callers may release/reuse the input.
            self._shared_output_ready.wait(main)
        if not capturing:
            shared.record_stream(main)
            gate.record_stream(main)
        return experts, shared, gate
