"""Overlap the local shared branch with routing and experts on SM120.

The two-event scheduling follows vLLM's SharedExperts runner (Apache-2.0).
The routed TP reduction stays on the original stream. The unsharded shared
branch retains its BF16 output and original fused sigmoid-scale-add boundary.
"""

from functools import lru_cache

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer


@lru_cache(maxsize=None)
def _shared_stream_for_device(device_index: int):
    """Reuse an auxiliary stream across layers, as vLLM's aux_stream does."""
    return torch.cuda.Stream(device=torch.device("cuda", device_index))


class Qwen4MoeLayer(GenericMoeLayer):
    _shared_stream = None
    _reused_shared_stream = None

    def prepare_shared_expert_overlap(self) -> None:
        """Allocate streams only for the explicitly qualified Qwen4 layout."""
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
            CutlassNvfp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNvfp4Cutlass,
        )
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        pc = self.parallelism_config
        role = str(getattr(pc, "role_type", "PDFUSION")).rsplit(".", 1)[-1]
        shared = self.shared_expert
        if not (
            role == "DECODE"
            and getattr(self.config, "_qwen4_full_fp8_nvfp4", False)
            and pc.tp_size == 2
            and pc.ep_size == 2
            and pc.dp_size == 1
            and pc.ffn_sp_size == 2
            and self.ffn_tp_size == 1
            and not self.use_ep_shared_allreduce
            and not self.use_unified_tp_allreduce
            and not self.config.eplb_config.enable_eplb()
            and type(shared) is DenseMLP
            and type(self.fused_moe) is FusedMoe
            and type(self.fused_moe.router) is PureTpRouterNvfp4Cutlass
            and type(self.fused_moe.fused_experts) is CutlassNvfp4Executor
            and type(shared.up_proj) is Fp8Linear
            and type(shared.down_proj) is Fp8Linear
            and type(self.shared_expert_gate) is Fp8Linear
            and tuple(shared.up_proj.weight.shape) == (1280, 2560)
            and tuple(shared.down_proj.weight.shape) == (2560, 640)
            and tuple(self.shared_expert_gate.weight.shape) == (1, 2560)
            and shared.up_proj.weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(shared.up_proj.weight.device)
            == (12, 0)
        ):
            return
        self._shared_stream = torch.cuda.Stream(device=shared.up_proj.weight.device)
        self._reused_shared_stream = _shared_stream_for_device(
            shared.up_proj.weight.device.index
        )
        self._shared_input_ready = torch.cuda.Event()
        self._shared_output_ready = torch.cuda.Event()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
            experts = self._forward_routed(hidden_states)
        finally:
            # Also join on an exception: callers may release/reuse the input.
            self._shared_output_ready.wait(main)
        if not capturing:
            shared.record_stream(main)
            gate.record_stream(main)
        self.sigmoid_gate_scale_add(gate, shared, experts)
        return experts
