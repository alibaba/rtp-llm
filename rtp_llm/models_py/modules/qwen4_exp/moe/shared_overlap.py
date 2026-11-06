"""Overlap the local shared branch with routing and experts on SM120.

The two-event scheduling follows vLLM's SharedExperts runner (Apache-2.0).
The routed TP reduction stays on the original stream. The unsharded shared
branch retains its BF16 output and original fused sigmoid-scale-add boundary.
"""

import os
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
    _fuse_ep_shared_gate = False
    _ep_shared_stream = None
    _ep_two_batch = None

    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        moe_config,
        max_generate_batch_size=0,
        enable_cuda_graph=False,
        hw_kernel_config=None,
        linear_factory=None,
    ):
        super().__init__(
            config, parallelism_config, weights, moe_config,
            max_generate_batch_size, enable_cuda_graph, hw_kernel_config,
            linear_factory,
        )
        self._sm90_mega_decode_limit = max_generate_batch_size
        self._sm90_mega_moe = None

    def prepare_shared_expert_overlap(self) -> None:
        """Allocate streams only for the explicitly qualified Qwen4 layout."""
        if os.environ.get("QWEN_MOE_SM90_MEGA_MOE", "0") == "1":
            from .sm90_mega_moe import QwenSm90Fp8MegaMoe

            self._sm90_mega_moe = QwenSm90Fp8MegaMoe(
                self, self._sm90_mega_decode_limit
            )
        self._fuse_ep_shared_gate = (
            os.environ.get("QWEN_MOE_SHARED_GATE_FUSION", "0") == "1"
        )
        if os.environ.get("QWEN_MOE_EP_SHARED_OVERLAP", "0") == "1":
            self._prepare_ep_shared_overlap()
        if not getattr(self.config, "_qwen4_full_fp8_nvfp4", False):
            return
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

    def _forward_routed(self, hidden_states):
        mega = getattr(self, "_sm90_mega_moe", None)
        if mega is None or not mega.supports(hidden_states):
            return super()._forward_routed(hidden_states)
        tokens = hidden_states.shape[0]
        logits = self.gate(hidden_states)
        weights = torch.empty(
            (tokens, self.top_k), device=hidden_states.device, dtype=torch.float32
        )
        ids = torch.empty(
            weights.shape, device=hidden_states.device,
            dtype=self.fused_moe.topk_ids_dtype,
        )
        self.select_topk(logits, ids, weights)
        return mega.forward(hidden_states, ids, weights)

    def _prepare_ep_shared_overlap(self) -> None:
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        pc = self.parallelism_config
        shared = self.shared_expert
        if not (
            (
                (
                    pc.tp_size == 2 and pc.ep_size == 2
                    and tuple(shared.up_proj.weight.shape) == (640, 2560)
                    and tuple(shared.down_proj.weight.shape) == (2560, 320)
                )
                or (
                    os.environ.get("QWEN_MOE_EP8_OVERLAP_EXPERIMENT", "0") == "1"
                    and pc.tp_size == 8 and pc.ep_size == 8
                    and tuple(shared.up_proj.weight.shape) == (160, 2560)
                    and tuple(shared.down_proj.weight.shape) == (2560, 80)
                )
            )
            and pc.dp_size == 1
            and self.use_ep_shared_allreduce
            and not self.use_unified_tp_allreduce
            and not self.config.eplb_config.enable_eplb()
            and type(shared) is DenseMLP
            and type(self.fused_moe) is FusedMoe
            and type(self.fused_moe.router) is DeepEpLowLatencyRouter
            and type(self.fused_moe.fused_experts) is DeepGemmMaskedExecutor
            and type(shared.up_proj) is Fp8Linear
            and type(shared.down_proj) is Fp8Linear
            and type(self.shared_expert_gate) is Fp8Linear
            and tuple(self.shared_expert_gate.weight.shape) == (1, 2560)
            and shared.up_proj.weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(shared.up_proj.weight.device) == (9, 0)
        ):
            return
        self._ep_shared_stream = _shared_stream_for_device(
            shared.up_proj.weight.device.index
        )
        self._ep_shared_input_ready = torch.cuda.Event()
        self._ep_shared_output_ready = torch.cuda.Event()
        if (
            os.environ.get("QWEN_MOE_EP_TBO", "0") == "1"
            and self.correction_bias is None
            and self.fake_balance_expert is None
        ):
            from .ep_tbo import QwenEpTwoBatch

            self._ep_two_batch = QwenEpTwoBatch(self)

    def _forward_ep_shared_overlap(self, hidden_states):
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        main = torch.cuda.current_stream(hidden_states.device)
        capturing = torch.cuda.is_current_stream_capturing()
        self._ep_shared_input_ready.record(main)
        # Keep TP collectives on the original stream, preserving rank order.
        # Only independent local shared-expert arithmetic uses the auxiliary.
        try:
            with torch.cuda.stream(self._ep_shared_stream):
                self._ep_shared_input_ready.wait(self._ep_shared_stream)
                try:
                    shared = self.shared_expert(hidden_states, skip_allreduce=True)
                    shared = self._gate_shared_expert_output(hidden_states, shared)
                finally:
                    self._ep_shared_output_ready.record(self._ep_shared_stream)
            if self._ep_two_batch is not None and self._ep_two_batch.supports(
                hidden_states
            ):
                experts = self._ep_two_batch.forward(self, hidden_states)
            else:
                experts = self._forward_routed(hidden_states)
        finally:
            # Allocator reuse on the original stream is ordered after the join,
            # including when either branch raises during eager execution.
            self._ep_shared_output_ready.wait(main)
        if not capturing:
            shared.record_stream(main)
        shared = getattr(self, "tp_all_reduce", all_reduce)(shared, group=Group.TP)
        return experts + shared

    def _gate_shared_expert_output(self, hidden_states, shared_expert_output):
        if not self._fuse_ep_shared_gate or self.shared_expert_gate is None:
            return super()._gate_shared_expert_output(
                hidden_states, shared_expert_output
            )
        from .shared_gate_triton import try_shared_sigmoid_mul

        gate = self.shared_expert_gate(hidden_states)
        fused = try_shared_sigmoid_mul(gate, shared_expert_output)
        if fused is not None:
            return fused
        return torch.sigmoid(gate) * shared_expert_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self._ep_shared_stream is not None
            and hidden_states.is_cuda
            and hidden_states.device == self._ep_shared_stream.device
            and hidden_states.ndim == 2
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and 1 <= hidden_states.shape[0] <= 32
            and hidden_states.shape[1] == 2560
        ):
            return self._forward_ep_shared_overlap(hidden_states)
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
