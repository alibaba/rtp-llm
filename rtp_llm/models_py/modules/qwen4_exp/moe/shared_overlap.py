"""Overlap the local shared branch with routing and experts on SM120.

The two-event scheduling follows vLLM's SharedExperts runner (Apache-2.0).
The routed TP reduction stays on the original stream. The unsharded shared
branch retains its BF16 output and original fused sigmoid-scale-add boundary.
"""

import os
from functools import lru_cache

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.utils.model_weight import W


@lru_cache(maxsize=None)
def _shared_stream_for_device(device_index: int):
    """Reuse an auxiliary stream across layers, as vLLM's aux_stream does."""
    return torch.cuda.Stream(device=torch.device("cuda", device_index))


class Qwen4MoeLayer(GenericMoeLayer):
    _shared_stream = None
    _reused_shared_stream = None
    _fuse_ep_shared_gate = False
    _tp_shared_packed_comm = None
    _tp_shared_pack_reduce = False
    _ep_shared_max_tokens = 32
    _ep_shared_stream = None
    _ep_two_batch = None
    _sm90_mega_moe = None
    _mega_prefill_graph = None
    _pack_ep_shared_reduce = False

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
        pc = self.parallelism_config
        shared_weight = weights.get(W.ffn_w2)
        if (
            enable_cuda_graph
            and self.use_ep_shared_allreduce
            and self.ffn_tp_size == pc.tp_size == pc.ep_size == 2
            and pc.dp_size == 1
            and shared_weight is not None
            and shared_weight.dtype == torch.bfloat16
            and shared_weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(shared_weight.device) == (9, 0)
        ):
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
                PureTpRouterNoQuant,
            )

            router = self.fused_moe.router
            self._pack_ep_shared_reduce = (
                type(router) is PureTpRouterNoQuant and router.tp_collective_size == 2
            )

    def prepare_shared_expert_overlap(self, layer_idx=None) -> None:
        """Allocate streams only for the explicitly qualified Qwen4 layout."""
        if os.environ.get("QWEN_MOE_SM90_MEGA_MOE", "0") == "1":
            from .sm90_mega_moe import QwenSm90Fp8MegaMoe

            self._sm90_mega_moe = QwenSm90Fp8MegaMoe(
                self, self._sm90_mega_decode_limit
            )
            if os.environ.get("QWEN_MOE_MEGA_PREFILL_GRAPH", "0") == "1":
                from .prefill_graph import QwenMegaGraphAdmission, QwenMegaPrefillGraph

                if layer_idx == 0:
                    self.config._qwen_mega_graph_admission = QwenMegaGraphAdmission()
                admission = getattr(self.config, "_qwen_mega_graph_admission", None)
                if admission is None:
                    raise RuntimeError(
                        "MegaMoE prefill graphs require ordered layer initialization"
                    )
                self._mega_prefill_graph = QwenMegaPrefillGraph(layer_idx, admission)
        self._fuse_ep_shared_gate = (
            os.environ.get("QWEN_MOE_SHARED_GATE_FUSION", "1") == "1"
        )
        if os.environ.get("RTP_LLM_H20_COMPACT_FP8_MOE", "1") == "1":
            self._prepare_tp_fp8_shared_overlap()
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

    def _prepare_tp_fp8_shared_overlap(self) -> bool:
        """Overlap H20 FP8 shared expert with routed MoE and pack TP reduction."""
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
            DeepGemmMaskedExecutorV2,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        from .moe_tp_reduce import prepare_moe_communicator

        pc, shared = self.parallelism_config, self.shared_expert
        if not (
            shared is not None
            and self._sm90_mega_moe is None
            and pc.tp_size == pc.ep_size == 2
            and pc.dp_size == 1
            and pc.tp_size <= getattr(pc, "local_world_size", pc.tp_size)
            and self.use_ep_shared_allreduce
            and not self.use_unified_tp_allreduce
            and not self.config.eplb_config.enable_eplb()
            and self.fake_balance_expert is None
            and type(shared) is DenseMLP
            and type(self.fused_moe.router) is PureTpRouterFp8PerBlock
            and self.fused_moe.router.supports_skip_tp_allreduce
            and type(self.fused_moe.fused_experts) is DeepGemmMaskedExecutorV2
            and type(shared.up_proj) is Fp8Linear
            and type(shared.down_proj) is Fp8Linear
            and type(self.shared_expert_gate) is Fp8Linear
            and tuple(shared.up_proj.weight.shape) == (640, 2560)
            and tuple(shared.down_proj.weight.shape) == (2560, 320)
            and tuple(self.shared_expert_gate.weight.shape) == (1, 2560)
            and shared.up_proj.weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(shared.up_proj.weight.device) == (9, 0)
        ):
            return False
        device = shared.up_proj.weight.device
        self._ep_shared_stream = _shared_stream_for_device(device.index)
        self._ep_shared_input_ready = torch.cuda.Event()
        self._ep_shared_output_ready = torch.cuda.Event()
        self._ep_shared_max_tokens = 128
        self._tp_shared_pack_reduce = True
        self._tp_shared_packed_comm = prepare_moe_communicator(
            device, max_decode_rows=self._ep_shared_max_tokens
        )
        return True

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
        from rtp_llm.models_py.modules.base.common.fp8_linear import (
            Fp8Linear,
            _prequantized_fp8_input,
            prequantize_fp8_linear_input,
        )
        from rtp_llm.models_py.modules.qwen4_exp._profiler import record_function_range

        rows = hidden_states.shape[0]
        use_pack = (
            self._tp_shared_pack_reduce
            and self._sm90_mega_moe is None
            and (self._ep_two_batch is None or not self._ep_two_batch.supports(hidden_states))
        )
        packed = (
            hidden_states.new_empty((2 * rows, hidden_states.shape[1]))
            if use_pack
            else None
        )
        main = torch.cuda.current_stream(hidden_states.device)
        capturing = torch.cuda.is_current_stream_capturing()
        use_aux_stream = (
            capturing or rows <= self._ep_shared_max_tokens or rows >= 1536
        )
        prequant_entry = None
        prequant_scope = None
        gate_proj = getattr(self, "gate", None)
        up_proj = getattr(self.shared_expert, "up_proj", None)
        if (
            type(gate_proj) is Fp8Linear
            and type(up_proj) is Fp8Linear
            and gate_proj.use_scaled_mm
            and up_proj.use_scaled_mm
            and gate_proj.compute_weight is None
            and up_proj.compute_weight is None
            and gate_proj.weight.shape[1]
            == up_proj.weight.shape[1]
            == hidden_states.shape[1]
            and gate_proj.weight.dtype == up_proj.weight.dtype
            and gate_proj.role == up_proj.role
            and not any(
                "forward" in proj.__dict__
                or proj._forward_pre_hooks
                or proj._forward_hooks
                for proj in (gate_proj, up_proj)
            )
            and not torch.nn.modules.module._global_forward_pre_hooks
            and not torch.nn.modules.module._global_forward_hooks
        ):
            prequant_entry = prequantize_fp8_linear_input(gate_proj, hidden_states)
            if prequant_entry is not None:
                prequant_scope = _prequantized_fp8_input.set(prequant_entry)
                if not capturing and use_aux_stream:
                    prequant_entry[3].record_stream(self._ep_shared_stream)
                    prequant_entry[4].record_stream(self._ep_shared_stream)
        if use_aux_stream:
            self._ep_shared_input_ready.record(main)
        # Keep TP collectives on the original stream, preserving rank order.
        # Only independent local shared-expert arithmetic uses the auxiliary.
        try:
            if use_aux_stream:
                with torch.cuda.stream(self._ep_shared_stream):
                    self._ep_shared_input_ready.wait(self._ep_shared_stream)
                    try:
                        shared = self.shared_expert(hidden_states, skip_allreduce=True)
                        shared = self._gate_shared_expert_output(
                            hidden_states,
                            shared,
                            out=packed[rows:] if packed is not None else None,
                        )
                    finally:
                        self._ep_shared_output_ready.record(self._ep_shared_stream)
            else:
                shared = self.shared_expert(hidden_states, skip_allreduce=True)
                shared = self._gate_shared_expert_output(
                    hidden_states,
                    shared,
                    out=packed[rows:] if packed is not None else None,
                )
            if self._ep_two_batch is not None and self._ep_two_batch.supports(
                hidden_states
            ):
                experts = self._ep_two_batch.forward(self, hidden_states)
            elif use_pack:
                from .compact_fp8_decode import compact_moe_output_buffer

                out_scope = compact_moe_output_buffer.set(packed[:rows])
                try:
                    experts = super()._forward_routed(
                        hidden_states, skip_tp_allreduce=True
                    )
                finally:
                    compact_moe_output_buffer.reset(out_scope)
            else:
                experts = self._forward_routed(hidden_states)
        finally:
            if prequant_scope is not None:
                _prequantized_fp8_input.reset(prequant_scope)
            # Allocator reuse on the original stream is ordered after the join,
            # including when either branch raises during eager execution.
            if use_aux_stream:
                self._ep_shared_output_ready.wait(main)
        if not capturing and use_aux_stream:
            shared.record_stream(main)
        reduce = getattr(self, "tp_all_reduce", all_reduce)
        if use_pack:
            # Retain separate BF16 rounding per branch before the final sum.
            if experts.data_ptr() != packed[:rows].data_ptr():
                packed[:rows].copy_(experts)
            if shared.data_ptr() != packed[rows:].data_ptr():
                packed[rows:].copy_(shared)
            comm = self._tp_shared_packed_comm
            if (
                comm is not None
                and packed.shape[0] <= 2 * self._ep_shared_max_tokens
                and comm.should_use(packed)
            ):
                with record_function_range("qwen4.tp_all_reduce"):
                    packed = comm.all_reduce(packed)
            else:
                packed = reduce(packed, group=Group.TP)
            return packed[:rows] + packed[rows:]
        shared = reduce(shared, group=Group.TP)
        return experts + shared

    def _gate_shared_expert_output(
        self, hidden_states, shared_expert_output, *, out=None
    ):
        if not self._fuse_ep_shared_gate or self.shared_expert_gate is None:
            gated = super()._gate_shared_expert_output(
                hidden_states, shared_expert_output
            )
            if out is not None:
                out.copy_(gated)
                return out
            return gated
        from .shared_gate_triton import try_shared_sigmoid_mul

        gate = self.shared_expert_gate(hidden_states)
        fused = try_shared_sigmoid_mul(gate, shared_expert_output, out=out)
        if fused is not None:
            return fused
        gated = torch.sigmoid(gate) * shared_expert_output
        if out is not None:
            out.copy_(gated)
            return out
        return gated

    def _forward_ep_with_packed_reduce(self, hidden_states):
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        routed = super()._forward_routed(hidden_states, skip_tp_allreduce=True)
        shared = self.shared_expert(hidden_states, skip_allreduce=True)
        shared = self._gate_shared_expert_output(hidden_states, shared)
        # Reduce both partial tensors in one collective, retaining each BF16
        # reduction's rounding before the local addition.
        packed = torch.stack((routed, shared))
        reduced = getattr(self, "tp_all_reduce", all_reduce)(packed, group=Group.TP)
        return reduced[0] + reduced[1]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        graph = self._mega_prefill_graph
        if (
            graph is not None
            and self._sm90_mega_moe is not None
            and self._sm90_mega_moe.supports(hidden_states)
            and graph.supports(hidden_states)
        ):
            return graph.forward(self, hidden_states, self._forward_impl)
        return self._forward_impl(hidden_states)

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self._ep_shared_stream is not None
            and hidden_states.is_cuda
            and hidden_states.device == self._ep_shared_stream.device
            and hidden_states.ndim == 2
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and hidden_states.shape[1] == 2560
        ):
            rows = hidden_states.shape[0]
            use_ep_overlap = 1 <= rows <= self._ep_shared_max_tokens
            if not use_ep_overlap and 1 <= rows <= 5120:
                from .compact_fp8_decode import (
                    _H20_COMPACT_EAGER_MAX_ROWS,
                    h20_compact_moe_phase,
                )

                use_ep_overlap = (
                    rows <= _H20_COMPACT_EAGER_MAX_ROWS
                    and h20_compact_moe_phase.get() == "eager"
                    and not torch.cuda.is_current_stream_capturing()
                    and os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") != "1"
                )
            if use_ep_overlap:
                return self._forward_ep_shared_overlap(hidden_states)
        if (
            self._pack_ep_shared_reduce
            and self._sm90_mega_moe is None
            and hidden_states.ndim == 2
            and 16 <= hidden_states.shape[0] <= 64
            and hidden_states.dtype == torch.bfloat16
        ):
            return self._forward_ep_with_packed_reduce(hidden_states)
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
