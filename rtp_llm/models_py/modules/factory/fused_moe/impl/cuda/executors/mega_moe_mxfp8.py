"""MegaMoE executors backed by the GLM5 mega-kernel wrappers.

These mirror the FP8/FP4 mega executors: a strategy in ``strategy/mxfp8.py``
selects one, the router hands over materialized top-k tensors, and the kernel
fuses EP dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine over a
symmetric-memory buffer. EP > 1 only, SM100+.

``mega_moe_fp8``/``mega_moe_fp8_se`` run DeepGEMM's FP8xFP8 entry points, which
have their own buffer type and weight transform rather than sitting behind the
FP4 buffer's ``mma_type`` switch -- so they cannot reuse ``MegaMoeExecutor``.
``mega_moe_fused`` keeps FP4 routed weights but folds the FP4 shared expert into
``fp8_fp4_mega_moe_fused``.

The kernel plumbing -- buffer cache, input packing, JIT warmup and capacity
chunking -- lives in ``modules.glm5_mega_moe``; the classes here only adapt it
to the fused-MoE executor contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

from .fp8_fp4_base import Fp8Fp4ExecutorBase


class MegaMoeWrapperExecutorBase(Fp8Fp4ExecutorBase):
    """Shared wiring for executors delegating to a GLM5 mega-kernel wrapper.

    Subclasses declare the checkpoint quantization they accept; this base only
    asserts the parallel topology the kernel requires. The FP8/FP4 ``execute``
    contract already matches -- the executor owns activation quantization and
    applies router weights during its own combine.
    """

    @classmethod
    def _impl_class(cls) -> Type[torch.nn.Module]:
        raise NotImplementedError

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.FP8_FP8

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        # The kernel is a peer-symmetric NVLink collective: every rank must
        # enter it together, so the EP group has to be the whole world.
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(not config.has_redundant_experts)

    def setup_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        cfg = self.cfg
        self._impl = self._impl_class()(
            cfg.model_config,
            cfg.parallelism_config,
            weights,
            cfg.moe_config,
            layer_idx=cfg.layer_id,
            max_generate_batch_size=cfg.max_generate_batch_size,
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self._impl(x, topk_weights, topk_ids)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """Forward ``activation`` and ``extra_expert_args`` into the kernel.

        The FP8/FP4 base drops both. The mega kernels need them: an
        ``swiglu_oai`` layer reads its alpha and clamp out of
        ``extra_expert_args``, and pinning the activation here would silently
        downgrade such a layer to plain SwiGLU.
        """
        if expert_map is not None:
            raise ValueError("MegaMoE does not support expert_map")
        if a2_scale is not None:
            raise ValueError("MegaMoE does not accept an external a2_scale")
        if apply_router_weight_on_input:
            raise ValueError("MegaMoE applies router weights during output combine")
        if payload.gate_payload is not None:
            raise ValueError(
                f"{type(self).__name__} has no fused gate-packing entry point"
            )
        topk_weights = payload.expert_topk_weights
        topk_ids = payload.expert_topk_ids
        if topk_weights is None or topk_ids is None:
            raise ValueError("MegaMoE requires routed top-k weights and ids")
        output = self._impl(
            payload.expert_x,
            topk_weights,
            topk_ids,
            activation=activation,
            extra_expert_args=extra_expert_args,
        )
        return CombineForwardPayload(
            fused_expert_output=self._restore_output_dtype(payload, output)
        )

    def clone_for_cuda_graph(self) -> "MegaMoeWrapperExecutorBase":
        """Clone with a graph-private output buffer.

        ``torch.nn.Module.__init__`` gives the clone fresh module dicts, so
        assigning ``_impl`` registers the cloned wrapper on the clone rather
        than mutating the original's submodules.
        """
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.config = self.config
        clone.quant_config = self.quant_config
        clone.weights = self.weights
        clone.cfg = self.cfg
        clone._impl = self._impl.clone_for_cuda_graph()
        return clone


class MegaMoeFp8Executor(MegaMoeWrapperExecutorBase):
    """Routed-only MXFP8 experts through ``fp8_fp8_mega_moe``."""

    includes_shared_expert = False

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.moe_quant_method == "MXFP8")

    @classmethod
    def _impl_class(cls) -> Type[torch.nn.Module]:
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_wrapper import (
            MegaMoeFp8Wrapper,
        )

        return MegaMoeFp8Wrapper


class MegaMoeFp8SEExecutor(MegaMoeFp8Executor):
    """MXFP8 routed experts plus the FP8 shared expert, fused in-kernel."""

    includes_shared_expert = True

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)

    @classmethod
    def _impl_class(cls) -> Type[torch.nn.Module]:
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se_wrapper import (
            MegaMoeFp8SEWrapper,
        )

        return MegaMoeFp8SEWrapper


class MegaMoeFusedExecutor(MegaMoeWrapperExecutorBase):
    """FP4 routed experts plus the FP4 shared expert, fused in-kernel.

    No quantization gate: the wrapper validates the routed and shared weight
    dtypes itself and reports which tensor was wrong, which is more useful than
    a strategy-selection miss. This matches how the strategy behaved before the
    factory owned its selection.
    """

    includes_shared_expert = True

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)

    @classmethod
    def _impl_class(cls) -> Type[torch.nn.Module]:
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fused_wrapper import (
            MegaMoeFusedWrapper,
        )

        return MegaMoeFusedWrapper
