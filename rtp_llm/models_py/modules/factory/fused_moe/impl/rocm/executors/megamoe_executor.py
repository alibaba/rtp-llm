import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.distributed.megamoe_wrapper import MegaMoeWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class MegaMoeFusedExecutor(FusedMoeExpertExecutor):
    """Monolithic executor that runs the full FlyDSL 2-stage fused MegaMoE.

    Unlike the standard executors which only run the expert GEMMs, this
    executor drives the entire FlyDSL FusedMoEZeroCopyFp8 pipeline
    (dispatch + GEMM1 fused, then GEMM2 + combine fused). It therefore relies
    on the passthrough router which forwards raw tokens + global routing.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.MEGAMOE_FUSED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(bool(getattr(config, "use_megamoe", False)))
        checker.check(MegaMoeWrapper.supported())

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.ep_rank = config.ep_rank
        self.ep_size = config.ep_size

        self.w1 = weights[W.moe_w1]
        self.w1_scale = weights.get(W.moe_s1)
        self.w2 = weights.get(W.moe_w2)
        self.w2_scale = weights.get(W.moe_s2)

        self._wrapper = MegaMoeWrapper.get_instance()
        if self._wrapper is None:
            raise RuntimeError(
                "MegaMoeWrapper is not initialized. "
                "Ensure init_megamoe_wrapper() ran during backend startup."
            )
        # FlyDSL stage1 needs FP8 preshuffled w1 + a per-row scale. For a bf16
        # model these are absent, so quantize/shuffle w1 up-front into the
        # layout FlyDSL expects ([epr*2*inter, model_dim] fp8, flat scale).
        self._w1_flat, self._scale_w1 = self._prepare_w1()
        self._w2_injected = False
        self._w2_flat = None
        self._scale_w2 = None
        self._maybe_inject_w2()

    def _quant_helpers(self):
        try:
            from flydsl.tests.utils import pertoken_quant, shuffle_weight
        except ImportError:
            from tests.utils import pertoken_quant, shuffle_weight
        return pertoken_quant, shuffle_weight

    def _prepare_w1(self):
        """Return (w1_flat_fp8, scale_w1_1d) in FlyDSL's expected layout.

        w1 from the model is [epr, 2*inter, model_dim]. FlyDSL wants a flat
        [epr*2*inter, model_dim] fp8 tensor (preshuffled) plus a 1-D per-row
        scale. If the model already provides an fp8 scale we reuse it, else we
        pertoken-quantize the bf16 weights here.
        """
        op = self._wrapper.op
        fp8_dtype = getattr(op, "_a2_fp8", torch.empty(0)).dtype
        w1 = self.w1
        if self.w1_scale is not None and w1.dtype == fp8_dtype:
            _, shuffle_weight = self._quant_helpers()
            w1_shuffled = shuffle_weight(w1)
            return (
                w1_shuffled.reshape(-1, w1.shape[-1]).contiguous(),
                self.w1_scale.reshape(-1).contiguous(),
            )

        pertoken_quant, shuffle_weight = self._quant_helpers()
        w1_q, scale_w1 = pertoken_quant(w1.to(torch.float32), quant_dtype=fp8_dtype)
        w1_shuffled = shuffle_weight(w1_q)
        w1_flat = w1_shuffled.reshape(-1, w1.shape[-1]).contiguous()
        scale_w1_1d = scale_w1.reshape(-1).contiguous()
        logger.info(
            "MegaMoE: quantized bf16 w1 -> fp8 flat=%s scale=%s",
            tuple(w1_flat.shape),
            tuple(scale_w1_1d.shape),
        )
        return w1_flat, scale_w1_1d

    def _maybe_inject_w2(self) -> None:
        """Inject the real GEMM2 weights into the FlyDSL op.

        FlyDSL's FusedMoEZeroCopyFp8 allocates a random placeholder W2 for
        benchmarking. For correct output we overwrite its ``_w2_storage`` /
        ``_w2_scale`` buffers with the model's real weights, following the same
        recipe FlyDSL uses for w1: per-token(row) quantize the fp32 weights to
        fp8, derive a per-output-row scale, then apply FlyDSL ``shuffle_weight``
        on the fp8 data (the scale is NOT shuffled). w2 is [epr, model_dim,
        inter_dim]; its scale flattens to [epr*model_dim].
        """
        if self.w2 is None:
            logger.warning("MegaMoE: no moe_w2 weight found; using placeholder W2")
            return
        op = self._wrapper.op
        w2_storage = getattr(op, "_w2_storage", None)
        w2_scale = getattr(op, "_w2_scale", None)
        if w2_storage is None or w2_scale is None:
            logger.warning(
                "MegaMoE: op has no _w2_storage/_w2_scale; skip W2 injection"
            )
            return

        pertoken_quant, shuffle_weight = self._quant_helpers()
        fp8_dtype = w2_storage.dtype
        try:
            if self.w2_scale is not None and self.w2.dtype == fp8_dtype:
                # Model already ships prequantized fp8 w2 + per-row scale.
                w2_q = self.w2
                scale_w2 = self.w2_scale
            else:
                w2_q, scale_w2 = pertoken_quant(
                    self.w2.to(torch.float32), quant_dtype=fp8_dtype
                )
            w2_shuffled = shuffle_weight(w2_q).contiguous().view(-1)
            if w2_shuffled.numel() != w2_storage.numel():
                logger.warning(
                    "MegaMoE: W2 numel mismatch (real=%d op=%d); skip injection",
                    w2_shuffled.numel(),
                    w2_storage.numel(),
                )
                return

            flat_scale = scale_w2.to(w2_scale.dtype).contiguous().view(-1)
            if flat_scale.numel() != w2_scale.numel():
                logger.warning(
                    "MegaMoE: W2 scale numel mismatch (real=%d op=%d); "
                    "output magnitude will be wrong",
                    flat_scale.numel(),
                    w2_scale.numel(),
                )
                return

            # The FlyDSL op is a SINGLETON shared by every MoE layer, but its
            # _w2_storage/_w2_scale hold only ONE layer's W2. If we merely copied
            # here, the LAST-constructed layer would win and every other layer
            # would run GEMM2 with the wrong W2. So keep this layer's prepared
            # W2 on the instance and re-activate it before each forward().
            self._w2_flat = w2_shuffled
            self._scale_w2 = flat_scale
            w2_storage.copy_(w2_shuffled)
            w2_scale.copy_(flat_scale)

            self._w2_injected = True
            logger.info(
                "MegaMoE: injected real W2 (pertoken-quant fp8) scale=%s",
                tuple(flat_scale.shape),
            )
        except Exception as e:  # noqa: BLE001 - bring-up robustness
            logger.warning("MegaMoE: W2 injection failed (%s); using placeholder", e)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        x_bf16 = payload.expert_x
        topk_weights = payload.expert_topk_weights
        topk_ids = payload.expert_topk_ids
        assert x_bf16 is not None, "expert_x is None"
        assert topk_weights is not None and topk_ids is not None

        x_bf16 = x_bf16.contiguous()
        wts = topk_weights.to(torch.float32).contiguous()
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        topk_ids = topk_ids.contiguous()

        w1 = self._w1_flat
        scale_w1 = self._scale_w1

        # The op is a singleton shared by every MoE layer; make sure GEMM2 runs
        # with THIS layer's W2 (see _maybe_inject_w2) and not a sibling layer's.
        self._activate_w2()

        out = self._wrapper.forward(x_bf16, wts, topk_ids, w1, scale_w1)
        # FlyDSL forward returns (out_tok, out_wts); we only need combined tokens.
        out_tok = out[0] if isinstance(out, (tuple, list)) else out
        out_tok = out_tok.to(x_bf16.dtype)

        return CombineForwardPayload(fused_expert_output=out_tok)

    def _activate_w2(self) -> None:
        """Copy this layer's prepared W2 into the shared singleton op buffers.

        FusedMoEZeroCopyFp8 keeps a single _w2_storage/_w2_scale, but each MoE
        layer has its own weights. Re-activating per forward keeps every layer
        correct without allocating a separate op (which would re-JIT-compile).
        """
        if self._w2_flat is None or self._scale_w2 is None:
            return
        op = self._wrapper.op
        w2_storage = getattr(op, "_w2_storage", None)
        w2_scale = getattr(op, "_w2_scale", None)
        if w2_storage is None or w2_scale is None:
            return
        w2_storage.copy_(self._w2_flat)
        w2_scale.copy_(self._scale_w2)
