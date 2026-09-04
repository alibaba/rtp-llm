import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


def _is_elastic_enabled() -> bool:
    return int(os.environ.get("USE_DEEPEP_ELASTIC", "0")) == 1


class DeepEpElasticRouter(FusedMoeDataRouter):
    """DeepEPv2 ElasticBuffer dispatch/combine router.

    Two layouts:
      (do_expand=False, do_cpu_sync=True)  → prefill (compact, no expand)
      (do_expand=False, do_cpu_sync=False) → decode cudagraph
    """

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.DEEPEP_ELASTIC

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(_is_elastic_enabled())
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(DeepEPWrapper.supported())
        try:
            from deep_ep import ElasticBuffer  # noqa: F401
        except ImportError:
            checker.check(False)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)

        self._num_experts: int = config.expert_num
        self._num_topk: int = config.moe_k
        self._ep_size: int = config.ep_size
        self._ep_rank: int = config.ep_rank
        self._tp_size: int = config.tp_size
        self._tp_rank: int = config.tp_rank
        assert (
            self._num_experts % self._ep_size == 0
        ), f"expert_num={self._num_experts} not divisible by ep_size={self._ep_size}"
        self._expert_per_rank: int = self._num_experts // self._ep_size
        self._rank_expert_offset: int = self._ep_rank * self._expert_per_rank

        self._use_fp8_dispatch: bool = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )
        self._use_fp4: bool = (
            MoeConfigResolver().get_quant_method(config) == "modelopt_fp4"
        )
        self._use_local_expert_ids: bool = self._use_fp8_dispatch or self._use_fp4
        self._expert_alignment: int = 128 if quant_config.is_block_quantized else 1

        deepep_config = DeepepWrapperConfig.from_config_adapter(self.config)

        self._do_expand: bool = deepep_config.elastic_do_expand
        self._do_cpu_sync: bool = deepep_config.elastic_do_cpu_sync
        # Allow (do_expand=False, do_cpu_sync=True) for prefill without GPU-CPU sync
        self._use_decode_cudagraph: bool = (not self._do_cpu_sync) and (
            not self._do_expand
        )

        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert wrapper.mode == DeepEPMode.ELASTIC, (
            f"DeepEpElasticRouter expects DeepEPMode.ELASTIC, got {wrapper.mode}. "
            "Make sure USE_DEEPEP_ELASTIC=1 is exported before initialising the "
            "wrapper singleton."
        )
        self._buffer = wrapper.elastic_buffer
        self._handle: Optional[Any] = None
        self._nan_guard_active: bool = bool(
            int(os.environ.get("DEEPEP_ELASTIC_NAN_GUARD", "1"))
        )
        self._elastic_num_sms: int = int(os.environ.get("DEEPEP_ELASTIC_NUM_SMS", "0"))

    @property
    def handle(self) -> Optional[Any]:
        return self._handle

    def _tp_slice(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_ids = topk_ids.to(torch.int64)
        tp_size = self._tp_size
        tp_rank = self._tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        return (
            torch.narrow(a1, 0, slice_begin, slice_size),
            torch.narrow(topk_ids, 0, slice_begin, slice_size),
            torch.narrow(topk_weights, 0, slice_begin, slice_size),
        )

    def _do_quant_fp8(self, a1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a1.size(0) == 0:
            hidden = a1.size(1)
            a1_q = torch.empty((0, hidden), dtype=torch.float8_e4m3fn, device=a1.device)
            scale_cols = hidden // 128 if hidden >= 128 else 1
            a1_scale = torch.empty(
                (0, scale_cols), dtype=torch.float32, device=a1.device
            )
            return a1_q, a1_scale
        if self.quant_config.is_block_quantized:
            if is_deep_gemm_e8m0_used():
                return sgl_per_token_group_quant_fp8(
                    a1,
                    128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
            return trt_fp8_quantize_128(a1, False)
        if self.quant_config.is_per_act_token:
            a1_q, a1_scale = scaled_fp8_per_token_quant(a1, None)
            assert a1_q.shape[1] % 128 == 0
            a1_scale = a1_scale.repeat(1, a1_q.shape[1] // 128)
            return a1_q, a1_scale
        raise ValueError(
            f"Unsupported FP8 quant config for elastic dispatch: {self.quant_config}"
        )

    def _finalize_post_tp_gather(
        self,
        combined_x: torch.Tensor,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        tp_size = self._tp_size
        original_num_tokens: int = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
        if tp_size > 1:
            if combined_x.size(0) < tp_token_size:
                padding = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding], dim=0)
            gathered = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            combined_x = gathered[:original_num_tokens, :]
        return combined_x

    def _nan_guard(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a1 = torch.where(torch.isnan(a1), torch.zeros_like(a1), a1)

        inv_k = 1.0 / float(self._num_topk)
        topk_weights = torch.where(
            torch.isnan(topk_weights),
            torch.full_like(topk_weights, inv_k),
            topk_weights,
        )

        if topk_ids.size(1) > 1:
            rows = topk_ids.size(0)
            k_topk = topk_ids.size(1)
            rr = (
                torch.arange(
                    rows * k_topk,
                    device=topk_ids.device,
                    dtype=topk_ids.dtype,
                )
                % self._num_experts
            ).reshape(rows, k_topk)
            sorted_ids, _ = torch.sort(topk_ids, dim=-1)
            has_dup = (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(
                dim=-1, keepdim=True
            )
            topk_ids = torch.where(has_dup, rr, topk_ids)

        return a1, topk_ids, topk_weights

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert self._handle is None, "elastic EPHandle leaked from previous step"
        if a1_scale is not None or a2_scale is not None:
            raise ValueError(
                "DeepEpElasticRouter handles fp8 quantization internally; "
                "external a1_scale / a2_scale must be None."
            )

        act_dtype = a1.dtype

        # nan数值校验和重复topk替换
        if self._nan_guard_active:
            a1, topk_ids, topk_weights = self._nan_guard(a1, topk_ids, topk_weights)

        # tp切分
        tp_a1, tp_topk_ids, tp_topk_weights = self._tp_slice(a1, topk_ids, topk_weights)

        # fp8量化
        if self._use_fp8_dispatch:
            x_payload = self._do_quant_fp8(tp_a1)
        else:
            x_payload = tp_a1

        # 启动dispatch的kernel
        recv_x, recv_topk_idx, recv_topk_weights, self._handle, _ = self._buffer.dispatch(
            x=x_payload,
            topk_idx=tp_topk_ids,
            topk_weights=tp_topk_weights,
            num_experts=self._num_experts,
            expert_alignment=self._expert_alignment,
            num_sms=self._elastic_num_sms,
            do_expand=self._do_expand,
            do_cpu_sync=self._do_cpu_sync,
            async_with_compute_stream=False,
        )

        if isinstance(recv_x, tuple):
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        # do_cpu_sync决定是否存在num_recv_tokens_per_expert_list，然后不同的方式生成expert_tokens_meta
        if self._do_cpu_sync:
            expert_num_tokens_cpu = self._handle.num_recv_tokens_per_expert_list
            if len(expert_num_tokens_cpu) == 0:
                expert_num_tokens_cpu = [0] * self._expert_per_rank
            elif len(expert_num_tokens_cpu) != self._expert_per_rank:
                raise AssertionError(
                    f"ElasticBuffer handle.num_recv_tokens_per_expert_list len "
                    f"{len(expert_num_tokens_cpu)} differs from E_local "
                    f"{self._expert_per_rank}; ep_size={self._ep_size}"
                )
            expert_num_tokens_gpu = torch.tensor(
                expert_num_tokens_cpu, device=expert_x.device, dtype=torch.int32
            )
        else:
            psum = self._handle.psum_num_recv_tokens_per_expert
            expert_num_tokens_gpu = torch.diff(
                psum, prepend=torch.zeros(1, device=psum.device, dtype=psum.dtype)
            ).to(torch.int32)
            expert_num_tokens_cpu = None

        # 根据不同要求，填充本地专家ID或者全局专家ID
        if self._use_local_expert_ids:
            expert_topk_ids = recv_topk_idx
        else:
            expert_topk_ids = torch.where(
                recv_topk_idx == -1,
                recv_topk_idx,
                recv_topk_idx + self._rank_expert_offset,
            )

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=act_dtype,
            expert_topk_ids=expert_topk_ids,
            expert_topk_weights=recv_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens_gpu,
                expert_num_tokens_cpu=expert_num_tokens_cpu,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        assert (
            self._handle is not None
        ), "DeepEpElasticRouter.finalize() called without a live EPHandle"

        x = payload.fused_expert_output
        assert (
            x.dtype == torch.bfloat16
        ), f"ElasticBuffer.combine requires bfloat16 input, got {x.dtype}"

        combined_x, _, _ = self._buffer.combine(
            x=x,
            handle=self._handle,
            topk_weights=None,
            num_sms=self._elastic_num_sms,
            async_with_compute_stream=False,
        )
        self._handle = None

        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)

        return combined_x
