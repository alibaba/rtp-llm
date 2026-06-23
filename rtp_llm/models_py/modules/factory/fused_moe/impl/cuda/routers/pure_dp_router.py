"""Pure DP router using allgather + reduce_scatter.

Replaces the per-token all_reduce pattern in PureTpRouter with a communication
pattern designed for Data Parallel (DP) MoE:

  prepare:  allgather(scattered_tokens, DP_AND_TP) -> full_tokens
  execute:  MoE on full_tokens (each rank computes partial sums)
  finalize: reduce_scatter(partial_output, DP_AND_TP) -> scattered_output

This avoids DeepEP entirely and uses standard NCCL collectives.

In DP mode, different ranks may have different batch sizes (real requests vs
fake/empty batches). Since NCCL allgather requires equal tensor sizes across
ranks, this router pads tensors to the max batch size before allgather and
trims the output after reduce_scatter.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    reduce_scatter,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
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
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class PureDpRouterBase(FusedMoeDataRouter):
    """Base class for pure DP routers using allgather + reduce_scatter.

    Instead of all_reduce at finalize time, this router:
    - In prepare(): allgathers scattered tokens across DP_AND_TP group
    - In finalize(): reduce_scatters the partial MoE output back

    In DP mode, ranks may have different batch sizes (real vs fake streams).
    Tensors are padded to the max batch size before allgather and trimmed after
    reduce_scatter.
    """

    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        # Pure DP + EP only: physical tp must be 1, dp > 1, ep == dp.
        # Read raw parallelism_config.tp_size so future CP+DP topologies
        # (where adapter tp_size==1 but physical tp>1) aren't misclassified.
        # Mixed tp>1+dp>1 (i.e. tp*dp == ep with tp>1) is intentionally
        # routed back to DeepEP.
        checker.check(config.parallelism_config.tp_size == 1)
        checker.check(config.dp_size > 1)
        checker.check(config.ep_size == config.dp_size)
        checker.check(resolver.use_all_gather(config))
        # CUDA Graph capture/replay freezes shapes; PureDP _pad_to_max derives
        # max_n via runtime all_gather + .item(), which is graph-unsafe (D2H
        # sync mid-capture, dynamic pad shape on replay). Force fallback to
        # DeepEP when CUDA Graph is on.
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        self.tp_size = config.tp_size
        self.dp_size = config.dp_size
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.expert_start_id = self.ep_rank * self.expert_num_per_rank
        self._reduce_scatter_output: Optional[torch.Tensor] = None

    @abstractmethod
    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def _pad_to_max(
        self, a1: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad tensors to the max batch size across all EP ranks.

        Returns the padded tensors plus the max batch size. The local (unpadded)
        batch size is recovered in finalize() from extra_finalize_args
        ("original_num_tokens"), so this method holds no instance state.
        """
        local_n = a1.shape[0]

        # TODO(perf): .item() forces a D2H sync on every MoE layer's hot path.
        # Plan to lift max_n to step-level host metadata (computed once before
        # entering the MoE stack) in a follow-up PR. Tracked on PR #968 review.
        n_tensor = torch.tensor([local_n], device=a1.device, dtype=torch.int64)
        all_n = all_gather(n_tensor, group=Group.DP_AND_TP)
        max_n = int(all_n.max().item())

        if local_n < max_n:
            pad_n = max_n - local_n
            a1 = F.pad(a1, (0, 0, 0, pad_n))
            topk_weights = F.pad(topk_weights, (0, 0, 0, pad_n))
            topk_ids = F.pad(topk_ids, (0, 0, 0, pad_n), value=-1)

        return a1, topk_weights, topk_ids, max_n

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert a1_scale is None and a2_scale is None, "not support quanted moe"

        a1, topk_weights, topk_ids, _ = self._pad_to_max(a1, topk_weights, topk_ids)

        a1_full = all_gather(a1, group=Group.DP_AND_TP)
        topk_weights_full = all_gather(topk_weights, group=Group.DP_AND_TP)
        topk_ids_full = all_gather(topk_ids, group=Group.DP_AND_TP)

        expert_x, expert_x_scale = self._do_quant(a1_full)

        adjusted_topk_ids, num_recv_tokens_per_expert = (
            recompute_topk_ids_sum_expert_count(
                topk_ids_full, self.expert_start_id, self.expert_num_per_rank
            )
        )

        return ExpertForwardPayload(
            expert_x,
            a1.dtype,
            expert_x_scale,
            ExpertTokensMetadata(None, num_recv_tokens_per_expert, None),
            adjusted_topk_ids,
            topk_weights_full,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        # Read the local batch size from extra_finalize_args (injected by
        # FusedMoeDataRouter.forward as original_num_tokens) instead of relying
        # on cross-call instance state. Same pattern as deepep_normal_router.
        assert (
            extra_finalize_args is not None
            and "original_num_tokens" in extra_finalize_args
        ), "PureDpRouter.finalize requires extra_finalize_args['original_num_tokens']"
        local_batch_size: int = extra_finalize_args["original_num_tokens"]

        input_tensor = payload.fused_expert_output
        world_size = self.dp_size * self.tp_size
        expected_shape = [input_tensor.shape[0] // world_size] + list(
            input_tensor.shape[1:]
        )
        if (
            self._reduce_scatter_output is None
            or tuple(self._reduce_scatter_output.shape) != tuple(expected_shape)
            or self._reduce_scatter_output.dtype != input_tensor.dtype
            or self._reduce_scatter_output.device != input_tensor.device
        ):
            self._reduce_scatter_output = torch.empty(
                expected_shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        output = reduce_scatter(
            input_tensor,
            group=Group.DP_AND_TP,
            output_tensor=self._reduce_scatter_output,
        )
        if output.shape[0] > local_batch_size:
            output = output[:local_batch_size]
        return output


class PureDpRouterFp8PerBlock(PureDpRouterBase):
    """Pure DP router with FP8 per-block quantization."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if is_deep_gemm_e8m0_used():
            return sgl_per_token_group_quant_fp8(
                a1,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            return trt_fp8_quantize_128(a1, False)
