from typing import Any, Optional, Tuple

import deep_ep
import torch
import torch.distributed as dist

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.models_py.modules.moe import TopKWeightAndReduceDelegate


class FakeExpert(mm.FusedMoeExpertExecutor):
    def __init__(self):
        super().__init__(None)

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def execute(self, payload: mm.ExpertForwardPayload, *args, **kwargs):
        # The model rank should not receive any tokens to process.
        assert (
            payload.expert_x.numel() == 0
        ), "FakeExpert should not receive any input tokens"
        # It must return a tensor of the correct shape for the finalize step, even if empty.
        return torch.empty_like(payload.expert_x)


class AfdCommMixin:
    """
    A Mixin that provides common data router initialization for deep_ep
    communication, including buffer setup.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        num_attn_ranks: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden_dim: int,
        group: Optional[dist.ProcessGroup],
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        assert group is not None
        self.group: dist.ProcessGroup = group

        self.num_attn_ranks = num_attn_ranks

        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint_m2n(
            num_max_dispatch_tokens_per_rank,
            hidden_dim,
            world_size,
            num_experts,
            num_attn_ranks,
        )

        self.buffer = deep_ep.Buffer(
            self.group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=max(1, num_experts // (world_size - num_attn_ranks)),
            allow_nvlink_for_low_latency_mode=False,
        )

        self.num_experts = num_experts
        self.comm_handle = None
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank

        self.last_send_event: Optional[deep_ep.EventOverlap] = None


class AfdDataRouterAttn(AfdCommMixin):
    def __init__(
        self,
        rank: int,
        world_size: int,
        num_attn_ranks: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden_dim: int,
        group: Optional[dist.ProcessGroup],
    ):
        super().__init__(
            rank,
            world_size,
            num_attn_ranks,
            num_experts,
            num_max_dispatch_tokens_per_rank,
            hidden_dim,
            group,
        )
        self.workspace: Optional[torch.Tensor] = None
        self.prefetch_output: Optional[torch.Tensor] = None
        self.cached_topk_ids: Optional[torch.Tensor] = None
        self.cached_topk_weights: Optional[torch.Tensor] = None

        self.hidden_dim = hidden_dim
        self.is_last_layer = False

    def fetch(self) -> torch.Tensor:
        assert self.cached_topk_ids is not None
        assert self.cached_topk_weights is not None
        assert self.comm_handle is not None

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()
            self.last_send_event = None

        num_tokens, num_topk = self.cached_topk_ids.size()

        _, event, _ = self.buffer.low_latency_combine_recv(
            self.cached_topk_ids,
            self.cached_topk_weights,
            self.comm_handle,
            self.num_attn_ranks,
            num_tokens,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
            out=self.workspace,
        )
        event.current_stream_wait()
        self.comm_handle = None
        self.cached_topk_ids = None
        self.cached_topk_weights = None

        output = self.workspace

        assert isinstance(output, torch.Tensor)

        return output

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,  # This is the total number of experts in the system
        quant_config: Any,
    ):
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)

        self.workspace = torch.zeros_like(a1)

        if self.comm_handle is not None:
            self.prefetch_output = self.fetch()

        num_tokens, hidden_dim = a1.size()
        num_topk = topk_ids.size(1)

        assert num_tokens <= self.num_max_dispatch_tokens_per_rank
        assert self.rank < self.num_attn_ranks, ""

        _, _, handle, event, _ = self.buffer.low_latency_dispatch_send(
            a1,
            topk_ids,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_attn_ranks,
            num_topk,
            use_fp8=False,
            async_finish=True,
            return_recv_hook=False,
        )
        self.last_send_event = event
        # event.current_stream_wait()
        self.comm_handle = handle
        self.cached_topk_ids = topk_ids
        self.cached_topk_weights = topk_weights

    def finalize(
        self,
    ) -> torch.Tensor:
        if self.prefetch_output is not None:
            output = self.prefetch_output
            self.prefetch_output = None
            return output
        return self.fetch()


class AfdDataRouterFfn(AfdCommMixin, mm.FusedMoeDataRouter):
    def __init__(
        self,
        rank: int,
        world_size: int,
        num_attn_ranks: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden_dim: int,
        group: Optional[dist.ProcessGroup],
    ):
        mm.FusedMoeDataRouter.__init__(self)
        AfdCommMixin.__init__(
            self,
            rank,
            world_size,
            num_attn_ranks,
            num_experts,
            num_max_dispatch_tokens_per_rank,
            hidden_dim,
            group,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,  # This is the total number of experts in the system
        quant_config: Any,
    ) -> mm.ExpertForwardPayload:
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        assert (
            self.comm_handle is None
        ), "Communication handle should be clean before prepare()."

        num_tokens, hidden_dim = a1.size()
        num_topk = topk_ids.size(1)

        assert num_tokens <= self.num_max_dispatch_tokens_per_rank

        assert self.rank >= self.num_attn_ranks

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()
            self.last_send_event = None

        packed_recv_x, packed_recv_count, handle, event, _ = (
            self.buffer.low_latency_dispatch_recv(
                hidden_dim,
                num_topk,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                self.num_attn_ranks,
                use_fp8=False,
                async_finish=True,
                return_recv_hook=False,
            )
        )
        event.current_stream_wait()
        assert isinstance(packed_recv_x, torch.Tensor)

        # N ranks wait for received data to build the payload for local experts.
        expert_tokens_meta = mm.ExpertTokensMetadata(
            expert_num_tokens=packed_recv_count, expert_num_tokens_cpu=None
        )
        assert isinstance(packed_recv_x, torch.Tensor)
        payload = mm.ExpertForwardPayload(
            expert_x=packed_recv_x,
            expert_x_scale=None,
            expert_tokens_meta=expert_tokens_meta,
        )

        self.comm_handle = handle

        return payload

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: Any,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert (
            self.comm_handle is not None
        ), "Communication handle is missing for finalize()."

        handle = self.comm_handle

        _, num_topk = topk_ids.size()

        _, event, _ = self.buffer.low_latency_combine_send(
            fused_expert_output,
            handle,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
        )
        self.last_send_event = event

        self.comm_handle = None

        return torch.empty(0)
