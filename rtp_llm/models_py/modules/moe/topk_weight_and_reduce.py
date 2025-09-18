import torch

import rtp_llm.models_py.modules.moe.fused_moe as mm


class TopKWeightAndReduceDelegate(mm.TopKWeightAndReduce):
    """
    A placeholder implementation of TopKWeightAndReduce.

    This class intentionally does not implement the weight-application and reduction
    logic. Instead, it serves as a placeholder to defer the choice of the actual
    implementation to the calling component.

    This is useful when a single expert module must be compatible with different
    execution strategies, some of which handle the final reduction step internally
    while others require an external implementation.
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceDelegate)

    def apply(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError(
            "The caller is expected to choose an appropriate "
            "TopKWeightAndReduce implementation."
        )


class TopKWeightAndReduceNaiveBatched(mm.TopKWeightAndReduce):
    def __init__(self, rank: int):
        self.rank = rank

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceNaiveBatched) and (
            other.rank == self.rank
        )

    def apply(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        assert fused_expert_output.ndim == 3
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        K = fused_expert_output.size(-1)

        output = torch.zeros(
            (num_tokens, K),
            device=fused_expert_output.device,
            dtype=fused_expert_output.dtype,
        )

        assert output.size() == (
            num_tokens,
            K,
        ), f"Expected output size {(num_tokens, K)}, but got {output.size()}"

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            matching_tokens = topk_ids == expert_id
            topks = torch.any(matching_tokens, dim=1).flatten()
            rows = torch.count_nonzero(topks)
            rhs = fused_expert_output[expert_id - first_expert, :rows, :]
            if not apply_router_weight_on_input:
                rhs.mul_(topk_weights[matching_tokens].view(rhs.size(0), 1))
            output[topks] = output[topks] + rhs

        return output


class TopKWeightAndReduceContiguous(mm.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (m, topk, K)
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceContiguous)

    def apply(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:

        m, num_topk = topk_ids.size()
        k = fused_expert_output.size(-1)
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(m, num_topk, k)

        assert fused_expert_output.size() == (m, num_topk, k), (
            f"Expected fused_expert_output size {(m, num_topk, k)}. But got "
            f"{fused_expert_output.size()}"
        )

        if not apply_router_weight_on_input:
            fused_expert_output.mul_(topk_weights.view(m, -1, 1))

        output = torch.empty(
            (m, k),
            device=fused_expert_output.device,
            dtype=fused_expert_output.dtype,
        )
        torch.sum(fused_expert_output, dim=1, out=output)
        return output
