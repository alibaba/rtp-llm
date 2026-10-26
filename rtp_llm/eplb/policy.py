"""EPLB policy from DeepInfer 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9.

Adapted from deepinfer/model_executor/eplb/policy.py (Apache-2.0). Planning
stays on the input device and preserves the reference's floating-point
operations, tie behavior and logical-to-physical replica ordering.
"""

import torch


def _batch_greedy_indices(
    tokens_per_expert: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Place replicas to minimize the sum of per-step peak rank loads.

    Inputs share a device and describe the same layers and experts. Physical
    experts fill ranks of ``capacity`` slots. Equal loads preserve original
    physical slot order; equal placement scores select the lowest rank.
    """
    weights = tokens_per_expert.sum(dim=0, dtype=torch.float32) / logical_count
    physical_experts = physical_to_logical_map.to(torch.int64)
    physical_order = weights.gather(-1, physical_experts).argsort(
        dim=-1, descending=True, stable=True
    )
    expert_order = physical_experts.gather(-1, physical_order)
    vectors = tokens_per_expert.permute(1, 2, 0).to(
        dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
    )
    vectors.div_(logical_count.unsqueeze(-1))

    num_layers, _, num_steps = vectors.shape
    num_physical = physical_order.size(1)
    num_ranks = num_physical // capacity
    loads = vectors.new_zeros((num_layers, num_ranks, num_steps))
    peaks = vectors.new_zeros((num_layers, num_steps))
    occupancy = torch.zeros(
        (num_layers, num_ranks), dtype=torch.int64, device=vectors.device
    )
    candidates = torch.empty_like(loads)
    scores = vectors.new_empty((num_layers, num_ranks))
    indices = torch.empty_like(physical_order)
    layers = torch.arange(num_layers, device=vectors.device)
    for job_index in range(num_physical):
        experts = expert_order[:, job_index]
        vector = vectors[layers, experts]
        torch.add(loads, vector.unsqueeze(1), out=candidates)
        torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)
        torch.sum(candidates, dim=-1, out=scores)
        scores.masked_fill_(occupancy == capacity, float("inf"))
        rank = scores.argmin(dim=-1)
        occupied = occupancy[layers, rank]
        indices[layers, rank * capacity + occupied] = physical_order[:, job_index]
        occupancy[layers, rank] = occupied + 1
        updated = loads[layers, rank] + vector
        loads[layers, rank] = updated
        torch.maximum(peaks, updated, out=peaks)
    return indices


def rebalance_experts_global(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return int32 physical map, inverse map and replica counts on the input device.

    History is nonnegative [step, layer, logical], normally CUDA int32 counts.
    Physical experts cover all logical experts and fill equally sized ranks.
    The caller guarantees count values; checking them here would synchronize
    CUDA. Inverse-map replicas retain their original allocation order.

    Contiguous NVIDIA CUDA int32 histories with 1, 2 or 4 steps and the FP32
    default dtype use fused planning. Other inputs keep the Torch algorithm:
    its division dtype and reduction order are part of the placement contract.
    """
    if tokens_per_expert.ndim != 3 or any(
        size == 0 for size in tokens_per_expert.shape
    ):
        raise ValueError("EPLB requires nonempty history [step, layer, logical]")
    num_logical_experts = tokens_per_expert.shape[2]
    if (
        num_physical_experts < num_logical_experts
        or num_local_physical_experts <= 0
        or num_physical_experts % num_local_physical_experts != 0
    ):
        raise ValueError(
            "EPLB physical experts must cover logical experts and fill ranks"
        )
    if (
        tokens_per_expert.is_cuda
        and tokens_per_expert.dtype == torch.int32
        and tokens_per_expert.shape[0] in (1, 2, 4)
        and tokens_per_expert.is_contiguous()
        and torch.get_default_dtype() == torch.float32
    ):
        from rtp_llm.device.device_type import is_cuda

        if is_cuda():
            from rtp_llm.eplb.policy_fused import rebalance_experts_global_fused

            return rebalance_experts_global_fused(
                tokens_per_expert, num_physical_experts, num_local_physical_experts
            )
    return _rebalance_experts_global_reference(
        tokens_per_expert, num_physical_experts, num_local_physical_experts
    )


def _rebalance_experts_global_reference(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch implementation for other devices, dtypes, layouts and windows."""
    _, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundant_experts = num_physical_experts - num_logical_experts
    device = tokens_per_expert.device

    physical_to_logical_map = torch.empty(
        num_moe_layers, num_physical_experts, dtype=torch.int, device=device
    )
    logical_to_physical_map = torch.full(
        (num_moe_layers, num_logical_experts, num_redundant_experts + 1),
        -1,
        dtype=torch.int,
        device=device,
    )
    logical_count = torch.ones(
        num_moe_layers, num_logical_experts, dtype=torch.int, device=device
    )
    arange_layers = torch.arange(num_moe_layers, dtype=torch.int, device=device)
    arange_logical = torch.arange(num_logical_experts, dtype=torch.int, device=device)
    physical_to_logical_map[:, :num_logical_experts] = arange_logical
    logical_to_physical_map[:, :, 0] = arange_logical

    # Preserve DeepInfer's epsilon tie handling, including its FP32 limitation:
    # large equal loads can round away epsilon and allocate an idle replica.
    tokens_per_expert_all_diff = tokens_per_expert + arange_logical * 1e-4
    for i in range(num_redundant_experts):
        score = tokens_per_expert_all_diff / logical_count
        score_added = tokens_per_expert / (logical_count + 1)
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, indices, score_added.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        redundancy_indices = values.sum(0).argmin(-1)

        physical_index = num_logical_experts + i
        physical_to_logical_map[:, physical_index] = redundancy_indices
        redundancy_count = logical_count.gather(
            -1, redundancy_indices.unsqueeze(-1)
        ).squeeze(-1)
        logical_to_physical_map[arange_layers, redundancy_indices, redundancy_count] = (
            physical_index
        )
        logical_count[arange_layers, redundancy_indices] += 1

    if num_local_physical_experts > 1:
        indices = _batch_greedy_indices(
            tokens_per_expert,
            physical_to_logical_map,
            logical_count,
            num_local_physical_experts,
        )
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        # Remap old slots without reordering a logical expert's replicas.
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(
                -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
            )
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count
