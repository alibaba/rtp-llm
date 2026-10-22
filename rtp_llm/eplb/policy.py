"""DeepInfer-style EPLB planning over a history of individual steps."""

import torch


def _batch_greedy_indices(
    tokens_per_expert: torch.Tensor,
    physical_to_logical: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Minimize sum(step peak rank load), with fixed slots per rank.

    Inputs are CPU tensors. Equal loads preserve physical slot order and equal
    placement scores select the lowest rank. The input history is never changed.
    """
    vectors = tokens_per_expert.permute(1, 2, 0).to(torch.float64, copy=True)
    vectors.div_(logical_count.unsqueeze(-1))
    order = (
        vectors.sum(-1)
        .gather(-1, physical_to_logical.long())
        .argsort(dim=-1, descending=True, stable=True)
    )
    expert_order = physical_to_logical.gather(-1, order).long()
    num_layers, _, num_steps = vectors.shape
    num_physical = physical_to_logical.size(1)
    num_ranks = num_physical // capacity
    loads = torch.zeros(
        num_layers, num_ranks, num_steps, dtype=torch.float64, device="cpu"
    )
    peaks = torch.zeros(num_layers, num_steps, dtype=torch.float64, device="cpu")
    occupancy = torch.zeros(num_layers, num_ranks, dtype=torch.int64, device="cpu")
    result = torch.empty_like(order)
    layers = torch.arange(num_layers, device="cpu")
    for job in range(num_physical):
        vector = vectors[layers, expert_order[:, job]]
        scores = torch.maximum(loads + vector.unsqueeze(1), peaks.unsqueeze(1)).sum(-1)
        scores.masked_fill_(occupancy == capacity, float("inf"))
        rank = scores.argmin(-1)
        result[layers, rank * capacity + occupancy[layers, rank]] = order[:, job]
        occupancy[layers, rank] += 1
        loads[layers, rank] += vector
        peaks = torch.maximum(peaks, loads[layers, rank])
    return result


def rebalance_experts_global(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU int32 (physical map, inverse map, replica counts).

    ``tokens_per_expert`` is nonnegative CPU history [step, layer, logical].
    Allocate replicas to minimize the sum of per-step peak replica loads, then
    place them using the per-step rank objective. Tied replication objectives
    prefer the higher current per-replica load, then the lowest logical expert
    ID. No floating epsilon is needed to distinguish tied counts.
    """
    if tokens_per_expert.device.type != "cpu" or tokens_per_expert.ndim != 3:
        raise ValueError("EPLB policy requires CPU history [step, layer, logical]")
    if any(size == 0 for size in tokens_per_expert.shape):
        raise ValueError("EPLB policy history dimensions must be positive")
    if bool((tokens_per_expert < 0).any()):
        raise ValueError("EPLB history counts must be nonnegative")
    _, num_layers, num_logical = tokens_per_expert.shape
    if (
        num_physical_experts < num_logical
        or num_local_physical_experts <= 0
        or num_physical_experts % num_local_physical_experts != 0
    ):
        raise ValueError(
            "EPLB physical experts must cover logical experts and fill ranks"
        )

    redundant = num_physical_experts - num_logical
    physical = torch.empty(
        num_layers, num_physical_experts, dtype=torch.int32, device="cpu"
    )
    physical[:, :num_logical] = torch.arange(num_logical, device="cpu")
    counts = torch.ones(num_layers, num_logical, dtype=torch.int32, device="cpu")
    history = tokens_per_expert.to(torch.float64)
    layers = torch.arange(num_layers, device="cpu")
    for slot in range(num_logical, num_physical_experts):
        scores = history / counts
        # The maximum excluding each candidate: only the first maximum needs
        # replacement by the second maximum, including the equal-maximum case.
        maximum, index = scores.max(-1, keepdim=True)
        without = maximum.expand_as(scores).clone()
        scores.scatter_(-1, index, -float("inf"))
        without.scatter_(-1, index, scores.max(-1, keepdim=True).values)
        objective = torch.maximum(without, history / (counts + 1)).sum(0)
        # With two equally hot experts, copying either leaves the other peak
        # unchanged. Break that tie by load so an idle expert cannot consume
        # every redundant slot just because it has the smallest logical ID.
        replica_load = (history / counts).sum(0)
        replica_load.masked_fill_(
            objective != objective.amin(-1, keepdim=True), -float("inf")
        )
        expert = replica_load.argmax(-1)
        physical[:, slot] = expert.to(torch.int32)
        counts[layers, expert] += 1

    if num_local_physical_experts > 1:
        order = _batch_greedy_indices(
            tokens_per_expert, physical, counts, num_local_physical_experts
        )
        physical = physical.gather(-1, order)

    inverse = torch.full(
        (num_layers, num_logical, redundant + 1), -1, dtype=torch.int32, device="cpu"
    )
    next_slot = torch.zeros_like(counts, dtype=torch.int64)
    for slot in range(num_physical_experts):
        expert = physical[:, slot].long()
        inverse[layers, expert, next_slot[layers, expert]] = slot
        next_slot[layers, expert] += 1
    return physical, inverse, counts
