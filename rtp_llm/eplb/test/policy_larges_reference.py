"""Independent checks for the streaming policy's numerical contract.

The frozen RTP policy remains the old-behavior oracle. These checks replay
the candidate's own decisions, allowing only local FP32 reduction error.
They do not establish instruction-level equivalence of the two kernels.
"""

import math

import torch

GROUPS = (("main", 40, 384, 416, 52), ("dspark", 3, 128, 160, 20))


def history(kind, steps, layers, experts, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = (steps, layers, experts)
    if kind == "zeros":
        return torch.zeros(shape, dtype=torch.int32)
    if kind == "constant":
        return torch.full(shape, 137, dtype=torch.int32)
    if kind == "tail":
        result = torch.zeros(shape, dtype=torch.int32)
        result[-1, :, -1] = 10000
        return result
    if kind == "sparse":
        result = torch.zeros(shape, dtype=torch.int32)
        hot = torch.rand(shape, generator=generator) < 0.05
        result[hot] = torch.randint(
            1, 5000, (int(hot.sum()),), generator=generator, dtype=torch.int32
        )
        return result
    if kind == "zipf":
        ranks = torch.rand(experts, generator=generator).argsort().float()
        popularity = 1.0 / (ranks + 1.0) ** 0.9
        popularity /= popularity.sum()
        per_layer = torch.stack(
            [
                popularity[torch.randperm(experts, generator=generator)]
                for _ in range(layers)
            ]
        )
        variation = 0.5 + torch.rand(shape, generator=generator)
        return (per_layer.unsqueeze(0) * 4096 * variation).to(torch.int32)
    if kind == "bimodal":
        active = max(1, experts // 4)
        ranks = torch.rand(active, generator=generator).argsort().float()
        popularity = 1.0 / (ranks + 1.0) ** 0.9
        popularity /= popularity.sum()
        result = torch.zeros(shape, dtype=torch.int32)
        for layer in range(layers):
            selected = torch.randperm(experts, generator=generator)[:active]
            variation = 0.5 + torch.rand(steps, active, generator=generator)
            result[:, layer, selected] = (
                (popularity.unsqueeze(0) * 4096 * variation)
                .to(torch.int32)
                .clamp(min=25)
            )
        return result
    raise ValueError(kind)


def reduction_depths(steps, layers, replica_chunk, replica_ctas, greedy_chunk):
    splits = max(1, min(math.ceil(steps / replica_chunk), replica_ctas // layers))
    slice_rows = math.ceil(steps / splits)
    # Fixed production chunks: replica=16 and greedy=512, with 4 greedy
    # warps. Triton 3.6 lowers the replica step dimension to 15 sequential
    # fadds, not a log2(16) tree. chunk-1 also bounds any binary sum order.
    assert replica_chunk == 16 and greedy_chunk == 512
    replica_chunk_depth = replica_chunk - 1
    # The greedy [rank, step] layout has threadsPerWarp=[1,32] and
    # warpsPerCTA=[1,4]. Each lane sums four steps (3 fadds), followed by
    # five warp-shuffle levels and two inter-warp levels: 3+5+2=10.
    # This bound depends on that lowering, not merely tl.sum's shape;
    # recheck generated IR if the Triton version or production layout changes.
    greedy_chunk_depth = greedy_chunk // 128 - 1 + 7
    # Initial +0 in each sequential accumulator is exact for nonnegative
    # FP32 operands. Remaining chunk/split additions each add one level.
    replica = (
        replica_chunk_depth + math.ceil(slice_rows / replica_chunk) - 1 + splits - 1
    )
    greedy = greedy_chunk_depth + math.ceil(steps / greedy_chunk) - 1
    return replica, greedy


def assert_local_choice(exact_scores, chosen, depth, label, summands=1):
    """Reject choices outside the two candidates' local reduction budgets.

    Scores are FP64 sums of the same nonnegative FP32 pointwise operands.
    gamma_d bounds relative error of a depth-d FP32 summation path. Both
    chosen and optimal candidates receive their own error budget, with no
    cross-layer/round maximum and no empirical tolerance multiplier.
    """
    assert exact_scores.dtype == torch.float64
    assert bool((exact_scores >= 0).all()), label
    assert bool(((chosen >= 0) & (chosen < exact_scores.size(-1))).all()), label
    selected = exact_scores.gather(-1, chosen.long().unsqueeze(-1)).squeeze(-1)
    best = exact_scores.min(-1).values
    assert bool(
        torch.isfinite(selected).all()
    ), f"{label}: selected unavailable candidate"
    unit = 2.0**-24
    assert 0 <= depth * unit < 1
    gamma = depth * unit / (1 - depth * unit)
    # Bound the diagnostic FP64 sum as well, conservatively using the full
    # summand count rather than assuming it is mathematically exact.
    double_error = summands * 2.0**-53
    assert double_error < 1
    gamma_double = double_error / (1 - double_error)
    budget = (gamma + gamma_double) / (1 - gamma_double) * (selected + best)
    regret = selected - best
    assert bool((regret <= budget).all()), (
        label,
        "choice exceeds local reduction error",
        float((regret - budget).max()),
    )
    return regret, budget


def assert_same_outputs(actual, expected):
    assert len(actual) == len(expected) == 3
    for name, value, oracle in zip(("physical", "inverse", "counts"), actual, expected):
        assert value.dtype == torch.int32, name
        assert value.device == oracle.device and value.shape == oracle.shape, name
        assert torch.equal(value, oracle), name


def replay_decisions(source, physical, local, replica_trace, indices, depths):
    """Rebuild all three outputs from validated candidate decisions.

    The pointwise operations below follow frozen RTP policy_reference.py.
    Only the selected decision is supplied by the candidate. Every round
    is checked, including rounds after an earlier acceptable divergence.
    """
    steps, layers, experts = source.shape
    redundant = physical - experts
    device = source.device
    layer_ids = torch.arange(layers, device=device)
    expert_ids = torch.arange(experts, dtype=torch.int32, device=device)
    counts = torch.ones((layers, experts), dtype=torch.int32, device=device)
    original_physical = expert_ids.expand(layers, -1).clone()
    inverse = torch.full(
        (layers, experts, redundant + 1), -1, dtype=torch.int32, device=device
    )
    inverse[:, :, 0] = expert_ids
    if redundant:
        assert replica_trace is not None
        replicas, slots, final_counts = replica_trace
        assert replicas.shape == slots.shape == (layers, redundant)
        assert replicas.dtype == slots.dtype == torch.int32
        perturbed = source + expert_ids * 1e-4
        for index in range(redundant):
            scores = perturbed / counts
            added = source / (counts + 1)
            values, maxima = scores.max(-1, keepdim=True)
            values = values.expand_as(scores).contiguous()
            scores.scatter_(-1, maxima, added.gather(-1, maxima))
            values.scatter_(-1, maxima, scores.max(-1, keepdim=True).values)
            chosen = replicas[:, index].long()
            assert_local_choice(
                values.double().sum(0),
                chosen,
                depths[0],
                f"replica round {index}",
                steps,
            )
            before = counts[layer_ids, chosen]
            assert torch.equal(slots[:, index], before), "replica slot order"
            inverse[layer_ids, chosen, before.long()] = experts + index
            counts[layer_ids, chosen] += 1
        assert torch.equal(counts, final_counts), "replica counts"
        original_physical = torch.cat((original_physical, replicas), dim=-1)
    else:
        assert replica_trace is None
    if local > 1:
        assert indices is not None
        assert indices.shape == (layers, physical) and indices.dtype == torch.int64
        assert torch.equal(
            indices.sort(-1).values,
            torch.arange(physical, device=device).expand(layers, -1),
        ), "placement must be a permutation"
        weights = source.sum(0, dtype=torch.float32) / counts
        order = weights.gather(-1, original_physical.long()).argsort(
            dim=-1, descending=True, stable=True
        )
        expert_order = original_physical.long().gather(-1, order)
        positions = indices.argsort(-1).gather(-1, order)
        vectors = source.permute(1, 2, 0).to(
            dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
        )
        vectors.div_(counts.unsqueeze(-1))
        ranks = physical // local
        loads = vectors.new_zeros((layers, ranks, steps))
        peaks = vectors.new_zeros((layers, steps))
        occupancy = torch.zeros((layers, ranks), dtype=torch.int64, device=device)
        for index in range(physical):
            vector = vectors[layer_ids, expert_order[:, index]]
            candidates = torch.maximum(loads + vector.unsqueeze(1), peaks.unsqueeze(1))
            exact_scores = candidates.double().sum(-1)
            exact_scores.masked_fill_(occupancy == local, float("inf"))
            chosen = positions[:, index] // local
            assert_local_choice(
                exact_scores, chosen, depths[1], f"greedy round {index}", steps
            )
            assert torch.equal(
                positions[:, index] % local, occupancy[layer_ids, chosen]
            ), "placement slot order"
            occupancy[layer_ids, chosen] += 1
            loads[layer_ids, chosen] += vector
            peaks = torch.maximum(peaks, loads[layer_ids, chosen])
        result_physical = original_physical.gather(-1, indices)
        valid = inverse >= 0
        # Independent inverse remapping; never allow -1 to index a real slot.
        remapped = indices.argsort(-1).gather(
            -1, inverse.clamp_min(0).long().flatten(1)
        )
        inverse = remapped.reshape_as(inverse).to(torch.int32).masked_fill(~valid, -1)
    else:
        assert indices is None
        result_physical = original_physical
    return result_physical, inverse, counts
