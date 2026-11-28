"""Exact stdlib oracle for DeepInfer's sum(max rank load) / sum(mean rank load)."""

from fractions import Fraction
from typing import List, Sequence


class StepRecorder:
    """Reference histogram for synthetic routing tests, including routed padding."""

    def __init__(self, num_layers: int, num_physical_experts: int):
        self.num_layers = num_layers
        self.num_physical_experts = num_physical_experts
        self._counts = [[0] * num_physical_experts for _ in range(num_layers)]

    def record(self, layer: int, expert_ids: Sequence[int]) -> None:
        assert 0 <= layer < self.num_layers
        for expert in expert_ids:
            assert 0 <= expert < self.num_physical_experts
            self._counts[layer][expert] += 1

    def snapshot(self) -> List[List[int]]:
        return [list(row) for row in self._counts]


def sum_histograms(hists: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """SUM routing-source histograms before assigning destination ownership."""
    if not hists:
        return []
    layers, experts = len(hists[0]), len(hists[0][0])
    result = [[0] * experts for _ in range(layers)]
    for hist in hists:
        assert len(hist) == layers
        for layer, row in enumerate(hist):
            assert len(row) == experts
            for expert, count in enumerate(row):
                result[layer][expert] += count
    return result


def reduce_compact_window(rank_samples, layer_count, ep_size):
    """Reduce synchronous [rank][step][layer] sums, including routed padding."""
    world_size = len(rank_samples)
    if layer_count <= 0 or not world_size or ep_size not in (1, world_size):
        raise ValueError("invalid expert load dimensions or EP layout")
    sample_count = len(rank_samples[0])
    for window in rank_samples:
        if len(window) != sample_count:
            raise ValueError("rank window sizes differ")
        for row in window:
            if len(row) != layer_count or any(type(x) is not int or x < 0 for x in row):
                raise ValueError("rows require nonnegative layer sums")

    def metrics(rank_totals, maximum):
        total = sum(rank_totals)
        return {
            "moe_lb": Fraction(maximum * ep_size, total) if total else Fraction(1),
            "total_activations": total,
            "max_rank_activations": maximum,
            "rank_activations": rank_totals,
        }

    layers = []
    for layer in range(layer_count):
        totals = [0] * ep_size
        maxima = 0
        for step in range(sample_count):
            counts = [0] * ep_size
            for rank, window in enumerate(rank_samples):
                counts[rank % ep_size] += window[step][layer]
            maxima += max(counts)
            totals = [old + new for old, new in zip(totals, counts)]
        layers.append(dict(layer_id=layer, **metrics(totals, maxima)))
    aggregate = metrics(
        [
            sum(layer["rank_activations"][rank] for layer in layers)
            for rank in range(ep_size)
        ],
        sum(layer["max_rank_activations"] for layer in layers),
    )
    return dict(aggregate, layers=layers)


if __name__ == "__main__":
    assert reduce_compact_window([[[4], [2]], [[4], [0]]], 1, 2)["moe_lb"] == Fraction(
        6, 5
    )
    print("ORACLE SELF-CHECK PASS")
