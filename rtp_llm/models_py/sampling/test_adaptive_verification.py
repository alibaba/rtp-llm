import itertools
import math
import unittest

import numpy as np
import torch

from rtp_llm.models_py.sampling.adaptive_verification import (
    AdaptiveBudgetInputError,
    allocate_draft_token_budget,
    assign_draft_token_budget,
    select_draft_token_budget,
)


def reference_allocation(
    probabilities: torch.Tensor, capacities: torch.Tensor, budget: int
) -> torch.Tensor:
    scores = probabilities.cpu().cumprod(-1)
    lengths = capacities.cpu().tolist()
    selected = [0] * len(lengths)
    for _ in range(budget):
        eligible = [i for i, length in enumerate(lengths) if selected[i] < length]
        request = max(eligible, key=lambda i: (scores[i, selected[i]].item(), -i))
        selected[request] += 1
    return torch.tensor(selected, dtype=capacities.dtype)


class BudgetSelectionTest(unittest.TestCase):
    def test_budget_matches_exhaustive_prefix_search(self) -> None:
        probabilities = np.array([[0.75, 0.5, 0.25], [0.5, 1.0, 0.75]])
        draft_costs = np.array([0.0, 0.1, 0.3])
        verify_costs = np.array([0.1, 0.2, 1.0, 1.1, 1.5, 3.0, 4.0, 5.0, 6.0])
        for lengths in ((0, 0), (0, 3), (1, 1), (3, 3)):
            for limit in range(sum(lengths) + 1):
                with self.subTest(capacities=lengths, limit=limit):
                    options = []
                    for allocation in itertools.product(
                        *(range(n + 1) for n in lengths)
                    ):
                        budget = sum(allocation)
                        if budget > limit:
                            continue
                        expected = 2.0 + math.fsum(
                            math.prod(probabilities[row, : step + 1])
                            for row, length in enumerate(allocation)
                            for step in range(length)
                        )
                        options.append(
                            (expected / (draft_costs[2] + verify_costs[2 + budget]), -budget)
                        )
                    expected_budget = -max(options)[1]

                    actual = select_draft_token_budget(
                        probabilities,
                        np.array(lengths),
                        draft_cost_ms=draft_costs,
                        verify_cost_ms=verify_costs,
                        num_non_draft_tokens=2,
                        num_sampling_requests=2,
                        max_draft_budget=limit,
                    )

                    self.assertEqual(actual, expected_budget)

    def test_zero_full_budget_and_smallest_budget_on_throughput_ties(self) -> None:
        probabilities, capacities = np.ones((1, 2)), np.array([2])
        for costs, expected in (([1.0, 1.0, 2.0, 3.0], 0), ([1.0] * 4, 2)):
            with self.subTest(costs=costs):
                actual = select_draft_token_budget(
                    probabilities,
                    capacities,
                    draft_cost_ms=np.zeros(2),
                    verify_cost_ms=np.array(costs),
                    num_non_draft_tokens=1,
                    num_sampling_requests=1,
                )
                self.assertEqual(actual, expected)

    def test_measured_draft_cost_changes_the_best_budget(self) -> None:
        results = []
        for draft_ms in (0.0, 10.0):
            results.append(
                select_draft_token_budget(
                    np.array([[0.1]]),
                    np.array([1]),
                    draft_cost_ms=np.array([0.0, draft_ms]),
                    verify_cost_ms=np.array([1.0, 1.0, 1.2]),
                    num_non_draft_tokens=1,
                    num_sampling_requests=1,
                )
            )
        self.assertEqual(results, [0, 1])

    def test_unused_cost_table_entries_do_not_need_measurements(self) -> None:
        actual = select_draft_token_budget(
            np.array([[0.5]]),
            np.array([1]),
            draft_cost_ms=np.array([np.nan, 0.0, np.nan]),
            verify_cost_ms=np.array([0.0, 1.0, 1.0, np.nan]),
            num_non_draft_tokens=1,
            num_sampling_requests=1,
        )
        self.assertEqual(actual, 1)

    def test_rejects_invalid_inputs_and_costs(self) -> None:
        invalid = (
            ("confidence_probs", np.array([[np.nan]])),
            ("confidence_probs", np.array([[np.inf]])),
            ("confidence_probs", np.array([[-0.1]])),
            ("confidence_probs", np.array([[1.1]])),
            ("confidence_probs", np.array([0.5])),
            ("capacities", np.array([0.5])),
            ("capacities", np.array([True])),
            ("capacities", np.array([-1])),
            ("capacities", np.array([2])),
            ("capacities", np.array([0, 1])),
            ("num_non_draft_tokens", -1),
            ("num_non_draft_tokens", 0.5),
            ("num_sampling_requests", 2),
            ("num_sampling_requests", True),
            ("max_draft_budget", -1),
            ("max_draft_budget", 0.5),
            ("draft_cost_ms", np.array([0.0])),
            ("draft_cost_ms", np.array([0.0, -1.0])),
            ("draft_cost_ms", np.array([0.0, np.inf])),
            ("verify_cost_ms", np.ones(2)),
            ("verify_cost_ms", np.array([1.0, 0.0, 1.0])),
            ("verify_cost_ms", np.array([1.0, 1.0, np.nan])),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                arguments = dict(
                    confidence_probs=np.array([[0.5]]),
                    capacities=np.array([1]),
                    draft_cost_ms=np.zeros(2),
                    verify_cost_ms=np.ones(3),
                    num_non_draft_tokens=1,
                    num_sampling_requests=1,
                )
                arguments[field] = value
                with self.assertRaises(AdaptiveBudgetInputError):
                    select_draft_token_budget(**arguments)
        with self.assertRaisesRegex(AdaptiveBudgetInputError, "total_cost_ms"):
            select_draft_token_budget(
                np.array([[0.5]]),
                np.array([1]),
                draft_cost_ms=np.full(2, np.finfo(np.float64).max),
                verify_cost_ms=np.full(3, np.finfo(np.float64).max),
                num_non_draft_tokens=1,
                num_sampling_requests=1,
            )


class DraftAllocationTest(unittest.TestCase):
    def test_allocation_matches_reference_including_zero_and_full(self) -> None:
        for requests in (1, 17, 300):
            generator = torch.Generator(device="cuda").manual_seed(20261109)
            confidence = torch.rand(requests + 5, 3, device="cuda", generator=generator)
            mapping = torch.arange(requests, device="cuda").flip(0) + 5
            scheduled = (
                torch.arange(requests, device="cuda", dtype=torch.int32) + 2
            ) % 4
            available = int(scheduled.sum().item())
            scratch = torch.empty(requests * 3, dtype=torch.int32, device="cuda")
            for budget in (0, available // 2, available):
                with self.subTest(requests=requests, budget=budget):
                    actual = scheduled.clone()
                    expected = reference_allocation(
                        confidence[mapping], scheduled, budget
                    )

                    allocate_draft_token_budget(
                        confidence, mapping, actual, budget, scratch
                    )

                    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_ties_follow_request_order_across_blocks_and_keep_prefixes(self) -> None:
        confidence = torch.ones(300, 3, device="cuda")
        mapping = torch.arange(300, device="cuda").flip(0)
        capacities = torch.full((300,), 3, dtype=torch.int32, device="cuda")
        scratch = torch.empty(900, dtype=torch.int32, device="cuda")

        allocate_draft_token_budget(confidence, mapping, capacities, 800, scratch)

        self.assertEqual(capacities.tolist(), [3] * 266 + [2] + [0] * 33)

    def test_padding_mapping_is_rejected_before_kernel_launch(self) -> None:
        confidence = torch.ones(2, 3, device="cuda")
        capacities = torch.ones(2, dtype=torch.int32, device="cuda")
        scratch = torch.empty(6, dtype=torch.int32, device="cuda")
        for slots in ([0, -1], [0, 2]):
            with self.subTest(slots=slots), self.assertRaisesRegex(
                AdaptiveBudgetInputError, "idx_mapping"
            ):
                allocate_draft_token_budget(
                    confidence,
                    torch.tensor(slots, device="cuda"),
                    capacities,
                    1,
                    scratch,
                )

    def test_raw_kernel_graph_replay_reads_changed_inputs(self) -> None:
        confidence = torch.full((5, 3), 0.5, device="cuda")
        mapping = torch.tensor([3, 1, 4], device="cuda")
        capacities = torch.full((3,), 3, dtype=torch.int32, device="cuda")
        scratch = torch.empty(9, dtype=torch.int32, device="cuda")
        for _ in range(3):
            capacities.fill_(3)
            assign_draft_token_budget(confidence, mapping, capacities, 3, scratch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assign_draft_token_budget(confidence, mapping, capacities, 3, scratch)
        confidence.copy_(torch.linspace(0.1, 0.9, 15, device="cuda").view(5, 3))
        mapping.copy_(torch.tensor([0, 4, 2], device="cuda"))
        scheduled = torch.tensor([0, 3, 2], dtype=torch.int32, device="cuda")
        capacities.copy_(scheduled)
        expected = reference_allocation(confidence[mapping], scheduled, 3)

        graph.replay()

        torch.testing.assert_close(capacities.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
