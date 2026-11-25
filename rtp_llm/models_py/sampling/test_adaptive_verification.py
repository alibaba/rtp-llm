import itertools
import math
import unittest

import numpy as np
import torch

from rtp_llm.models_py.sampling.adaptive_verification import (
    _FOUR_GAP_SURVIVAL_COEFFICIENTS,
    AdaptiveBudgetInputError,
    allocate_draft_token_budget,
    assign_draft_token_budget,
    pack_ep_margin_rows,
    pack_verification_rows,
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
    def test_draft5_workload_budget_prices_padded_rows_and_fixed_cost(self) -> None:
        # Synthetic heterogeneous workload: about 3.175 output tokens/round.
        # These are cost-model invariants, not measured GLM speedups.
        for batch in (16, 24, 30):
            confidence = np.array(
                [[p] * 5 for p in [0.95] * (batch // 2) + [0.05] * (batch // 2)]
            )
            capacities = np.full(batch, 5, dtype=np.int32)
            scores = np.sort(np.cumprod(confidence, axis=1).reshape(-1))[::-1]
            expected = np.r_[batch, batch + np.cumsum(scores)]
            self.assertAlmostEqual(expected[-1] / batch, 3.175396875)
            padded = ((np.arange(6 * batch + 1) + 15) // 16) * 16
            for draft_cost in (0.0, 5.0, 500.0):
                target = 0.1 + padded * 0.1
                actual = select_draft_token_budget(
                    confidence,
                    capacities,
                    draft_cost_ms=np.full(batch + 1, draft_cost),
                    verify_cost_ms=target,
                    num_non_draft_tokens=batch,
                    num_sampling_requests=batch,
                )
                ratios = expected / (draft_cost + target[batch:])
                self.assertEqual(actual, int(np.argmax(ratios)))
                self.assertGreaterEqual(ratios[actual], ratios[-1])
                # There is no reason to leave a positive-value draft unverified
                # when its row is free inside the selected execution bucket.
                if actual < 5 * batch:
                    self.assertNotEqual(
                        padded[batch + actual], padded[batch + actual + 1]
                    )
            # If measured execution cost is flat, trimming cannot buy throughput.
            self.assertEqual(
                select_draft_token_budget(
                    confidence,
                    capacities,
                    draft_cost_ms=np.ones(batch + 1),
                    verify_cost_ms=np.ones(6 * batch + 1),
                    num_non_draft_tokens=batch,
                    num_sampling_requests=batch,
                ),
                5 * batch,
            )

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
                            (
                                expected / (draft_costs[2] + verify_costs[2 + budget]),
                                -budget,
                            )
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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MarginPrefixPackingTest(unittest.TestCase):
    def test_exact_prefix_and_fallback_for_padded_rank_shapes(self):
        batch = 14
        prior = torch.full((batch, 5), 0.5, device="cuda")
        gaps = torch.arange(batch, device="cuda", dtype=torch.float32).repeat(4, 1)
        fallback = torch.tensor([5] * 7 + [2] * 7, device="cuda", dtype=torch.int32)
        width = batch + 49
        caps, rows = pack_ep_margin_rows(gaps, prior, fallback, width)
        actual = caps.tolist()
        self.assertEqual(sum(actual) + batch, width)
        self.assertGreater(sum(actual[7:]), sum(actual[:7]))
        self.assertEqual(
            rows.tolist(),
            [req * 6 + hop for req, cap in enumerate(actual) for hop in range(cap + 1)],
        )
        self.assertTrue(all(1 <= cap <= 5 for cap in actual))
        for invalid in (float("inf"), float("nan"), -1.0):
            bad = gaps.clone()
            bad[0, 0] = invalid
            result, retained = pack_ep_margin_rows(bad, prior, fallback, width)
            self.assertTrue(torch.equal(result, fallback))
            self.assertEqual(
                retained.tolist(),
                [
                    req * 6 + hop
                    for req, cap in enumerate(fallback.tolist())
                    for hop in range(cap + 1)
                ],
            )

    def test_one_batch_supports_varying_runtime_widths(self):
        batch = 14
        prior = torch.full((batch, 5), 0.5, device="cuda")
        gaps = torch.arange(batch, device="cuda", dtype=torch.float32).repeat(4, 1)
        for width in (29, 55, 63, 64, 72, 76, 82):
            with self.subTest(width=width):
                head, tail = divmod(width - 2 * batch, 4)
                fallback = [5] * head + [1 + tail] + [1] * (batch - head - 1)
                caps, rows = pack_ep_margin_rows(
                    gaps,
                    prior,
                    torch.tensor(fallback, device="cuda", dtype=torch.int32),
                    width,
                )
                actual = caps.tolist()
                self.assertEqual(sum(actual) + batch, width)
                self.assertEqual(
                    rows.tolist(),
                    [
                        request * 6 + hop
                        for request, cap in enumerate(actual)
                        for hop in range(cap + 1)
                    ],
                )

    def test_four_gap_rerank_matches_independent_prefix_scores(self):
        batch = 14
        rng = torch.Generator().manual_seed(1212)
        gaps = torch.rand((4, batch), generator=rng) * 36
        prior = (
            (torch.rand((batch, 5), generator=rng) * 0.95 + 0.02)
            .sort(dim=1, descending=True)
            .values
        )
        ranked = []
        for req in range(batch):
            logged_gaps = [math.log1p(min(32, gap)) for gap in gaps[:, req].tolist()]
            prefix_score = math.inf
            for depth, (a, b, *c) in enumerate(_FOUR_GAP_SURVIVAL_COEFFICIENTS):
                p = prior[req, depth].item()
                prefix_score = min(
                    prefix_score,
                    a
                    + b * math.log(p / (1 - p))
                    + sum(x * w for x, w in zip(logged_gaps, c)),
                )
                if depth:
                    ranked.append((-prefix_score, req, depth))
        ranked.sort()
        device_gaps, device_prior = gaps.cuda(), prior.cuda()
        for width in (48, 56, 64, 72):
            with self.subTest(width=width):
                optional = width - 2 * batch
                expected = [1] * batch
                for _, req, _ in ranked[:optional]:
                    expected[req] += 1
                head, tail = divmod(optional, 4)
                fallback = [5] * head + [1 + tail] + [1] * (batch - head - 1)
                device_fallback = torch.tensor(
                    fallback, device="cuda", dtype=torch.int32
                )
                caps, rows = pack_ep_margin_rows(
                    device_gaps, device_prior, device_fallback, width, four_gaps=True
                )
                self.assertEqual(caps.tolist(), expected)
                self.assertEqual(
                    rows.tolist(),
                    [
                        req * 6 + hop
                        for req, cap in enumerate(expected)
                        for hop in range(cap + 1)
                    ],
                )
                bad_prior = device_prior.clone()
                bad_prior[-1, -1] = float("nan")
                result, retained = pack_ep_margin_rows(
                    device_gaps, bad_prior, device_fallback, width, four_gaps=True
                )
                self.assertTrue(torch.equal(result, device_fallback))
                self.assertEqual(
                    retained.tolist(),
                    [
                        req * 6 + hop
                        for req, cap in enumerate(fallback)
                        for hop in range(cap + 1)
                    ],
                )

    def test_equal_scores_keep_prefixes_and_are_deterministic(self):
        prior = torch.full((14, 5), 0.75, device="cuda")
        gaps = torch.full((4, 14), 5.0, device="cuda")
        fallback = torch.full((14,), 4, device="cuda", dtype=torch.int32)
        for use_four in (False, True):
            with self.subTest(use_four=use_four):
                first = pack_ep_margin_rows(
                    gaps, prior, fallback, 70, four_gaps=use_four
                )
                second = pack_ep_margin_rows(
                    gaps, prior, fallback, 70, four_gaps=use_four
                )
                self.assertTrue(
                    all(torch.equal(left, right) for left, right in zip(first, second))
                )
                self.assertEqual(
                    first[1].tolist(),
                    [
                        i * 6 + h
                        for i, c in enumerate(first[0].tolist())
                        for h in range(c + 1)
                    ],
                )


class VerificationPackingTest(unittest.TestCase):
    def test_strided_and_broadcast_capacities(self) -> None:
        for dtype in (torch.int32, torch.int64):
            backing = torch.tensor([3, 99, 0, 99, 2, 99], dtype=dtype, device="cuda")
            self.assertEqual(
                pack_verification_rows(backing[::2], 3, 8).tolist(),
                [0, 1, 2, 3, 4, 8, 9, 10],
            )
            caps = torch.tensor([1], dtype=dtype, device="cuda").expand(3)
            self.assertEqual(
                pack_verification_rows(caps, 3, 6).tolist(), [0, 1, 4, 5, 8, 9]
            )

    def test_invalid_packing_metadata_fails_before_launch(self) -> None:
        caps = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        for bad_caps, steps, total in [
            (caps.reshape(1, 2), 3, 5),
            (caps.float(), 3, 5),
            (caps, -1, 5),
            (caps, True, 5),
            (caps, 3, 1),
            (caps, 3, 9),
            (caps, 3, 5.5),
        ]:
            with self.assertRaises(AdaptiveBudgetInputError):
                pack_verification_rows(bad_caps, steps, total)

    def test_all_small_prefix_combinations(self) -> None:
        for steps in (1, 3, 5):
            for values in itertools.product(range(steps + 1), repeat=3):
                with self.subTest(steps=steps, caps=values):
                    caps = torch.tensor(values, dtype=torch.int32, device="cuda")
                    expected = [
                        i * (steps + 1) + j
                        for i, cap in enumerate(values)
                        for j in range(cap + 1)
                    ]
                    actual = pack_verification_rows(caps, steps, len(expected))
                    self.assertEqual(actual.tolist(), expected)

    def test_large_odd_batches_and_empty(self) -> None:
        for batch in (0, 1, 21, 255, 256, 257, 1025, 4097):
            for steps in (1, 5, 8):
                for mode in ("zero", "full", "mixed"):
                    with self.subTest(batch=batch, steps=steps, mode=mode):
                        values = [
                            (
                                0
                                if mode == "zero"
                                else steps if mode == "full" else (i * 7) % (steps + 1)
                            )
                            for i in range(batch)
                        ]
                        caps = torch.tensor(values, dtype=torch.int32, device="cuda")
                        expected = [
                            i * (steps + 1) + j
                            for i, cap in enumerate(values)
                            for j in range(cap + 1)
                        ]
                        self.assertEqual(
                            pack_verification_rows(caps, steps, len(expected)).tolist(),
                            expected,
                        )

    def test_graph_replay_uses_new_capacities_with_same_budget(self) -> None:
        caps = torch.tensor([0, 5, 2], dtype=torch.int32, device="cuda")
        for _ in range(3):
            pack_verification_rows(caps, 5, 10)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = pack_verification_rows(caps, 5, 10)
        caps.copy_(torch.tensor([4, 1, 2], dtype=torch.int32, device="cuda"))
        graph.replay()
        self.assertEqual(result.tolist(), [0, 1, 2, 3, 4, 6, 7, 12, 13, 14])


class DraftAllocationTest(unittest.TestCase):
    def test_fused_unknown_mask_and_nonfinite_with_remapped_slots(self) -> None:
        for batch in (16, 24, 30, 257):
            with self.subTest(batch=batch):
                torch.manual_seed(batch)
                state = torch.rand((batch * 2, 5), device="cuda")
                mapping = torch.arange(batch - 1, -1, -1, device="cuda") * 2
                known = torch.arange(batch, device="cuda") % 3 != 0
                state[mapping[1], 2] = float("nan")
                state[mapping[2], 1] = float("inf")
                state[mapping[3], 0] = float("-inf")
                state[mapping[4], 3] = -0.2
                state[mapping[5], 4] = 1.2
                initial = torch.full((batch,), 5, dtype=torch.int32, device="cuda")
                scratch = torch.empty(batch * 5, dtype=torch.int32, device="cuda")
                for mask in (None, known):
                    probabilities = state[mapping].cpu()
                    valid = (probabilities >= 0) & (probabilities <= 1)
                    probabilities = torch.where(valid, probabilities, 1.0)
                    if mask is not None:
                        probabilities[~mask.cpu()] = 1.0
                    for budget in (1, batch * 2, batch * 5 - 1):
                        caps = initial.clone()
                        assign_draft_token_budget(
                            state, mapping, caps, budget, scratch, known=mask
                        )
                        expected = reference_allocation(probabilities, initial, budget)
                        self.assertTrue(torch.equal(caps.cpu(), expected))
                for invalid in (known.to(torch.int32), known[:-1], known.cpu()):
                    with self.assertRaises(ValueError):
                        assign_draft_token_budget(
                            state, mapping, initial, 1, scratch, known=invalid
                        )

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
