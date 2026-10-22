import unittest

import torch

from rtp_llm.eplb.policy import _batch_greedy_indices, rebalance_experts_global


class PolicyTest(unittest.TestCase):
    def assert_layout(self, physical, inverse, counts):
        self.assertTrue(torch.equal(counts, (inverse >= 0).sum(-1)))
        self.assertTrue(bool((counts > 0).all()))
        for layer in range(physical.size(0)):
            self.assertEqual(int(counts[layer].sum()), physical.size(1))
            self.assertEqual(
                sorted(inverse[layer][inverse[layer] >= 0].tolist()),
                list(range(physical.size(1))),
            )
            for expert in range(counts.size(1)):
                slots = inverse[layer, expert, : counts[layer, expert]].long()
                self.assertTrue(bool((physical[layer, slots] == expert).all()))

    def test_structural_invariants_and_determinism(self):
        for steps, layers, logical, physical, local in (
            (1, 1, 8, 8, 1),
            (3, 2, 8, 12, 3),
            (2, 3, 4, 20, 5),
            (3, 1, 1, 8, 4),
            (4, 2, 12, 12, 4),
        ):
            with self.subTest(shape=(steps, layers, logical, physical, local)):
                generator = torch.Generator().manual_seed(steps + physical)
                history = torch.randint(
                    0, 200, (steps, layers, logical), generator=generator
                )
                saved = history.clone()
                result = rebalance_experts_global(history, physical, local)
                self.assert_layout(*result)
                for first, second in zip(
                    result, rebalance_experts_global(history, physical, local)
                ):
                    self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(history, saved))

    def test_hot_expert_receives_multiple_replicas(self):
        history = torch.tensor([[[1000, 1, 0, 0]], [[900, 0, 1, 0]]])
        physical, inverse, counts = rebalance_experts_global(history, 6, 3)
        self.assertEqual(counts.tolist(), [[3, 1, 1, 1]])
        self.assert_layout(physical, inverse, counts)

    def test_cpu_policy_under_meta_device_context(self):
        history = torch.tensor([[[1000, 1, 0, 0]], [[900, 0, 1, 0]]])
        saved = history.clone()
        for local in (1, 3):
            with self.subTest(local=local):
                expected = rebalance_experts_global(history, 6, local)
                with torch.device("meta"):
                    result = rebalance_experts_global(history, 6, local)
                for actual, baseline in zip(result, expected):
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertFalse(actual.is_meta)
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertTrue(torch.equal(actual, baseline))
                self.assertEqual(result[2].tolist(), [[3, 1, 1, 1]])
                self.assert_layout(*result)
                self.assertTrue(torch.equal(history, saved))

    def test_float_history_not_modified(self):
        history = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8)
        original = history.clone()
        rebalance_experts_global(history, 12, 3)
        self.assertTrue(torch.equal(history, original))

    def test_equal_hot_experts_do_not_allocate_replicas_to_idle_expert(self):
        history = torch.tensor([[[0, 2**33, 2**33]]], dtype=torch.int64)
        self.assertEqual(
            rebalance_experts_global(history, 5, 1)[2].tolist(), [[1, 2, 2]]
        )

    def test_ties_and_cross_step_complementarity(self):
        cases = (
            ([[0, 0, 0, 0, 0, 0]], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
            ([[0, 0, 0]], [2, 0, 1, 0, 2, 1], [0, 1, 2, 3, 4, 5]),
            (
                [[10, 0, 0, 0, 0, 0], [0, 6, 4, 0, 0, 0]],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 3, 2, 4, 5],
            ),
            ([[1, 1, 1, 1, 1, 1]], [0, 1, 2, 3, 4, 5], [0, 2, 4, 1, 3, 5]),
            (
                [[10, 0, 6, 0, 4, 0], [0, 10, 0, 6, 0, 4]],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 5, 2, 3, 4],
            ),
        )
        for history, physical, expected in cases:
            with self.subTest(history=history):
                history = torch.tensor(history).unsqueeze(1)
                physical = torch.tensor([physical])
                counts = torch.bincount(
                    physical.flatten(), minlength=history.size(-1)
                ).unsqueeze(0)
                actual = _batch_greedy_indices(history, physical, counts, 3)
                self.assertEqual(actual.tolist(), [expected])

    def test_step_history_is_not_collapsed(self):
        history = torch.tensor([[[10, 0, 6, 0, 4, 0]], [[0, 10, 0, 6, 0, 4]]])
        full = rebalance_experts_global(history, 6, 3)[0]
        collapsed = rebalance_experts_global(history.sum(0, keepdim=True), 6, 3)[0]
        self.assertFalse(torch.equal(full, collapsed))

        def peak_cost(layout):
            return (
                history[..., layout[0].long()]
                .reshape(2, 1, 2, 3)
                .sum(-1)
                .amax(-1)
                .sum()
            )

        self.assertLess(int(peak_cost(full)), int(peak_cost(collapsed)))

    def test_int64_history_larger_than_int32(self):
        history = torch.tensor(
            [[[2**33, 1, 0, 0]], [[2**33, 0, 1, 0]]], dtype=torch.int64
        )
        self.assertEqual(
            rebalance_experts_global(history, 6, 3)[2].tolist(), [[3, 1, 1, 1]]
        )

    def test_invalid_policy_inputs_rejected(self):
        for history, physical, local in (
            (torch.zeros(1, 4), 4, 2),
            (torch.zeros(0, 1, 4), 4, 2),
            (-torch.ones(1, 1, 4), 4, 2),
            (torch.zeros(1, 1, 4), 3, 1),
            (torch.zeros(1, 1, 4), 5, 2),
            (torch.zeros(1, 1, 4), 4, 0),
        ):
            with self.subTest(shape=history.shape, physical=physical, local=local):
                with self.assertRaises(ValueError):
                    rebalance_experts_global(history, physical, local)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
