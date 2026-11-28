import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.eplb.policy import _batch_greedy_indices, rebalance_experts_global


class PolicyTest(unittest.TestCase):
    # The GPU target explicitly selects CUDA. CPU tests do not initialize CUDA.
    device = torch.device(os.environ.get("EPLB_POLICY_TEST_DEVICE", "cpu"))

    def tensor(self, values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=self.device)

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

    def test_fixed_deepinfer_reference_vectors(self):
        # Generated solely by the pinned DeepInfer source, never this policy.
        golden = json.loads(Path(__file__).with_name("policy_golden.json").read_text())
        self.assertEqual(
            golden["reference_commit"], "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
        )
        self.assertEqual(
            golden["reference_source_sha256"],
            "52241603827a1a1d19a395d48f5d618a7336fd1ceee631127e6782c644f736dc",
        )
        for case in golden["cases"]:
            with self.subTest(case=case["name"]):
                history = self.tensor(case["history"])
                saved = history.clone()
                result = rebalance_experts_global(
                    history, case["physical_experts"], case["local_experts"]
                )
                repeated = rebalance_experts_global(
                    history, case["physical_experts"], case["local_experts"]
                )
                for actual, again, key in zip(
                    result,
                    repeated,
                    ("physical_map", "inverse_map", "replica_counts"),
                ):
                    self.assertEqual(actual.device, history.device)
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertTrue(torch.equal(actual, self.tensor(case[key])), key)
                    self.assertTrue(torch.equal(actual, again), key)
                self.assert_layout(*result)
                self.assertTrue(torch.equal(history, saved))

    def test_reference_replica_order_is_preserved(self):
        history = self.tensor([[[4, 39, 13, 0]], [[3, 19, 27, 23]]])
        physical, inverse, counts = rebalance_experts_global(history, 8, 2)
        self.assertEqual(physical.tolist(), [[2, 1, 2, 0, 1, 3, 1, 3]])
        self.assertEqual(counts.tolist(), [[1, 3, 2, 2]])
        # Sorting physical slots would change the GPU chosen for replica zero.
        self.assertEqual(inverse[0, 1, :3].tolist(), [4, 6, 1])

    def test_reference_epsilon_rounding_limitation_is_preserved(self):
        # FP32 rounds away 1e-4/2e-4 at this load. DeepInfer gives both spare
        # slots to idle expert 0; improving this tie would change the algorithm.
        history = self.tensor([[[0, 10000, 10000]]])
        physical, inverse, counts = rebalance_experts_global(history, 5, 1)
        self.assertEqual(counts.tolist(), [[3, 1, 1]])
        self.assertEqual(physical.tolist(), [[0, 1, 2, 0, 0]])
        self.assertEqual(inverse[0, 0].tolist(), [0, 3, 4])

    def test_allocations_follow_input_device(self):
        history = self.tensor([[[1000, 1, 0, 0]], [[900, 0, 1, 0]]])
        for local in (1, 3):
            with self.subTest(local=local):
                expected = rebalance_experts_global(history, 6, local)
                with torch.device("meta"):
                    result = rebalance_experts_global(history, 6, local)
                for actual, baseline in zip(result, expected):
                    self.assertEqual(actual.device, history.device)
                    self.assertFalse(actual.is_meta)
                    self.assertTrue(torch.equal(actual, baseline))

    def test_float_and_noncontiguous_history_not_modified(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 1, 8), (2, 3, 8)):
                with self.subTest(dtype=dtype, shape=shape):
                    history = torch.arange(
                        2 * shape[0] * shape[1] * shape[2],
                        dtype=dtype,
                        device=self.device,
                    ).reshape(*shape, 2)[..., 0]
                    original = history.clone()
                    result = rebalance_experts_global(history, 12, 3)
                    contiguous_result = rebalance_experts_global(
                        history.contiguous(), 12, 3
                    )
                    for actual, expected in zip(result, contiguous_result):
                        self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(history, original))

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
                history = self.tensor(history).unsqueeze(1)
                physical = self.tensor([physical])
                counts = torch.bincount(
                    physical.flatten().long(), minlength=history.size(-1)
                ).unsqueeze(0)
                actual = _batch_greedy_indices(history, physical, counts, 3)
                self.assertEqual(actual.tolist(), [expected])

    def test_step_history_is_not_collapsed(self):
        history = self.tensor([[[10, 0, 6, 0, 4, 0]], [[0, 10, 0, 6, 0, 4]]])
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

    def test_invalid_policy_shapes_rejected(self):
        for shape, physical, local in (
            ((1, 4), 4, 2),
            ((0, 1, 4), 4, 2),
            ((1, 0, 4), 4, 2),
            ((1, 1, 0), 4, 2),
            ((1, 1, 4), 3, 1),
            ((1, 1, 4), 5, 2),
            ((1, 1, 4), 4, 0),
        ):
            with self.subTest(shape=shape, physical=physical, local=local):
                with self.assertRaises(ValueError):
                    rebalance_experts_global(
                        torch.zeros(shape, dtype=torch.int32, device=self.device),
                        physical,
                        local,
                    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
