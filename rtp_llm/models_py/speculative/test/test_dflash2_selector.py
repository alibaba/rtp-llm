# SPDX-License-Identifier: Apache-2.0
"""CPU probability contracts and CUDA/HIP selector numerical/graph tests.

Set GPU_COUNT=1 for GPU targets: missing GPUs then fail instead of silently
skipping. DFLASH2_SELECTOR_BENCHMARK=1 adds a bounded real-shape microbenchmark.
"""

import json
import math
import os
import time
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.speculative.dflash2_selector import (
    DFlash2CandidateSelector,
    dense_probabilities,
    selector_reference,
)


def _fixture(device="cpu", dtype=torch.float32, batch=4, slots=7, k=5):
    generator = torch.Generator(device=device).manual_seed(781)
    hidden_size, rank, vocab = 17, 9, 37

    def randn(*shape):
        return (
            torch.randn(*shape, generator=generator, device=device, dtype=dtype) * 0.2
        )

    selector = DFlash2CandidateSelector(
        randn(rank, hidden_size), randn(vocab, rank), randn(vocab, rank), k
    )
    inputs = (
        randn(batch, slots, hidden_size),
        randn(batch, slots, vocab),
        torch.arange(batch, device=device, dtype=torch.int64),
        torch.linspace(0.3, 1.2, batch, device=device),
        torch.arange(batch, device=device) % 2 == 0,
        torch.rand(batch, slots, generator=generator, device=device),
    )
    return selector, inputs


class DFlash2SelectorContractTest(unittest.TestCase):
    def test_exact_conditional_probabilities(self):
        # H=R=1 makes every edge independently calculable without calling the
        # implementation's scorer. The anchor materially changes the first q.
        projection = torch.ones(1, 1)
        predecessor = torch.tensor([[0.0], [1.0], [2.0], [-1.0]])
        successor = torch.tensor([[0.0], [0.4], [-0.2], [0.8]])
        selector = DFlash2CandidateSelector(projection, predecessor, successor, 3)
        hidden = torch.tensor([[[1.0], [0.5], [-1.0]]])
        logits = torch.tensor(
            [[[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 2.0, 0.0], [2.0, 0.0, 3.0, 1.0]]]
        )
        uniforms = torch.tensor([[0.2, 0.9, 0.5]])
        tokens, ids, q = selector(
            hidden,
            logits,
            torch.tensor([2]),
            torch.tensor([0.7]),
            torch.tensor([False]),
            uniforms,
        )
        previous = 2
        for slot in range(3):
            candidates = sorted(
                range(4), key=lambda t: float(logits[0, slot, t]), reverse=True
            )[:3]
            scores = [
                float(logits[0, slot, token])
                + float(predecessor[previous, 0])
                * float(hidden[0, slot, 0])
                * float(successor[token, 0])
                for token in candidates
            ]
            maximum = max(scores)
            weights = [math.exp((s - maximum) / 0.7) for s in scores]
            expected = [w / sum(weights) for w in weights]
            self.assertEqual(ids[0, slot].tolist(), candidates)
            torch.testing.assert_close(
                q[0, slot], torch.tensor(expected), atol=1e-6, rtol=1e-6
            )
            cumulative, index = 0.0, len(candidates) - 1
            for i, probability in enumerate(expected):
                cumulative += probability
                if float(uniforms[0, slot]) < cumulative:
                    index = i
                    break
            previous = candidates[index]
            self.assertEqual(int(tokens[0, slot]), previous)

    def test_mixed_greedy_point_mass_and_dense_reuse(self):
        selector, inputs = _fixture()
        tokens, ids, q = selector(*inputs)
        dense = torch.full((*q.shape[:2], selector.vocab_size + 7), float("nan"))
        result = dense_probabilities(ids, q, dense.shape[-1], dense)
        self.assertIs(result, dense)
        torch.testing.assert_close(dense.sum(-1), torch.ones_like(dense[..., 0]))
        self.assertTrue(torch.isfinite(dense).all())
        self.assertEqual(int(torch.count_nonzero(dense[..., selector.vocab_size :])), 0)
        for row in (0, 2):
            torch.testing.assert_close(
                dense[row].max(-1).values, torch.ones(q.shape[1])
            )
            self.assertTrue(torch.equal(dense[row].argmax(-1), tokens[row].long()))
        old = dense.clone()
        next_ids = (ids + 11) % selector.vocab_size
        dense_probabilities(next_ids, q, dense.shape[-1], dense)
        expected = torch.zeros_like(old).scatter_(-1, next_ids, q)
        torch.testing.assert_close(dense, expected)

    def test_rejection_preserves_target_distribution(self):
        # Enumerate the acceptance/residual identity for several actual
        # conditional selector distributions, including q=0 outside top-k.
        selector, inputs = _fixture(batch=3)
        _, ids, q = selector(*inputs)
        proposal = dense_probabilities(ids, q, selector.vocab_size).double()
        target = torch.arange(1, selector.vocab_size + 1, dtype=torch.float64)
        target /= target.sum()
        accepted = torch.minimum(proposal, target)
        residual = (target - proposal).clamp_min(0)
        rejection_probability = 1 - accepted.sum(-1, keepdim=True)
        recovered = accepted + rejection_probability * residual / residual.sum(
            -1, keepdim=True
        )
        torch.testing.assert_close(
            recovered, target.expand_as(recovered), atol=1e-7, rtol=1e-7
        )

    def test_padded_vocabulary_does_not_enter_candidates(self):
        selector, inputs = _fixture(batch=1)
        values = list(inputs)
        values[1] = torch.cat(
            [values[1], torch.full((*values[1].shape[:2], 9), float("inf"))], -1
        )
        _, ids, q = selector(*values)
        self.assertTrue((ids < selector.vocab_size).all())
        self.assertTrue(torch.isfinite(q).all())

    def test_extreme_temperature_and_empty_shapes(self):
        selector, inputs = _fixture(batch=4)
        values = list(inputs)
        values[3] = torch.tensor([0.0, 1e-30, float("nan"), 1e30])
        _, _, q = selector(*values)
        self.assertTrue(torch.isfinite(q).all())
        torch.testing.assert_close(q.sum(-1), torch.ones(q.shape[:2]))
        for batch, slots in ((0, 7), (2, 0)):
            selector, values = _fixture(batch=batch, slots=slots)
            tokens, ids, q = selector(*values)
            self.assertEqual(tokens.shape, (batch, slots))
            self.assertEqual(q.shape, (batch, slots, selector.top_k))

    def test_graph_configuration_is_lazy_and_validated(self):
        selector, inputs = _fixture()
        selector.configure_graph(True, max_cached_shapes=2)
        self.assertTrue(selector.graph_stats["enabled"])
        self.assertEqual(selector.graph_stats["cached_shapes"], 0)
        # CPU contract tests keep their oracle path without importing Triton.
        selector(*inputs)
        self.assertEqual(selector.graph_stats["captures"], 0)
        selector.configure_graph(False)
        self.assertFalse(selector.graph_stats["enabled"])
        with self.assertRaisesRegex(ValueError, "at least one shape"):
            selector.configure_graph(True, max_cached_shapes=0)

    def test_invalid_shape_and_weight_contracts(self):
        selector, inputs = _fixture()
        values = list(inputs)
        values[2] = values[2].float()
        with self.assertRaisesRegex(ValueError, "integer token"):
            selector(*values)
        values = list(inputs)
        values[1] = values[1][..., :-1]
        with self.assertRaisesRegex(ValueError, "complete valid vocabulary"):
            selector(*values)
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            DFlash2CandidateSelector(
                torch.ones(1, 1), torch.ones(2, 1), torch.ones(2, 1), 3
            )


class DFlash2SelectorGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            if int(os.environ.get("GPU_COUNT", "0")) > 0:
                raise RuntimeError("GPU target requires a real CUDA or HIP device")
            raise unittest.SkipTest("GPU tests require a CUDA or HIP device")

    def _check_reference(self, selector, inputs, atol=1e-5):
        actual = selector(*inputs)
        reference = selector_reference(
            selector.hidden_projection,
            selector.predecessor_codebook,
            selector.successor_codebook,
            *inputs,
            selector.top_k
        )
        self.assertTrue(torch.equal(actual[0], reference[0]))
        self.assertTrue(torch.equal(actual[1], reference[1]))
        torch.testing.assert_close(actual[2], reference[2], atol=atol, rtol=atol)
        torch.testing.assert_close(
            actual[2].sum(-1),
            torch.ones(actual[2].shape[:2], device="cuda"),
            atol=1e-6,
            rtol=1e-6,
        )
        return actual

    def test_fp32_bf16_fp16_non_power_of_two(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for batch, slots, k in ((1, 1, 1), (4, 7, 5), (8, 3, 16)):
                with self.subTest(dtype=dtype, batch=batch, slots=slots, k=k):
                    selector, inputs = _fixture("cuda", dtype, batch, slots, k)
                    self._check_reference(selector, inputs)

    def test_extreme_temperatures_and_masked_padding(self):
        selector, inputs = _fixture("cuda", batch=4)
        inputs[1][0].fill_(float("-inf"))
        inputs[1][1, :, :2] = float("inf")
        inputs[3].copy_(torch.tensor([0.0, 0.8, 0.0, 1e-30], device="cuda"))
        result = self._check_reference(selector, inputs)
        self.assertTrue(torch.isfinite(result[2]).all())

    def test_real_rank_hidden_and_anchor(self):
        # Real rank/hidden dimensions, small vocabulary to keep the correctness
        # test lightweight. The opt-in benchmark below uses the full V=248320.
        torch.manual_seed(134)
        b, slots, h, r, v = 4, 7, 5120, 256, 109

        def randn(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.03

        selector = DFlash2CandidateSelector(randn(r, h), randn(v, r), randn(v, r), 16)
        inputs = (
            randn(b, slots, h),
            randn(b, slots, v),
            torch.tensor([0, 37, 57, v - 1], device="cuda", dtype=torch.int32),
            torch.tensor([0.0, 0.5, 1.1, 5.0], device="cuda"),
            torch.tensor([True, False, False, False], device="cuda"),
            torch.rand(b, slots, device="cuda"),
        )
        self._check_reference(selector, inputs)

    def test_graph_replay_refreshes_anchor_uniforms_and_q(self):
        selector, inputs = _fixture("cuda", batch=4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                selector(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        dense = torch.empty((*inputs[0].shape[:2], selector.vocab_size), device="cuda")
        with torch.cuda.graph(graph):
            tokens, ids, q = selector(*inputs)
            dense_probabilities(ids, q, selector.vocab_size, dense)
        for update in range(3):
            inputs[2].add_(1).remainder_(selector.vocab_size)
            inputs[5].uniform_()
            inputs[4].logical_not_()
            graph.replay()
            expected = self._check_reference(selector, inputs)
            self.assertTrue(torch.equal(tokens, expected[0]))
            torch.testing.assert_close(q, expected[2], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                dense,
                dense_probabilities(expected[1], expected[2], selector.vocab_size),
            )

    def test_enabled_runner_replays_updates_and_preserves_prior_outputs(self):
        selector, inputs = _fixture("cuda", dtype=torch.bfloat16, batch=4)
        selector.configure_graph(True)
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            # First capture must see work queued on the calling stream.
            inputs[0].add_(0.1)
            with self.assertLogs(
                "rtp_llm.models_py.speculative.dflash2_selector", level="INFO"
            ) as logs:
                first = selector(*inputs)
            saved = tuple(t.clone() for t in first)
        torch.cuda.current_stream().wait_stream(producer)
        self.assertTrue(any("graph captured" in message for message in logs.output))
        self.assertTrue(any("first replay" in message for message in logs.output))
        expected = selector_reference(
            selector.hidden_projection,
            selector.predecessor_codebook,
            selector.successor_codebook,
            *inputs,
            selector.top_k
        )
        for actual, reference in zip(first, expected):
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        self.assertEqual(selector.graph_stats["captures"], 1)
        self.assertEqual(selector.graph_stats["replays"], 1)
        for _ in range(3):
            inputs[0].mul_(0.8)
            inputs[1].add_(torch.randn_like(inputs[1]) * 0.01)
            inputs[2].add_(1).remainder_(selector.vocab_size)
            inputs[3].add_(0.1)
            inputs[4].logical_not_()
            inputs[5].uniform_()
            current = selector(*inputs)
            expected = selector_reference(
                selector.hidden_projection,
                selector.predecessor_codebook,
                selector.successor_codebook,
                *inputs,
                selector.top_k
            )
            for actual, reference in zip(current, expected):
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
            for original, previous, actual in zip(first, saved, current):
                self.assertNotEqual(original.data_ptr(), actual.data_ptr())
                self.assertTrue(torch.equal(original, previous))
        self.assertEqual(selector.graph_stats["captures"], 1)
        self.assertEqual(selector.graph_stats["replays"], 4)

    def test_enabled_runner_bounds_shape_cache_and_reports_eager(self):
        selector, inputs = _fixture("cuda", batch=1)
        selector.configure_graph(True, max_cached_shapes=1)
        selector(*inputs)
        _, larger_inputs = _fixture("cuda", batch=2)
        with self.assertLogs(
            "rtp_llm.models_py.speculative.dflash2_selector", level="WARNING"
        ) as logs:
            self._check_reference(selector, larger_inputs)
        self.assertTrue(any("cache limit=1" in message for message in logs.output))
        stats = selector.graph_stats
        self.assertEqual(stats["cached_shapes"], 1)
        self.assertEqual(stats["captures"], 1)
        self.assertEqual(stats["replays"], 1)
        self.assertEqual(stats["eager_cache_limit"], 1)
        selector.configure_graph(False)
        self._check_reference(selector, inputs)
        self.assertEqual(selector.graph_stats["replays"], 1)

    def test_enabled_runner_capture_failure_is_reported_and_reuses_eager(self):
        selector, inputs = _fixture("cuda", batch=1)
        selector.configure_graph(True)
        with patch.object(
            torch.cuda, "CUDAGraph", side_effect=RuntimeError("capture unavailable")
        ) as capture:
            with self.assertLogs(
                "rtp_llm.models_py.speculative.dflash2_selector", level="WARNING"
            ) as logs:
                first = selector(*inputs)
            self.assertTrue(
                any("falling back to eager" in message for message in logs.output)
            )
            inputs[2].add_(1)
            inputs[5].uniform_()
            second = selector(*inputs)
            self.assertEqual(capture.call_count, 1)
        expected = selector_reference(
            selector.hidden_projection,
            selector.predecessor_codebook,
            selector.successor_codebook,
            *inputs,
            selector.top_k
        )
        for actual, reference in zip(second, expected):
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        for before, after in zip(first, second):
            self.assertNotEqual(before.data_ptr(), after.data_ptr())
        self.assertEqual(selector.graph_stats["capture_failures"], 1)
        self.assertEqual(selector.graph_stats["disabled_shapes"], 1)
        self.assertEqual(selector.graph_stats["eager_capture_failure"], 2)
        self.assertEqual(selector.graph_stats["captures"], 0)
        self.assertEqual(selector.graph_stats["replays"], 0)

    def test_runtime_capture_error_restores_stream_and_falls_back(self):
        selector, inputs = _fixture("cuda", batch=1)
        selector.configure_graph(True)
        eager = selector._forward_gpu
        capture_attempts = []
        original_stream = torch.cuda.current_stream()

        def unsupported_during_capture(*args):
            if torch.cuda.is_current_stream_capturing():
                capture_attempts.append(True)
                # A real backend error (device synchronization is forbidden
                # during stream capture), not a mocked CUDAGraph constructor.
                torch.cuda.synchronize()
            return eager(*args)

        with patch.object(
            selector, "_forward_gpu", side_effect=unsupported_during_capture
        ):
            with self.assertLogs(
                "rtp_llm.models_py.speculative.dflash2_selector", level="WARNING"
            ):
                actual = selector(*inputs)
            self.assertEqual(torch.cuda.current_stream(), original_stream)
            selector(*inputs)
        self.assertEqual(len(capture_attempts), 1)
        expected = selector_reference(
            selector.hidden_projection,
            selector.predecessor_codebook,
            selector.successor_codebook,
            *inputs,
            selector.top_k
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
        self.assertEqual(selector.graph_stats["capture_failures"], 1)
        self.assertEqual(selector.graph_stats["eager_capture_failure"], 2)
        self.assertEqual(selector.graph_stats["captures"], 0)

    @unittest.skipUnless(
        os.environ.get("DFLASH2_SELECTOR_BENCHMARK") == "1",
        "opt-in full-vocabulary benchmark",
    )
    def test_real_shape_benchmark(self):
        torch.manual_seed(918)
        h, r, v, slots, k = 5120, 256, 248320, 7, 16

        def randn(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.03

        selector = DFlash2CandidateSelector(randn(r, h), randn(v, r), randn(v, r), k)
        for batch in (1, 8, 32):
            inputs = (
                randn(batch, slots, h),
                randn(batch, slots, v),
                torch.zeros(batch, device="cuda", dtype=torch.int32),
                torch.ones(batch, device="cuda"),
                torch.zeros(batch, device="cuda", dtype=torch.bool),
                torch.rand(batch, slots, device="cuda"),
            )
            for _ in range(10):
                selector(*inputs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(50):
                selector(*inputs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000 / 50
            print(
                json.dumps(
                    {
                        "benchmark": "dflash2_selector_full_topk",
                        "device": torch.cuda.get_device_name(),
                        "hip": torch.version.hip,
                        "batch": batch,
                        "slots": slots,
                        "K": k,
                        "rank": r,
                        "vocab": v,
                        "eager_ms": elapsed,
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
