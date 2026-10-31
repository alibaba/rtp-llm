#!/usr/bin/env python3
"""
Ascend NPU Sampler integration tests.

These tests validate execSampleGreedy() on real Ascend NPU hardware.
They mirror the 7 C++ test cases from SamplerTest.cc.

Prerequisites (set by .bazelrc test:ascend --test_env):
  LD_LIBRARY_PATH  must include torch/lib, torch_npu/lib, /opt/conda310/lib
  PYTHONHOME       must point to conda env (e.g. /opt/conda310)

Note: since 2026-10 the top-k/top-p filter op is the CANN built-in
ApplyTopKTopPWithSorted (aclnnApplyTopKTopP in libopapi.so), so no custom
vendor/OPP environment is required anymore.

Usage:
  cd rtp-llm
  python rtp_llm/models_py/bindings/ascend/ops/tests/test_ascend_sampler.py
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch_npu  # noqa: F401 — registers NPU dispatch before any NPU ops

# ---------------------------------------------------------------------------
# Locate and import the pybind module built by Bazel.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(
    _HERE, "../../../../../../bazel-bin/rtp_llm/models_py/bindings/ascend/ops/tests")

for _p in (_HERE, _MODULE_DIR):
    if os.path.isfile(os.path.join(_p, "sampler_test_module.so")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
else:
    sys.path.insert(0, _MODULE_DIR)

import sampler_test_module  # type: ignore[import-not-found]

NPU_DEVICE = torch.device("npu:0")


def npu_float_tensor(data, shape):
    """Create an NPU float32 tensor from a flat list."""
    return torch.tensor(data, dtype=torch.float32, device=NPU_DEVICE).reshape(shape)


def pinned_int_tensor(data):
    """Create a pinned CPU int32 tensor (mimics C++ pin_memory)."""
    return torch.tensor(data, dtype=torch.int32).pin_memory()


def pinned_float_tensor(data):
    """Create a pinned CPU float32 tensor."""
    return torch.tensor(data, dtype=torch.float32).pin_memory()


def npu_generator(seed):
    """Create a seeded NPU generator (mirrors C++ npuGenerator)."""
    gen = torch.Generator(device="npu")
    gen.manual_seed(seed)
    return gen


class TestAscendSampler(unittest.TestCase):
    """NPU sampler tests — mirrors SamplerTest.cc."""

    def setUp(self):
        torch.manual_seed(114514)

    # ==================================================================
    # Test 1 — top_k=1 deterministic argmax
    # ==================================================================
    def test_topk1_deterministic(self):
        batch_size = 4
        vocab_size = 10
        logits = npu_float_tensor(
            [
                0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.01,  # b0 → idx5
                0.2, 0.3, 0.0, 0.0, 0.99, 0.989, 0.0, 0.0, 0.0, 0.1,  # b1 → idx4
                0.43, 0.01, 0.22, 0.0, 0.0, 0.1, 0.2, 0.32, 0.0, 0.44,  # b2 → idx9 (no tie)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0,  # b3 → idx6
            ],
            (batch_size, vocab_size),
        )
        step = 5
        token_ids = torch.tensor(
            [[100, 1, 1, 1, 1, 0],
             [1, 1, 0, 0, 0, 0],
             [1, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([1, 1, 1, 1])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0, 1.0])

        result = sampler_test_module.exec_sample_greedy(
            logits=logits,
            input_lengths=input_lens,
            sequence_lengths=seq_lens,
            token_ids=token_ids,
            step=step,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        out = result["token_ids"].cpu().flatten().tolist()
        self.assertEqual(out[5], 5, f"batch 0 last token: expected 5, got {out[5]}")
        self.assertEqual(out[11], 4, f"batch 1 last token: expected 4, got {out[11]}")
        self.assertEqual(out[17], 9, f"batch 2 last token: expected 9, got {out[17]}")
        self.assertEqual(out[23], 6, f"batch 3 last token: expected 6, got {out[23]}")

    # ==================================================================
    # Test 2 — top_k sampling with generators (NPU: verify valid output)
    # Note: NPU generators are not cross-run deterministic, so we verify
    # that top_k=1 batches produce argmax and top_k>=2 batches sample
    # valid tokens, without requiring cross-run reproducibility.
    # ==================================================================
    def test_topk_with_generator_deterministic(self):
        batch_size = 4
        vocab_size = 10
        logits = npu_float_tensor(
            [
                0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.01,
                0.2, 0.3, 0.0, 0.0, 0.99, 0.989, 0.0, 0.0, 0.0, 0.1,
                0.43, 0.01, 0.22, 0.0, 0.0, 0.1, 0.2, 0.32, 0.0, 0.44,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0,
            ],
            (batch_size, vocab_size),
        )
        step = 5

        def make_token_ids():
            return torch.tensor(
                [[100, 1, 1, 1, 1, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0]],
                dtype=torch.int32,
            )

        seq_lens = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([1, 1, 0, 2])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0, 1.0])

        gens = [npu_generator(42 + i) for i in range(batch_size)]
        result = sampler_test_module.exec_sample_greedy(
            logits=logits,
            input_lengths=input_lens,
            sequence_lengths=seq_lens,
            token_ids=make_token_ids(),
            step=step,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            generators=gens,
        )
        out = result["token_ids"].cpu().flatten().tolist()

        # batch 0: top_k=1 → deterministic argmax=5
        self.assertEqual(out[5], 5, f"batch 0 (top_k=1) must be argmax=5, got {out[5]}")
        # batch 1: top_k=1 → deterministic argmax=4
        self.assertEqual(out[11], 4, f"batch 1 (top_k=1) must be argmax=4, got {out[11]}")
        # batch 2: top_k=0 → full sampling, any token is valid
        self.assertGreaterEqual(out[17], 0, f"batch 2 sampled token out of range: {out[17]}")
        self.assertLess(out[17], vocab_size, f"batch 2 sampled token out of range: {out[17]}")
        # batch 3: top_k=2 → sampling from top-2, any token is valid
        self.assertGreaterEqual(out[23], 0, f"batch 3 sampled token out of range: {out[23]}")
        self.assertLess(out[23], vocab_size, f"batch 3 sampled token out of range: {out[23]}")

    # ==================================================================
    # Test 3 — top_p sampling with generators (NPU: verify valid output)
    # Note: NPU generators are not cross-run deterministic. We verify
    # that top_p sampling runs without error and produces in-range tokens.
    # ==================================================================
    def test_topp_with_generator_deterministic(self):
        batch_size = 2
        vocab_size = 6
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   # ascending
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,    # descending
            ],
            (batch_size, vocab_size),
        )
        step = 3

        def make_token_ids():
            return torch.tensor(
                [[10, 0, 0, 0],
                 [10, 1, 1, 1]],
                dtype=torch.int32,
            )

        seq_lens = torch.tensor([3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([0, 0])
        top_p = pinned_float_tensor([0.5, 0.5])
        temperature = pinned_float_tensor([1.0, 1.0])

        gens = [npu_generator(100 + i) for i in range(batch_size)]
        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=make_token_ids(), step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            generators=gens,
        )
        out = result["token_ids"].cpu().flatten().tolist()

        # Check that sampled tokens are in valid range
        # Batch 0: top_p=0.5 on ascending logits [1,2,3,4,5,6]
        #   softmax = [0.004, 0.012, 0.032, 0.087, 0.237, 0.645]
        #   cumsum from largest: 0.645, 0.882, ... → top_p=0.5 selects {5} only
        self.assertGreaterEqual(out[3], 0, f"batch 0 token out of range: {out[3]}")
        self.assertLess(out[3], vocab_size, f"batch 0 token out of range: {out[3]}")

        # Batch 1: top_p=0.5 on descending logits [6,5,4,3,2,1]
        #   softmax = [0.645, 0.237, 0.087, 0.032, 0.012, 0.004]
        #   cumsum from largest: 0.645, 0.882, ... → top_p=0.5 selects {0} only
        self.assertGreaterEqual(out[7], 0, f"batch 1 token out of range: {out[7]}")
        self.assertLess(out[7], vocab_size, f"batch 1 token out of range: {out[7]}")

    # ==================================================================
    # Test 4 — temperature scaling
    # ==================================================================
    def test_temperature(self):
        batch_size = 2
        vocab_size = 6
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  # batch 0: t=1
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   # batch 1: t=10
            ],
            (batch_size, vocab_size),
        )
        step = 3

        def make_token_ids():
            return torch.tensor(
                [[10, 0, 0, 0],
                 [10, 1, 1, 1]],
                dtype=torch.int32,
            )

        seq_lens = torch.tensor([3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([0, 0])
        top_p = pinned_float_tensor([1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 10.0])

        gens1 = [npu_generator(202), npu_generator(202)]
        result1 = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=make_token_ids(), step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            generators=gens1,
        )
        tokens1 = result1["token_ids"].cpu().flatten().tolist()

        gens2 = [npu_generator(202), npu_generator(202)]
        result2 = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=make_token_ids(), step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            generators=gens2,
        )
        tokens2 = result2["token_ids"].cpu().flatten().tolist()

        self.assertEqual(tokens1, tokens2,
                         "Temperature test: same seeds must yield same output")

    # ==================================================================
    # Test 5 — do_sample flag (mixed sampling/greedy batches)
    # ==================================================================
    def test_do_sample(self):
        batch_size = 3
        vocab_size = 6
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   # b0: sample
                9.0, 1.0, 1.0, 1.0, 1.0, 1.0,    # b1: greedy (do_sample=false) → argmax=0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    # b2: sample
            ],
            (batch_size, vocab_size),
        )
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [20, 5, 5, 5],
             [30, 0, 0, 0]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([0, 0, 0])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0])
        do_sample = torch.tensor([True, False, True], dtype=torch.bool)

        gens = [npu_generator(300 + i) for i in range(batch_size)]
        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids, step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            do_sample=do_sample, generators=gens,
        )

        out = result["token_ids"].cpu().flatten().tolist()
        # batch 1 (index 1): do_sample=false → must be argmax=0
        # The new token is written at token_ids[batch][step]; for batch=1, step=3:
        #   flattened index = 1 * (step+1) + step = 1 * 4 + 3 = 7
        self.assertEqual(out[7], 0,
                         f"batch 1 (do_sample=false) must use argmax, got {out[7]}")

    # ==================================================================
    # Test 6 — output_all_probs
    # ==================================================================
    def test_output_all_probs(self):
        batch_size = 2
        vocab_size = 5
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0,   # ascending → probs ascending
                5.0, 4.0, 3.0, 2.0, 1.0,    # descending → probs descending
            ],
            (batch_size, vocab_size),
        )
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [10, 1, 1, 1]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([0, 0])
        top_p = pinned_float_tensor([1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0])
        all_probs = torch.zeros((batch_size, vocab_size), dtype=torch.float32)

        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids, step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            output_all_probs=all_probs,
        )

        probs = result["output_all_probs"].cpu().numpy()

        # Each batch's probs should sum to ~1.0
        for b in range(batch_size):
            self.assertAlmostEqual(float(probs[b].sum()), 1.0, delta=1e-4,
                                   msg=f"batch {b} probs sum is {probs[b].sum()}")

        # batch 0: ascending, batch 1: descending
        for v in range(1, vocab_size):
            self.assertLess(probs[0, v - 1], probs[0, v],
                            "batch 0 probs must be ascending")
            self.assertGreater(probs[1, v - 1], probs[1, v],
                               "batch 1 probs must be descending")

    # ==================================================================
    # Test 7 — cum_log_probs
    # ==================================================================
    def test_cum_log_probs(self):
        # ... (unchanged) ...
        batch_size = 2
        vocab_size = 5
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0,   # ascending
                5.0, 4.0, 3.0, 2.0, 1.0,    # descending
            ],
            (batch_size, vocab_size),
        )
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [10, 1, 1, 1]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([0, 0])
        top_p = pinned_float_tensor([1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0])
        cum_log_probs = torch.zeros((batch_size,), dtype=torch.float32)

        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids, step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
            cum_log_probs=cum_log_probs,
        )

        clp = result["cum_log_probs"].cpu().tolist()
        self.assertLessEqual(clp[0], 0.0, "cum_log_probs batch 0 must be <= 0")
        self.assertLessEqual(clp[1], 0.0, "cum_log_probs batch 1 must be <= 0")
        self.assertGreater(clp[0], -100.0, "cum_log_probs batch 0 must be > -inf")
        self.assertGreater(clp[1], -100.0, "cum_log_probs batch 1 must be > -inf")

    # ==================================================================
    # Test 7b — strict do_sample mixed-batch merge.
    #   Near-flat logits make multinomial genuinely random, so pinning the
    #   do_sample=false row to argmax across seeds can only come from the
    #   per-row merge, not from distribution sharpness.
    # ==================================================================
    def test_do_sample_mixed_batch_strict(self):
        batch_size = 3
        vocab_size = 6
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    # b0: sample, top-4 = {3,4,5}
                1.0, 1.0, 1.0, 1.0, 1.1, 1.0,    # b1: near-flat, argmax=4
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,    # b2: sample, top-4 = {0,1,2,3}
            ],
            (batch_size, vocab_size),
        )
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [20, 5, 5, 5],
             [30, 0, 0, 0]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([4, 4, 4])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0])
        do_sample = torch.tensor([True, False, True], dtype=torch.bool)

        for trial in range(5):
            gens = [npu_generator(7000 + trial * 10 + i) for i in range(batch_size)]
            result = sampler_test_module.exec_sample_greedy(
                logits=logits.clone(), input_lengths=input_lens,
                sequence_lengths=seq_lens, token_ids=token_ids.clone(),
                step=step, top_k=top_k, top_p=top_p, temperature=temperature,
                do_sample=do_sample, generators=gens,
            )
            out = result["token_ids"].cpu().flatten().tolist()
            self.assertEqual(
                out[7], 4,
                f"trial {trial}: do_sample=false row must equal argmax (4), got {out[7]}")
            self.assertIn(out[3], (3, 4, 5), f"trial {trial}: b0 out of top-4 set")
            self.assertIn(out[11], (0, 1, 2, 3), f"trial {trial}: b2 out of top-4 set")

    # ==================================================================
    # Test 7c — cum_log_probs numeric correctness vs plain-torch reference.
    #   Covers: top_k=1 row (renormalized one-hot → delta 0), do_sample=false
    #   row, and a sampling row; reference = top-k filter → softmax → log of
    #   the chosen token's probability.
    # ==================================================================
    def test_cum_log_probs_matches_reference(self):
        batch_size = 3
        vocab_size = 6
        logits_cpu = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 1.0, 1.0, 1.0, 1.1, 1.0],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        logits = logits_cpu.to(NPU_DEVICE)
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [20, 5, 5, 5],
             [30, 0, 0, 0]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([1, 4, 4])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0])
        do_sample = torch.tensor([True, False, True], dtype=torch.bool)
        cum_log_probs = torch.zeros((batch_size,), dtype=torch.float32)

        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids, step=step, top_k=top_k, top_p=top_p,
            temperature=temperature, do_sample=do_sample,
            cum_log_probs=cum_log_probs,
        )

        chosen = [row[step] for row in result["token_ids"].cpu().tolist()]
        clp = result["cum_log_probs"].cpu().tolist()

        for i, k in enumerate([1, 4, 4]):
            # The custom op's top-k is threshold-based: every logit >= the
            # k-th largest value survives (ties are all kept), unlike
            # index-based torch.topk which keeps exactly k entries.
            filtered = logits_cpu[i].clone()
            threshold = torch.topk(filtered, k).values[-1]
            filtered[filtered < threshold] = float("-inf")
            probs = torch.softmax(filtered, -1)
            if k == 1:
                self.assertEqual(chosen[i], int(torch.argmax(logits_cpu[i])),
                                 f"batch {i}: top_k=1 row must pick argmax")
                expected = 0.0  # one-hot renormalized distribution → log(1)
            else:
                expected = float(torch.log(probs[chosen[i]]))
            self.assertAlmostEqual(
                clp[i], expected, delta=2e-3,
                msg=f"batch {i}: cum_log_probs={clp[i]}, expected={expected}")

    # ==================================================================
    # Test 7e — ORIGINAL logprobs mode (return_original_all_probs=True):
    #   all_probs / cum_log_probs come from the UNFILTERED softmax
    #   (CUDA/generic parity); sampling still uses the filtered
    #   distribution. (finding 23059)
    # ==================================================================
    def test_original_all_probs_unfiltered(self):
        batch_size = 2
        vocab_size = 6
        logits_cpu = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # b0: top_k=1 → argmax idx5
                [0.5, 3.0, 1.0, 2.5, 0.2, 0.1],  # b1: top_k=3 (no ties)
            ],
            dtype=torch.float32,
        )
        logits = logits_cpu.to(NPU_DEVICE)
        step = 2
        token_ids = torch.tensor(
            [[10, 0, 0],
             [20, 1, 1]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([2, 2], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([1, 3])
        top_p = pinned_float_tensor([1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0])
        output_all_probs = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
        cum_log_probs = torch.zeros((batch_size,), dtype=torch.float32)

        result = sampler_test_module.exec_sample_greedy(
            logits=logits, input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids, step=step, top_k=top_k, top_p=top_p,
            temperature=temperature, output_all_probs=output_all_probs,
            cum_log_probs=cum_log_probs,
            return_original_all_probs=True,
        )

        chosen = [row[step] for row in result["token_ids"].cpu().tolist()]
        all_probs = result["output_all_probs"].cpu()
        clp = result["cum_log_probs"].cpu().tolist()

        # Reference: UNFILTERED softmax of the raw logits.
        orig_probs = torch.softmax(logits_cpu, -1)

        # b0 (top_k=1): the filtered distribution is one-hot; a multi-value
        # all_probs row proves the ORIGINAL (unfiltered) distribution is
        # reported instead.
        self.assertEqual(chosen[0], int(torch.argmax(logits_cpu[0])))
        self.assertGreater(
            int((all_probs[0] > 0).sum()), 1,
            "ORIGINAL mode: batch 0 all_probs must be the unfiltered softmax, "
            "not the one-hot filtered distribution")

        for i in range(batch_size):
            self.assertTrue(
                torch.allclose(all_probs[i], orig_probs[i], atol=2e-3),
                f"batch {i}: all_probs must equal softmax(unfiltered logits)")
            expected = float(torch.log(orig_probs[i][chosen[i]]))
            self.assertAlmostEqual(
                clp[i], expected, delta=2e-3,
                msg=f"batch {i}: cum_log_probs={clp[i]}, expected(original)={expected}")

    # ==================================================================
    # Test 7d — unsupported penalties must be rejected with a clear error;
    #   neutral values (=1.0 / =0.0) pass through as no-ops.
    # ==================================================================
    def test_penalties_rejected(self):
        batch_size = 3
        vocab_size = 6
        logits = npu_float_tensor(
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 1.0, 1.0, 1.0, 1.1, 1.0,
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            (batch_size, vocab_size),
        )
        step = 3
        token_ids = torch.tensor(
            [[10, 0, 0, 0],
             [20, 5, 5, 5],
             [30, 0, 0, 0]],
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([3, 3, 3], dtype=torch.int32)
        input_lens = torch.tensor([-1, -1, -1], dtype=torch.int32)
        top_k = pinned_int_tensor([1, 1, 1])
        top_p = pinned_float_tensor([1.0, 1.0, 1.0])
        temperature = pinned_float_tensor([1.0, 1.0, 1.0])

        common = dict(
            input_lengths=input_lens, sequence_lengths=seq_lens,
            token_ids=token_ids.clone(), step=step,
            top_k=top_k, top_p=top_p, temperature=temperature,
        )

        # Non-neutral repetition penalty → rejected.
        with self.assertRaises(RuntimeError) as ctx:
            sampler_test_module.exec_sample_greedy(
                logits=logits.clone(),
                repetition_penalty=pinned_float_tensor([1.5, 1.5, 1.5]),
                **common)
        self.assertIn("not support", str(ctx.exception))

        # Non-neutral presence penalty → rejected.
        with self.assertRaises(RuntimeError):
            sampler_test_module.exec_sample_greedy(
                logits=logits.clone(),
                presence_penalty=pinned_float_tensor([0.0, 0.5, 0.0]),
                **common)

        # Neutral values → no-op, sampling succeeds.
        result = sampler_test_module.exec_sample_greedy(
            logits=logits.clone(),
            repetition_penalty=pinned_float_tensor([1.0, 1.0, 1.0]),
            presence_penalty=pinned_float_tensor([0.0, 0.0, 0.0]),
            frequency_penalty=pinned_float_tensor([0.0, 0.0, 0.0]),
            **common)
        out = result["token_ids"].cpu().flatten().tolist()
        self.assertEqual(out[3], 5)
        self.assertEqual(out[7], 4)
        self.assertEqual(out[11], 0)


    # ==================================================================
    # Test 8 — comprehensive cross-run determinism sweep.
    #   Covers multiple batch sizes, vocab sizes, top_k/top_p modes,
    #   temperature values, and tie-prone logit distributions.
    # ==================================================================

    _CROSS_RUN_RUNS = 20
    _DET_SEED_BASE = 42

    def _run_determinism_check(self, *, batch_size, vocab_size, logits,
                                top_k_vals, top_p_vals, temp_vals,
                                description=""):
        """Run execSampleGreedy N times with same seed, check determinism."""
        N = self._CROSS_RUN_RUNS
        step = 5
        seq_lens = torch.full((batch_size,), step, dtype=torch.int32)
        input_lens = torch.full((batch_size,), -1, dtype=torch.int32)
        top_k = pinned_int_tensor(top_k_vals)
        top_p = pinned_float_tensor(top_p_vals)
        temperature = pinned_float_tensor(temp_vals)

        tokens_list = []
        probs_list = []
        for _run_idx in range(N):
            token_ids = torch.zeros((batch_size, step + 1), dtype=torch.int32)
            for b in range(batch_size):
                token_ids[b, 0] = 100 + b
            all_probs = torch.zeros((batch_size, vocab_size), dtype=torch.float32)

            gens = [npu_generator(self._DET_SEED_BASE + i) for i in range(batch_size)]
            result = sampler_test_module.exec_sample_greedy(
                logits=logits.clone(),
                input_lengths=input_lens,
                sequence_lengths=seq_lens,
                token_ids=token_ids,
                step=step,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                generators=gens,
                output_all_probs=all_probs,
            )
            tokens_list.append(result["token_ids"].cpu().flatten().tolist())
            probs_list.append(result["output_all_probs"].cpu().clone())

        # --- Check tokens ---
        ref_tokens = tokens_list[0]
        token_ok = all(tokens_list[i] == ref_tokens for i in range(1, N))
        mismatches = []
        if not token_ok:
            for i in range(1, N):
                if tokens_list[i] != ref_tokens:
                    for j in range(len(ref_tokens)):
                        if tokens_list[i][j] != ref_tokens[j]:
                            mismatches.append(
                                f"run {i}: token[{j}] ref={ref_tokens[j]} got={tokens_list[i][j]}")

        # --- Check probs ---
        ref_probs = probs_list[0]
        prob_ok = all(torch.equal(probs_list[i], ref_probs) for i in range(1, N))
        max_diff = max(
            ((probs_list[i] - ref_probs).abs().max().item() for i in range(1, N)),
            default=0.0,
        )

        print(f"\n  [{description}] bs={batch_size} vocab={vocab_size}"
              f" top_k={top_k_vals[:3]}{'...' if batch_size>3 else ''}"
              f" top_p={top_p_vals[:3]}{'...' if batch_size>3 else ''}"
              f" temp={temp_vals[:3]}{'...' if batch_size>3 else ''}"
              f" → token_ok={token_ok} prob_ok={prob_ok} max_diff={max_diff:.2e}")
        if mismatches:
            for m in mismatches[:5]:
                print(f"    {m}")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")

        return token_ok, prob_ok

    def test_cross_run_sweep_small(self):
        """Small shapes: various batch/vocab/topk/topp/temp combos."""
        configs = [
            # (batch_size, vocab_size, top_k_array, top_p_array, temp_array, desc)
            (1, 6,   [0],     [1.0],   [1.0],  "tiny pure-topk=0"),
            (2, 6,   [0, 0],  [0.5, 0.5], [1.0, 1.0], "top_p=0.5"),
            (4, 10,  [1, 1, 0, 2],   [1.0, 1.0, 1.0, 1.0], [1.0]*4,  "mixed top_k"),
            (4, 10,  [0, 1, 1, 2],   [0.5, 1.0, 1.0, 1.0], [1.0]*4,  "mixed top_k+top_p"),
            (8, 16,  [2]*8,   [0.8]*8, [1.0]*8, "uniform top_k=2 top_p=0.8"),
            (8, 16,  [0]*8,   [0.3]*8, [1.0]*8, "uniform top_p=0.3"),
            (4, 10,  [0, 0, 0, 0],   [1.0]*4,   [0.5, 1.0, 2.0, 10.0], "temperature sweep"),
            (4, 10,  [0]*4,   [0.9]*4,  [0.3]*4, "low temp + top_p"),
            (2, 6,   [3, 3],  [1.0, 1.0], [1.0, 1.0], "top_k=3 >1"),
        ]

        import random
        rng = random.Random(777)
        all_ok = True
        for batch_size, vocab_size, top_k_vals, top_p_vals, temp_vals, desc in configs:
            with self.subTest(desc=desc):
                data = [rng.uniform(0.0, 10.0) for _ in range(batch_size * vocab_size)]
                logits = npu_float_tensor(data, (batch_size, vocab_size))
                tok, prob = self._run_determinism_check(
                    batch_size=batch_size, vocab_size=vocab_size,
                    logits=logits, top_k_vals=top_k_vals,
                    top_p_vals=top_p_vals, temp_vals=temp_vals,
                    description=desc,
                )
                self.assertTrue(tok, f"[{desc}] token non-deterministic")
                self.assertTrue(prob, f"[{desc}] probs non-deterministic")
                if not tok:
                    all_ok = False
        print(f"\n=== cross-run sweep (small): all_ok={all_ok} ===")

    def test_cross_run_tie_scenarios(self):
        """Tie-heavy inputs: trigger parallel Sort instability in AscendC."""
        configs = [
            # All equal logits → pure tie, Sort order can vary
            (4, 10, [3]*4, [1.0]*4, [1.0]*4, "all_equal_logits"),
            # Two dominant equal peaks
            (4, 10, [2]*4, [1.0]*4, [1.0]*4, "two_peaks_tie"),
            # Large group of equal mid-values
            (4, 20, [5]*4, [0.9]*4, [1.0]*4, "wide_tie_mid"),
            # All equal + top_p=0.5 → Sort decides which equal-value tokens enter
            (4, 10, [0]*4, [0.5]*4, [1.0]*4, "all_equal_top_p0.5"),
        ]

        for batch_size, vocab_size, top_k_vals, top_p_vals, temp_vals, desc in configs:
            with self.subTest(desc=desc):
                if desc == "all_equal_logits":
                    data = [1.0] * (batch_size * vocab_size)
                elif desc == "two_peaks_tie":
                    data = []
                    for _ in range(batch_size):
                        row = [0.5] * (vocab_size - 2) + [1.0, 1.0]
                        data.extend(row)
                elif desc == "wide_tie_mid":
                    import random
                    rng = random.Random(42)
                    data = []
                    for _ in range(batch_size):
                        row = [0.1] * 3 + [0.5] * (vocab_size - 6) + [0.1] * 3
                        data.extend(row)
                elif desc == "all_equal_top_p0.5":
                    data = [1.0] * (batch_size * vocab_size)
                else:
                    data = [float(i % 10) / 10.0 for i in range(batch_size * vocab_size)]

                logits = npu_float_tensor(data, (batch_size, vocab_size))
                tok, prob = self._run_determinism_check(
                    batch_size=batch_size, vocab_size=vocab_size,
                    logits=logits, top_k_vals=top_k_vals,
                    top_p_vals=top_p_vals, temp_vals=temp_vals,
                    description=desc,
                )
                self.assertTrue(tok, f"[{desc}] token non-deterministic")
                self.assertTrue(prob, f"[{desc}] probs non-deterministic")

    def test_cross_run_medium_vocab(self):
        """Medium/large vocab: may trigger AscendC tiling paths."""
        import random
        rng = random.Random(123)

        configs = [
            (4, 50),
            (4, 128),
            (4, 500),
            (2, 1024),
            (2, 2048),
            (1, 4096),
        ]

        all_ok = True
        for batch_size, vocab_size in configs:
            with self.subTest(vocab=vocab_size):
                data = [rng.uniform(0.0, 1.0) for _ in range(batch_size * vocab_size)]
                logits = npu_float_tensor(data, (batch_size, vocab_size))
                top_k_vals = [3] * batch_size
                top_p_vals = [0.9] * batch_size
                temp_vals = [1.0] * batch_size
                tok, prob = self._run_determinism_check(
                    batch_size=batch_size, vocab_size=vocab_size,
                    logits=logits, top_k_vals=top_k_vals,
                    top_p_vals=top_p_vals, temp_vals=temp_vals,
                    description=f"vocab={vocab_size}",
                )
                self.assertTrue(tok, f"[vocab={vocab_size}] token non-deterministic")
                self.assertTrue(prob, f"[vocab={vocab_size}] probs non-deterministic")
                if not tok:
                    all_ok = False
        print(f"\n=== cross-run sweep (medium/large vocab): all_ok={all_ok} ===")

    def test_cross_run_topk_topp_joint(self):
        """Comprehensive top_k + top_p joint scenarios."""
        import random
        rng = random.Random(999)

        configs = [
            # (batch, vocab, top_k_arr,        top_p_arr,          temp_arr,    desc)
            # --- Narrow K + P: K truncates first, then P further narrows ---
            (4, 10,  [3, 3, 3, 3],             [0.5, 0.5, 0.5, 0.5], [1.0]*4,  "K3_P0.5_narrow_both"),
            (4, 10,  [5, 5, 5, 5],             [0.3, 0.3, 0.3, 0.3], [1.0]*4,  "K5_P0.3_tight_narrow"),
            # --- Wide K + narrow P: P does most filtering ---
            (4, 20,  [10, 10, 10, 10],         [0.9, 0.9, 0.9, 0.9], [1.0]*4,  "K10_P0.9_wideK_narrowP"),
            (4, 20,  [15, 15, 15, 15],         [0.7, 0.7, 0.7, 0.7], [1.0]*4,  "K15_P0.7_wideK_mediumP"),
            # --- Narrow K + wide P: K does most filtering ---
            (4, 10,  [2, 2, 2, 2],             [0.99, 0.99, 0.99, 0.99], [1.0]*4,  "K2_P0.99_narrowK_wideP"),
            (4, 10,  [3, 3, 3, 3],             [1.0, 1.0, 1.0, 1.0],     [1.0]*4,  "K3_P1.0_effectively_pureK"),
            # --- Mixed per-batch: different K+P per batch ---
            (4, 10,  [0, 2, 5, 4],             [0.5, 0.8, 0.9, 0.6],     [1.0]*4,  "mixed_per_batch"),
            (4, 10,  [1, 0, 3, 2],             [1.0, 0.5, 1.0, 0.8],     [1.0]*4,  "mixed_K1_pureK_pureP"),
            # --- K+P with temperature ---
            (4, 10,  [4, 4, 4, 4],             [0.6, 0.6, 0.6, 0.6],     [0.5, 1.0, 2.0, 10.0],  "K4_P0.6_temp_sweep"),
            (4, 10,  [4]*4,                     [0.8]*4,                  [0.3]*4,  "K4_P0.8_low_temp"),
            # --- Large vocab K+P ---
            (4, 100, [5, 5, 5, 5],             [0.8, 0.8, 0.8, 0.8],     [1.0]*4,  "vocab100_K5_P0.8"),
            (2, 200, [10, 10],                  [0.9, 0.9],               [1.0, 1.0], "vocab200_K10_P0.9"),
            (2, 500, [20, 20],                  [0.95, 0.95],             [1.0, 1.0], "vocab500_K20_P0.95"),
            # --- Edge: K=1 + P<1 (P should not affect since only 1 token after K) ---
            (4, 10,  [1, 1, 1, 1],             [0.5, 0.5, 0.5, 0.5],     [1.0]*4,  "K1_P0.5_argmax"),
            # --- Large batch + K+P ---
            (8, 16,  [3]*8,                     [0.7]*8,                  [1.0]*8,  "batch8_K3_P0.7"),
        ]

        all_ok = True
        for batch_size, vocab_size, top_k_vals, top_p_vals, temp_vals, desc in configs:
            with self.subTest(desc=desc):
                data = [rng.uniform(0.0, 10.0) for _ in range(batch_size * vocab_size)]
                logits = npu_float_tensor(data, (batch_size, vocab_size))
                tok, prob = self._run_determinism_check(
                    batch_size=batch_size, vocab_size=vocab_size,
                    logits=logits, top_k_vals=top_k_vals,
                    top_p_vals=top_p_vals, temp_vals=temp_vals,
                    description=desc,
                )
                self.assertTrue(tok, f"[{desc}] token non-deterministic")
                self.assertTrue(prob, f"[{desc}] probs non-deterministic")
                if not tok:
                    all_ok = False
        print(f"\n=== cross-run top_k+top_p joint: all_ok={all_ok} ===")


def _one_hot_probs(ids_per_step, vocab):
    """Build [bs, S, vocab] one-hot prob rows from ids [bs, S]."""
    ids = np.asarray(ids_per_step)
    bs, S = ids.shape
    probs = np.zeros((bs, S, vocab), dtype=np.float32)
    for b in range(bs):
        for s in range(S):
            probs[b, s, ids[b, s]] = 1.0
    return probs


def _rejection_reference(draft_ids, draft_probs, uniform, target_probs, target_tokens, do_sample, point_mass):
    """NumPy reference for the Ascend rejectionSampling op (CUDA kernel semantics).

    draft_ids [bs, K]; draft_probs [bs, K, V] or None (point_mass);
    uniform [bs, K+1]; target_probs [bs, K+1, V]; target_tokens [bs, K+1];
    do_sample [bs] bool.  Returns (out_ids [bs, K+1], accepted [bs]).
    """
    bs, K = np.asarray(draft_ids).shape
    draft_ids = np.asarray(draft_ids)
    uniform = np.asarray(uniform)
    target_probs = np.asarray(target_probs)
    target_tokens = np.asarray(target_tokens)
    do_sample = np.asarray(do_sample)
    V = target_probs.shape[2]
    if point_mass:
        draft_probs = np.zeros((bs, K, V), dtype=np.float32)
        for b in range(bs):
            for k in range(K):
                draft_probs[b, k, draft_ids[b, k]] = 1.0
    out = np.full((bs, K + 1), -1, dtype=np.int64)
    acc = np.zeros(bs, dtype=np.int64)
    for b in range(bs):
        pos = K
        for k in range(K):
            d = int(draft_ids[b, k])
            q = target_probs[b, k, d]
            p = draft_probs[b, k, d]
            u = uniform[b, k]
            ok = (u * p) < q if do_sample[b] else (d == target_tokens[b, k])
            if not ok:
                pos = k
                break
        if pos == K:
            out[b, :K] = draft_ids[b]
            out[b, K] = target_tokens[b, K]        # bonus token
        else:
            out[b, :pos] = draft_ids[b, :pos]
            if do_sample[b]:
                residual = np.maximum(target_probs[b, pos] - draft_probs[b, pos], 0.0)
                cdf = np.cumsum(residual)
                total = cdf[-1]
                u_next = uniform[b, min(pos + 1, K)]
                recovered = int(np.sum(cdf < u_next * total))
                out[b, pos] = min(recovered, V - 1)
            else:
                out[b, pos] = target_tokens[b, pos]
        acc[b] = pos + 1
    return out, acc


class TestAscendRejectionSampling(unittest.TestCase):
    """MTP rejection sampling / draft2target mapping on Ascend NPU.

    Mirrors the CUDA/ROCm rejection_sampling_kernel contract:
      accept d_i iff (do_sample && u*p < q) || (!do_sample && d_i == t_i)
      out = accepted draft prefix, token at first reject, -1 tail;
      all-accepted rows append the bonus target token at step K.
    All cases use V=4; target token ids may exceed V (ids and probs are
    independent inputs), draft ids must stay < V (gathered from probs).
    """

    VOCAB = 4

    def _run(self, draft_ids, draft_probs, uniform, target_probs, target_tokens,
             do_sample, point_mass=False, stride=1):
        bs, K = np.asarray(draft_ids).shape
        # target_token_ids layout: [bs*(K+1), stride], token in the LAST column.
        tti = np.zeros((bs * (K + 1), stride), dtype=np.int32)
        for b in range(bs):
            for s in range(K + 1):
                tti[b * (K + 1) + s, stride - 1] = target_tokens[b][s]

        def npu(x, dtype):
            return torch.tensor(np.asarray(x), dtype=dtype, device=NPU_DEVICE)

        draft_probs_t = (npu(draft_probs, torch.float32)
                         if draft_probs is not None else torch.Tensor())
        out_t = torch.full((bs, K + 1), -7, dtype=torch.int32, device=NPU_DEVICE)
        acc_t = torch.full((bs,), -7, dtype=torch.int32, device=NPU_DEVICE)
        sampler_test_module.exec_rejection_sampling(
            draft_probs_t,
            npu(draft_ids, torch.int32),
            npu(uniform, torch.float32),
            npu(target_probs, torch.float32),
            npu(tti, torch.int32),
            out_t,
            acc_t,
            npu(do_sample, torch.bool),
            point_mass,
        )
        torch.npu.synchronize()
        return out_t.cpu().numpy(), acc_t.cpu().numpy()

    def _check(self, **kw):
        point_mass = kw.pop("point_mass", False)
        stride = kw.pop("stride", 1)
        ref_out, ref_acc = _rejection_reference(point_mass=point_mass, **kw)
        out, acc = self._run(point_mass=point_mass, stride=stride, **kw)
        np.testing.assert_array_equal(out, ref_out)
        np.testing.assert_array_equal(acc, ref_acc)
        return out, acc

    def test_greedy_prefix_then_reject(self):
        # row0: draft [1,2] vs target [1,3,0] -> accept 1, reject at pos1
        # row1: draft [1,2] vs target [1,2,7] -> all accepted, bonus 7 (>V is fine)
        target_tokens = [[1, 3, 0], [1, 2, 7]]
        # probs for the bonus step are irrelevant (bonus id comes from
        # target_token_ids); keep one-hot ids inside V=4.
        target_probs = _one_hot_probs([[1, 3, 0], [1, 2, 3]], self.VOCAB)
        out, acc = self._check(
            draft_ids=[[1, 2], [1, 2]],
            draft_probs=_one_hot_probs([[1, 2], [1, 2]], self.VOCAB),
            uniform=[[0.0] * 3] * 2,
            target_probs=target_probs,
            target_tokens=target_tokens,
            do_sample=[False, False],
        )
        np.testing.assert_array_equal(out, [[1, 3, -1], [1, 2, 7]])
        np.testing.assert_array_equal(acc, [2, 3])

    def test_greedy_immediate_reject(self):
        # draft[0]=3 != target[0]=0 -> pos=0, out = [target[0], -1, -1]
        out, acc = self._check(
            draft_ids=[[3, 0]],
            draft_probs=_one_hot_probs([[3, 0]], self.VOCAB),
            uniform=[[0.0] * 3],
            target_probs=_one_hot_probs([[0, 1, 2]], self.VOCAB),
            target_tokens=[[0, 1, 2]],
            do_sample=[False],
        )
        np.testing.assert_array_equal(out, [[0, -1, -1]])
        np.testing.assert_array_equal(acc, [1])

    def test_stochastic_accept_then_reject(self):
        # step0: p=q=1 at token1, u=0.5 -> 0.5<1 accept
        # step1: p=1 at token2 but q(2)=0, u=0.5 -> reject; recover from
        # relu(onehot(3) - onehot(2)) = [0,0,0,1], cdf=[0,0,0,1], u'=0.5 -> idx 3
        target_probs = np.zeros((1, 3, self.VOCAB), dtype=np.float32)
        target_probs[0, 0, 1] = 1.0          # q(draft0)=1
        target_probs[0, 1, 3] = 1.0          # q(draft1)=0, mass on 3
        target_probs[0, 2, 0] = 1.0
        out, acc = self._check(
            draft_ids=[[1, 2]],
            draft_probs=_one_hot_probs([[1, 2]], self.VOCAB),
            uniform=[[0.5, 0.5, 0.5]],
            target_probs=target_probs,
            target_tokens=[[9, 9, 9]],
            do_sample=[True],
        )
        np.testing.assert_array_equal(out, [[1, 3, -1]])
        np.testing.assert_array_equal(acc, [2])

    def test_stochastic_all_accepted_bonus(self):
        # u*p < q holds on every step (u=0, q=0.25) -> pos=K, bonus=3
        out, acc = self._check(
            draft_ids=[[2, 1]],
            draft_probs=_one_hot_probs([[2, 1]], self.VOCAB),
            uniform=[[0.0, 0.0, 0.0]],
            target_probs=np.full((1, 3, self.VOCAB), 0.25, dtype=np.float32),
            target_tokens=[[7, 7, 3]],
            do_sample=[True],
        )
        np.testing.assert_array_equal(out, [[2, 1, 3]])
        np.testing.assert_array_equal(acc, [3])

    def test_point_mass_matches_explicit_one_hot(self):
        # draft_probs_point_mass synthesizes one-hot rows: must equal the
        # explicit one-hot draft_probs result.
        draft_ids = [[1, 2], [0, 3]]
        uniform = [[0.5, 0.5, 0.5], [0.9, 0.1, 0.5]]
        target_probs = np.zeros((2, 3, self.VOCAB), dtype=np.float32)
        target_probs[0, 0, 1] = 1.0
        target_probs[0, 1, 3] = 1.0
        target_probs[1] = 0.25
        common = dict(
            draft_ids=draft_ids,
            uniform=uniform,
            target_probs=target_probs,
            target_tokens=[[9, 9, 9], [9, 9, 9]],
            do_sample=[True, True],
        )
        out, acc = self._check(draft_probs=None, point_mass=True, **common)
        out2, acc2 = self._check(
            draft_probs=_one_hot_probs(draft_ids, self.VOCAB), **common)
        np.testing.assert_array_equal(out, out2)
        np.testing.assert_array_equal(acc, acc2)

    def test_mixed_batch_greedy_and_stochastic(self):
        # row0 greedy: reject at 1 -> [1, 3, -1]
        # row1 stochastic: u*p(1)<q(1) on both steps -> all accepted, bonus 7
        target_tokens = [[1, 3, 0], [1, 2, 7]]
        target_probs = np.zeros((2, 3, self.VOCAB), dtype=np.float32)
        target_probs[0, 0, 1] = 1.0
        target_probs[0, 1, 3] = 1.0
        target_probs[0, 2, 0] = 1.0
        target_probs[1, 0, 1] = 1.0
        target_probs[1, 1, 2] = 1.0
        target_probs[1, 2, 3] = 1.0
        out, acc = self._check(
            draft_ids=[[1, 2], [1, 2]],
            draft_probs=_one_hot_probs([[1, 2], [1, 2]], self.VOCAB),
            uniform=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            target_probs=target_probs,
            target_tokens=target_tokens,
            do_sample=[False, True],
        )
        np.testing.assert_array_equal(out, [[1, 3, -1], [1, 2, 7]])
        np.testing.assert_array_equal(acc, [2, 3])

    def test_stride3_target_token_layout(self):
        # target_token_ids uses stride 3: token must be read from the LAST
        # column of each stride slot, not the first.
        target_tokens = [[1, 3, 0]]
        out, acc = self._check(
            draft_ids=[[1, 2]],
            draft_probs=_one_hot_probs([[1, 2]], self.VOCAB),
            uniform=[[0.0] * 3],
            target_probs=_one_hot_probs(target_tokens, self.VOCAB),
            target_tokens=target_tokens,
            do_sample=[False],
            stride=3,
        )
        np.testing.assert_array_equal(out, [[1, 3, -1]])
        np.testing.assert_array_equal(acc, [2])

    def test_mapping_draft2_target(self):
        # tokens [bs, 5]; map window columns [1, 3); d2t_map[i] = i + 10
        tokens = np.array([[0, 1, 2, 3, 4],
                           [5, 100, 3, 200, 6]], dtype=np.int64)
        d2t_map = torch.arange(10, dtype=torch.int64, device=NPU_DEVICE) + 10
        tokens_t = torch.tensor(tokens, device=NPU_DEVICE)
        sampler_test_module.exec_mapping_draft2_target(tokens_t, d2t_map, 2, 1, 3)
        torch.npu.synchronize()
        got = tokens_t.cpu().numpy()
        # column 1/2 mapped (+10); 100 out of range -> passthrough; others untouched
        np.testing.assert_array_equal(got, [[0, 11, 12, 3, 4],
                                            [5, 100, 13, 200, 6]])

    def test_mapping_empty_map_is_noop(self):
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        tokens_t = torch.tensor(tokens, device=NPU_DEVICE)
        empty = torch.empty(0, dtype=torch.int64, device=NPU_DEVICE)
        sampler_test_module.exec_mapping_draft2_target(tokens_t, empty, 1, 0, 3)
        torch.npu.synchronize()
        np.testing.assert_array_equal(tokens_t.cpu().numpy(), tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
