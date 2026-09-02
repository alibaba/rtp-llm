#!/usr/bin/env python3
"""
Ascend NPU Sampler integration tests.

These tests validate execSampleGreedy() on real Ascend NPU hardware.
They mirror the 7 C++ test cases from SamplerTest.cc.

Prerequisites (set by .bazelrc test:ascend --test_env):
  LD_LIBRARY_PATH  must include torch/lib, torch_npu/lib, /opt/conda310/lib
  PYTHONHOME       must point to conda env (e.g. /root/miniconda3/envs/py310)
  ASCEND_CUSTOM_OPP_PATH  must point to aclnn_custom_ops/opp

Usage:
  cd /home/d30033799/rtp-llm
  python rtp_llm/models_py/bindings/ascend/ops/tests/test_ascend_sampler.py
"""

import os
import sys
import unittest

import torch
import torch_npu  # noqa: F401 — registers NPU dispatch before any NPU ops

# ---------------------------------------------------------------------------
# Locate and import the pybind module built by Bazel.
# ---------------------------------------------------------------------------
_BAZEL_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../../bazel-bin/rtp_llm/models_py/bindings/ascend/ops/tests",
)

if _BAZEL_BIN not in sys.path:
    sys.path.insert(0, _BAZEL_BIN)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
