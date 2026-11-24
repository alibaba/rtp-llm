"""Streaming-policy contracts against frozen RTP and a local-error oracle."""

import gc
import hashlib
import json
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.eplb import policy
from rtp_llm.eplb.test import policy_reference
from rtp_llm.eplb.test.policy_larges_reference import (
    GROUPS,
    assert_local_choice,
    assert_same_outputs,
    history,
    reduction_depths,
    replay_decisions,
)


@contextmanager
def observed_streaming():
    from rtp_llm.eplb import policy_fused_larges as module

    trace = {}

    def observe(name, function):
        def invoke(*args, **kwargs):
            result = function(*args, **kwargs)
            assert name not in trace, f"duplicate {name} phase"
            trace[name] = result
            return result

        return invoke

    with patch.object(
        module,
        "rebalance_experts_global_streaming",
        wraps=module.rebalance_experts_global_streaming,
    ) as entry, patch.object(
        module,
        "_replica_phase_streaming",
        side_effect=observe("replica", module._replica_phase_streaming),
    ), patch.object(
        module,
        "_batch_greedy_indices_stream",
        side_effect=observe("greedy", module._batch_greedy_indices_stream),
    ):
        yield module, entry, trace


def checked_plan(source, physical, local, exact=False):
    original = source.clone()
    identity = (source.data_ptr(), source.stride(), source.storage_offset())
    expected = policy_reference.rebalance_experts_global(source, physical, local)
    with observed_streaming() as (module, entry, trace):
        actual = policy.rebalance_experts_global(source, physical, local)
        assert entry.call_count == 1, "production did not enter streaming policy"
    depths = reduction_depths(
        source.size(0),
        source.size(1),
        module._REPLICA_CHUNK_S,
        module._REPLICA_TARGET_CTAS,
        module._GREEDY_CHUNK_S,
    )
    rebuilt = replay_decisions(
        source, physical, local, trace.get("replica"), trace.get("greedy"), depths
    )
    assert_same_outputs(actual, rebuilt)
    assert torch.equal(source, original), "history was modified"
    assert identity == (source.data_ptr(), source.stride(), source.storage_offset())
    if exact:
        assert_same_outputs(actual, expected)
    return actual, all(torch.equal(a, b) for a, b in zip(actual, expected))


class PolicyLargeCpuTest(unittest.TestCase):
    def test_reference_is_unchanged(self):
        self.assertEqual(
            hashlib.sha256(Path(policy_reference.__file__).read_bytes()).hexdigest(),
            "56b5ee082422c3ed526357a0393d731e2d3970185501419291eaf5132ae82f3a",
        )

    def test_local_oracle_rejects_bad_choice_even_with_close_top_two(self):
        scores = torch.tensor([[1, 1.0000001, 100]], dtype=torch.float64)
        assert_local_choice(scores, torch.tensor([1]), 20, "near tie")
        with self.assertRaises(AssertionError):
            assert_local_choice(scores, torch.tensor([2]), 20, "wrong third candidate")
        # A permitted earlier divergence must not exempt a later bad choice.
        assert_local_choice(scores, torch.tensor([1]), 20, "earlier round")
        with self.assertRaises(AssertionError):
            assert_local_choice(scores * 2, torch.tensor([2]), 20, "later round")

    def test_reduction_depth_and_local_budget_boundary(self):
        for steps, layers, expected in (
            (100, 40, (21, 10)),
            (1000, 40, (77, 11)),
            (4097, 40, (118, 18)),
            (17, 4100, (16, 10)),
        ):
            with self.subTest(steps=steps, layers=layers):
                depths = reduction_depths(steps, layers, 16, 4096, 512)
                self.assertEqual(depths, expected)
                for depth in depths:
                    # For scores [1, 1+gap], solve gap <= g*(2+gap)
                    # independently. A choice just outside the derived bound
                    # must fail; close top-two scores alone cannot exempt it.
                    u = 2.0**-24
                    gamma = depth * u / (1 - depth * u)
                    d = steps * 2.0**-53
                    gamma_double = d / (1 - d)
                    g = (gamma + gamma_double) / (1 - gamma_double)
                    cutoff = 2 * g / (1 - g)
                    chosen = torch.tensor([1])
                    scores = torch.tensor([[1, 1 + 0.99 * cutoff]], dtype=torch.float64)
                    regret, budget = assert_local_choice(
                        scores, chosen, depth, "inside bound", steps
                    )
                    self.assertTrue(bool((regret <= budget).all()))
                    scores[0, 1] = 1 + 1.01 * cutoff
                    with self.assertRaises(AssertionError):
                        assert_local_choice(
                            scores, chosen, depth, "outside bound", steps
                        )
        # A changed production chunk needs a new derivation, not reuse of a
        # proof that only covers the current compiler's 16/512 lowering.
        with self.assertRaises(AssertionError):
            reduction_depths(100, 40, 32, 4096, 512)
        with self.assertRaises(AssertionError):
            reduction_depths(100, 40, 16, 4096, 256)

    def test_reconstruction_rejects_corruption_without_decision_divergence(self):
        source = torch.zeros((3, 1, 2), dtype=torch.int32)
        expected = replay_decisions(source, 2, 1, None, None, (1, 1))
        for index in range(3):
            with self.subTest(output=index):
                corrupted = [tensor.clone() for tensor in expected]
                corrupted[index].view(-1)[0] += 1
                with self.assertRaises(AssertionError):
                    assert_same_outputs(corrupted, expected)
        with self.assertRaises(AssertionError):
            replay_decisions(source, 2, 2, None, torch.tensor([[0, 0]]), (1, 1))

    def test_replica_slot_and_count_are_strict(self):
        source = torch.zeros((3, 1, 2), dtype=torch.int32)
        trace = (
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[1, 2]], dtype=torch.int32),
        )
        replay_decisions(source, 3, 1, trace, None, (4, 4))
        for index in (1, 2):
            corrupted = [tensor.clone() for tensor in trace]
            corrupted[index].view(-1)[0] += 1
            with self.assertRaises(AssertionError):
                replay_decisions(source, 3, 1, corrupted, None, (4, 4))

    def test_cpu_large_windows_do_not_import_triton_or_initialize_cuda(self):
        script = """
import builtins
import torch
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if ('triton' in name or 'policy_fused' in name
        or any('policy_fused' in item for item in (fromlist or ()))):
        raise AssertionError('CPU imported ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
from rtp_llm.eplb.policy import rebalance_experts_global
from rtp_llm.eplb.test.policy_reference import rebalance_experts_global as reference
for steps in (100, 1000):
    source = torch.arange(steps * 6, dtype=torch.int32).reshape(steps, 1, 6) % 31
    actual = rebalance_experts_global(source, 8, 4)
    expected = reference(source, 8, 4)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
assert not torch.cuda.is_initialized()
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class PolicyLargeGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.device.device_type import is_cuda

        if not is_cuda() or not torch.cuda.is_available():
            raise RuntimeError("Streaming GPU tests require NVIDIA CUDA")

    def test_production_shapes_all_rounds_and_frozen_reference(self):
        exact_count = 0
        for steps in (100, 1000):
            for name, layers, experts, physical, local in GROUPS:
                for kind in ("zipf", "sparse", "bimodal"):
                    for seed in (0, 1):
                        with self.subTest(
                            steps=steps, group=name, kind=kind, seed=seed
                        ):
                            source = history(kind, steps, layers, experts, seed).cuda()
                            _, exact = checked_plan(
                                source, physical, local, exact=kind == "bimodal"
                            )
                            exact_count += exact
        print(json.dumps({"production_cases": 24, "exact_matches": exact_count}))

    def test_chunk_tails_and_layout_boundaries(self):
        steps_values = (
            3,
            5,
            15,
            16,
            17,
            99,
            100,
            101,
            127,
            128,
            129,
            511,
            512,
            513,
            999,
            1000,
            1001,
            1025,
        )
        for steps in steps_values:
            for kind in ("zeros", "tail", "sparse"):
                with self.subTest(steps=steps, kind=kind):
                    checked_plan(
                        history(kind, steps, 2, 7, 735).cuda(),
                        12,
                        4,
                        exact=kind in ("zeros", "tail"),
                    )
        for experts, physical, local in ((1, 4, 2), (7, 7, 7), (7, 12, 1), (7, 12, 12)):
            with self.subTest(experts=experts, physical=physical, local=local):
                checked_plan(
                    history("sparse", 1000, 2, experts, 1).cuda(), physical, local
                )

    def test_replica_multiple_chunks_and_layer_count_above_cta_target(self):
        # S=4097/L=40 has 102 slices of up to 41 rows: three 16-row
        # chunks, with both slice-boundary and final-input padding.
        # L=4100 exceeds the CTA target, forcing one slice with two chunks.
        for steps, layers, experts, physical, local in (
            (4097, 40, 7, 12, 4),
            (17, 4100, 3, 4, 2),
        ):
            for kind in ("tail", "sparse"):
                with self.subTest(steps=steps, layers=layers, kind=kind):
                    checked_plan(
                        history(kind, steps, layers, experts, 735).cuda(),
                        physical,
                        local,
                        exact=kind == "tail",
                    )

    def test_large_window_dtype_layout_default_dtype_and_backend_fallbacks(self):
        from rtp_llm.eplb import policy_fused_larges

        def check(source):
            original = source.clone()
            identity = (source.data_ptr(), source.stride(), source.storage_offset())
            expected = policy_reference.rebalance_experts_global(source, 12, 3)
            actual = policy.rebalance_experts_global(source, 12, 3)
            assert_same_outputs(actual, expected)
            self.assertTrue(torch.equal(source, original))
            self.assertEqual(
                identity, (source.data_ptr(), source.stride(), source.storage_offset())
            )

        with patch.object(
            policy_fused_larges,
            "rebalance_experts_global_streaming",
            side_effect=AssertionError("fallback input entered streaming policy"),
        ):
            source = history("sparse", 100, 2, 6, 735).cuda()
            for dtype in (torch.int64, torch.float32, torch.float64):
                with self.subTest(dtype=dtype):
                    converted = source.to(dtype)
                    if dtype.is_floating_point:
                        converted += 0.125
                    check(converted)
            contiguous = history("sparse", 100, 3, 12, 736).cuda()
            for noncontiguous in (contiguous[..., ::2], contiguous.permute(0, 2, 1)):
                with self.subTest(stride=noncontiguous.stride()):
                    self.assertFalse(noncontiguous.is_contiguous())
                    check(noncontiguous)
            previous = torch.get_default_dtype()
            try:
                for dtype in (torch.float16, torch.bfloat16, torch.float64):
                    with self.subTest(default_dtype=dtype):
                        torch.set_default_dtype(dtype)
                        check(source)
            finally:
                torch.set_default_dtype(previous)
            # Only platform detection is isolated. Torch planning runs on the
            # actual CUDA tensor; this is not a claim of ROCm hardware coverage.
            with patch("rtp_llm.device.device_type.is_cuda", return_value=False):
                check(source)

    def test_interleaved_calls_and_nondefault_stream_are_stable(self):
        cases = [
            (history("zipf", steps, layers, experts, 1).cuda(), physical, local)
            for steps in (100, 1000)
            for _, layers, experts, physical, local in GROUPS
        ]
        expected = [
            checked_plan(source, physical, local)[0]
            for source, physical, local in cases
        ]
        originals = [source.clone() for source, _, _ in cases]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for (source, physical, local), oracle in zip(cases, expected):
                assert_same_outputs(
                    policy.rebalance_experts_global(source, physical, local), oracle
                )
        stream.synchronize()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(20):
            for (source, physical, local), oracle, original in zip(
                cases, expected, originals
            ):
                result = policy.rebalance_experts_global(source, physical, local)
                assert_same_outputs(result, oracle)
                self.assertTrue(torch.equal(source, original))
                del result
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), allocated)


def load_tests(loader, tests, pattern):
    device = os.environ.get("EPLB_POLICY_TEST_DEVICE", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError(device)
    return loader.loadTestsFromTestCase(
        PolicyLargeGpuTest if device == "cuda" else PolicyLargeCpuTest
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
