"""Compare production dispatch with the frozen pre-fusion RTP policy.

The GPU target replays all 249 random/adversarial and 354 EP/redundancy cases
from DeepInfer's prototype, then 24 captured service windows. The oracle is
RTP commit 9576870171e8be83688f820e628d16bcbdc67fc9, not the prototype.
The CPU target verifies fallback contracts without requiring Triton or CUDA.
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.eplb import policy
from rtp_llm.eplb.test import policy_reference

KINDS = ("zeros", "const", "sparse", "uniform", "large", "mixed")
REFERENCE_SHA256 = "56b5ee082422c3ed526357a0393d731e2d3970185501419291eaf5132ae82f3a"
REPLAY_SOURCE_SHA256 = (
    "f4d4d9447a86fc53949175271e7ded3b20d75e422d072fbd2e6ce88189a77151"
)


def child_environment():
    # Bazel adds runfile dependencies to sys.path in its launcher. A child
    # launched with sys.executable needs the same dependency closure.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    return environment


def history(kind, steps, layers, experts, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (steps, layers, experts)
    if kind == "zeros":
        result = torch.zeros(shape, dtype=torch.int32)
    elif kind == "const":
        result = torch.full(shape, 137, dtype=torch.int32)
    elif kind == "sparse":
        result = torch.zeros(shape, dtype=torch.int32)
        hot = torch.rand(shape, generator=generator) < 0.05
        result[hot] = torch.randint(
            1, 5000, (int(hot.sum()),), generator=generator, dtype=torch.int32
        )
    elif kind == "uniform":
        result = torch.randint(0, 1000, shape, generator=generator, dtype=torch.int32)
    elif kind == "large":
        result = torch.randint(
            1 << 28, 1 << 30, shape, generator=generator, dtype=torch.int32
        )
    elif kind == "int32_max":
        result = torch.full(shape, 2**31 - 1, dtype=torch.int32)
        result[..., 0] = 0
        result[0, :, 1] -= 1
    elif kind == "mixed":
        base = torch.randint(0, 100, shape, generator=generator)
        spike = (torch.rand(shape, generator=generator) < 0.01).to(
            torch.int64
        ) * torch.randint(1, 10_000_000, shape, generator=generator)
        result = (base + spike).clamp(0, 2**31 - 1).to(torch.int32)
    else:
        raise ValueError(kind)
    return result.to(device)


def parity_cases():
    # name, steps, layers, logical, physical, local, kinds, seed count
    for name, steps, layers, experts, physical, local, kinds, seeds in (
        ("main", 4, 40, 384, 416, 52, KINDS, 10),
        ("dspark", 4, 3, 128, 160, 20, KINDS, 10),
        ("odd", 4, 5, 10, 24, 4, KINDS, 10),
        ("odd2", 2, 7, 6, 9, 3, KINDS, 10),
        ("s1", 1, 40, 384, 416, 52, ("sparse", "uniform", "large"), 3),
    ):
        for kind in kinds:
            for seed in range(seeds):
                yield (
                    f"{name}/{kind}/{seed}",
                    (kind, steps, layers, experts, seed),
                    physical,
                    local,
                )


def sweep_cases():
    for ep in (4, 8, 16, 32, 64, 72):
        minimum = 4 * ep
        levels = set(range(1, min(16, minimum) + 1))
        levels.update((ep, 2 * ep, 3 * ep, minimum // 2, minimum - 1, minimum))
        for redundant in sorted(levels):
            if 1 <= redundant <= minimum:
                physical = -(-(minimum + redundant) // ep) * ep
                for kind in ("sparse", "uniform", "large"):
                    # At the endpoint R=E, total capacity is 2E; individual
                    # experts need not receive exactly two replicas.
                    yield (
                        f"ep{ep}/r{redundant}/{kind}",
                        (kind, 4, 3, physical - redundant, ep * 1000 + redundant),
                        physical,
                        physical // ep,
                    )


CANONICAL = (
    ("main_sparse", ("sparse", 4, 40, 384, 42), 416, 52),
    ("main_large", ("large", 4, 40, 384, 7), 416, 52),
    ("dspark_sparse", ("sparse", 4, 3, 128, 11), 160, 20),
    ("division_rounding", ("sparse", 4, 3, 128, 32128), 256, 8),
)


class PolicyParityAssertions:
    def assert_outputs(self, actual, expected, source, physical):
        _, layers, experts = source.shape
        shapes = (
            (layers, physical),
            (layers, experts, physical - experts + 1),
            (layers, experts),
        )
        self.assertEqual(len(actual), 3)
        for name, value, oracle, shape in zip(
            ("physical_map", "inverse_map", "replica_counts"), actual, expected, shapes
        ):
            self.assertEqual(value.dtype, torch.int32, name)
            self.assertEqual(value.device, source.device, name)
            self.assertEqual(tuple(value.shape), shape, name)
            self.assertTrue(torch.equal(value, oracle), name)

    def assert_parity(self, source, physical, local):
        original = source.clone()
        metadata = (source.stride(), source.data_ptr(), source.storage_offset())
        expected = policy_reference.rebalance_experts_global(source, physical, local)
        self.assertTrue(torch.equal(source, original), "reference mutated history")
        actual = policy.rebalance_experts_global(source, physical, local)
        self.assert_outputs(actual, expected, source, physical)
        self.assertTrue(torch.equal(source, original), "production mutated history")
        self.assertEqual(
            metadata, (source.stride(), source.data_ptr(), source.storage_offset())
        )
        return actual

    def check_fallbacks(self, device):
        if torch.device(device).type == "cpu":
            for steps in (3, 5, 8, 16):
                with self.subTest(steps=steps):
                    self.assert_parity(history("mixed", steps, 2, 6, 7, device), 9, 3)
        for dtype in (torch.int64, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = history("mixed", 4, 2, 6, 8, device).to(dtype)
                if dtype.is_floating_point:
                    source = source + 0.125
                self.assert_parity(source, 9, 3)
        contiguous = history("mixed", 4, 3, 12, 9, device)
        for source in (contiguous[..., ::2], contiguous.permute(0, 2, 1)):
            with self.subTest(stride=source.stride()):
                self.assertFalse(source.is_contiguous())
                self.assert_parity(source, 12, 3)
        previous = torch.get_default_dtype()
        try:
            for dtype in (torch.float16, torch.bfloat16, torch.float64):
                with self.subTest(default_dtype=dtype):
                    torch.set_default_dtype(dtype)
                    self.assert_parity(history("uniform", 4, 2, 6, 10, device), 9, 3)
        finally:
            torch.set_default_dtype(previous)

    def check_boundaries(self, device):
        for steps, experts, physical, local in (
            (1, 1, 1, 1),  # no redundant slots, no placement
            (2, 1, 4, 2),  # single logical expert
            (4, 2, 8, 2),  # more spare slots than logical experts
            (4, 6, 6, 3),  # placement without redundancy
            (2, 6, 9, 1),  # redundancy without placement
            (1, 5, 9, 9),  # one rank
        ):
            for kind in ("zeros", "const", "large"):
                with self.subTest(
                    steps=steps,
                    experts=experts,
                    physical=physical,
                    local=local,
                    kind=kind,
                ):
                    self.assert_parity(
                        history(kind, steps, 2, experts, 11, device), physical, local
                    )


class PolicyFallbackTest(PolicyParityAssertions, unittest.TestCase):
    def test_reference_is_frozen_rtp_source(self):
        self.assertEqual(
            hashlib.sha256(Path(policy_reference.__file__).read_bytes()).hexdigest(),
            REFERENCE_SHA256,
        )
        self.assertEqual(len(list(parity_cases())), 249)
        self.assertEqual(len(list(sweep_cases())), 354)

    def test_cpu_boundary_inputs_match_reference(self):
        self.check_boundaries("cpu")

    def test_cpu_dtype_window_layout_fallbacks(self):
        self.check_fallbacks("cpu")

    def test_cpu_dispatch_needs_no_fused_dependency_or_cuda_context(self):
        # Run before importing production in a fresh interpreter: an already
        # imported fused module would hide an accidental eager dependency.
        program = """
import builtins
import torch
torch.set_num_threads(1)
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if (name == 'triton' or name.startswith('triton.')
        or name == 'rtp_llm.eplb.policy_fused'
        or (name == 'rtp_llm.eplb' and 'policy_fused' in fromlist)):
        raise AssertionError('CPU policy imported ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
from rtp_llm.eplb.policy import rebalance_experts_global
from rtp_llm.eplb.test.policy_reference import rebalance_experts_global as reference
source = torch.tensor([[[4, 39, 13, 0]], [[3, 19, 27, 23]]], dtype=torch.int32)
actual = rebalance_experts_global(source, 8, 2)
assert all(torch.equal(a, b) for a, b in zip(actual, reference(source, 8, 2)))
assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=60,
            env=child_environment(),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class PolicyFusedGpuTest(PolicyParityAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.device.device_type import is_cuda
        from rtp_llm.eplb import policy_fused

        # An explicitly selected GPU target must fail, not silently skip.
        if not is_cuda() or not torch.cuda.is_available():
            raise RuntimeError("The fused policy target requires NVIDIA CUDA")
        cls.fused_module = policy_fused

    def observed_fused(self):
        # Observe the real function; all kernels and outputs execute unchanged.
        return patch.object(
            self.fused_module,
            "rebalance_experts_global_fused",
            wraps=self.fused_module.rebalance_experts_global_fused,
        )

    def run_cases(self, cases, count):
        with self.observed_fused() as fused:
            completed = 0
            for name, arguments, physical, local in cases:
                with self.subTest(case=name):
                    self.assert_parity(
                        history(*arguments, device="cuda"), physical, local
                    )
                    completed += 1
            self.assertEqual(completed, count)
            self.assertEqual(
                fused.call_count, count, "eligible inputs bypassed fused dispatch"
            )

    def test_249_random_and_adversarial_cases(self):
        self.run_cases(parity_cases(), 249)

    def test_354_ep_and_redundancy_cases(self):
        self.run_cases(sweep_cases(), 354)

    def test_24_real_service_windows(self):
        replay = json.loads(Path(__file__).with_name("policy_replay.json").read_text())
        self.assertEqual(replay["source_fixture_sha256"], REPLAY_SOURCE_SHA256)
        self.assertEqual(replay["source_sequences"], list(range(19, 31)))
        self.assertEqual([g["name"] for g in replay["groups"]], ["backbone", "draft"])
        completed = 0
        with self.observed_fused() as fused:
            for group in replay["groups"]:
                dims = group["dimensions"]
                self.assertEqual(len(group["logical_windows"]), 12)
                for index, window in enumerate(group["logical_windows"]):
                    with self.subTest(
                        group=group["name"], sequence=replay["source_sequences"][index]
                    ):
                        source = torch.tensor(window, dtype=torch.int32, device="cuda")
                        self.assertEqual(
                            tuple(source.shape),
                            (dims["window"], dims["layers"], dims["logical"]),
                        )
                        actual = self.assert_parity(
                            source, dims["physical"], dims["physical"] // 8
                        )
                        for output, key in (
                            (actual[0], "expected_physical_to_logical_maps"),
                            (actual[2], "expected_logical_counts"),
                        ):
                            recorded = torch.tensor(
                                group[key][index], dtype=torch.int32, device="cuda"
                            )
                            self.assertTrue(torch.equal(output, recorded), key)
                        completed += 1
            self.assertEqual(completed, 24)
            self.assertEqual(fused.call_count, 24)

    def test_boundary_inputs_use_fused(self):
        with self.observed_fused() as fused:
            self.check_boundaries("cuda")
            self.assertEqual(fused.call_count, 18)

    def test_dtype_layout_fallbacks(self):
        with patch.object(
            self.fused_module,
            "rebalance_experts_global_fused",
            side_effect=AssertionError("unsupported input entered fused policy"),
        ):
            self.check_fallbacks("cuda")

    def test_platform_filter_keeps_torch_policy(self):
        # Isolate only platform detection; actual CUDA Torch planning still
        # executes. This tests the route, not hardware coverage for ROCm.
        with patch(
            "rtp_llm.device.device_type.is_cuda", return_value=False
        ), patch.object(
            self.fused_module,
            "rebalance_experts_global_fused",
            side_effect=AssertionError("non-NVIDIA platform entered fused policy"),
        ):
            self.assert_parity(history("mixed", 4, 2, 6, 17, "cuda"), 9, 3)

    def test_large_ep_padding_and_int32_max_counts(self):
        cases = (
            (
                f"ep{ep}/{kind}",
                (kind, 4, 2, 2 * ep - 3, 43),
                2 * ep,
                2,
            )
            for ep in (128, 129, 256)
            for kind in ("uniform", "large", "int32_max")
        )
        self.run_cases(cases, 9)

    def test_round_to_nearest_division_regression(self):
        # Approximate Triton '/' changes a replica decision at round 118.
        self.run_cases((CANONICAL[3],), 1)

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), self.observed_fused() as fused:
            self.assert_parity(history("mixed", 4, 3, 128, 32, "cuda"), 160, 20)
            self.assertEqual(fused.call_count, 1)
        stream.synchronize()

    def test_repeated_interleaved_calls_do_not_retain_gpu_allocations(self):
        cases = []
        for _, arguments, physical, local in CANONICAL:
            source = history(*arguments, device="cuda")
            expected = policy_reference.rebalance_experts_global(
                source, physical, local
            )
            cases.append((source, source.clone(), physical, local, expected))

        def run_block():
            # Alternate shapes and values, repeating each original input after
            # other inputs to expose cached-output or scratch-state reuse.
            for _ in range(8):
                for source, original, physical, local, expected in cases:
                    actual = policy.rebalance_experts_global(source, physical, local)
                    self.assert_outputs(actual, expected, source, physical)
                    self.assertTrue(torch.equal(source, original))

        with self.observed_fused() as fused:
            run_block()  # Compile every shape and populate allocator caches.
            gc.collect()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            for block in range(4):
                run_block()
                gc.collect()
                torch.cuda.synchronize()
                self.assertEqual(
                    torch.cuda.memory_allocated(),
                    allocated,
                    f"live tensor memory grew after block {block}",
                )
            self.assertEqual(fused.call_count, 160)
        # Reserved allocator/JIT memory is intentionally not a leak metric;
        # no empty_cache() masks growth in live tensors between the blocks.

    def test_two_fresh_processes_are_deterministic(self):
        expected = canonical_hashes()
        for run in range(2):
            with self.subTest(process=run):
                result = subprocess.run(
                    [sys.executable, __file__, "--determinism-worker"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=child_environment(),
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    json.loads(result.stdout.strip().splitlines()[-1]), expected
                )


def canonical_hashes():
    case = PolicyFusedGpuTest()
    case.setUpClass()
    hashes = {}
    with case.observed_fused() as fused:
        for name, arguments, physical, local in CANONICAL:
            result = case.assert_parity(
                history(*arguments, device="cuda"), physical, local
            )
            hashes[name] = [
                hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
                for t in result
            ]
        case.assertEqual(fused.call_count, len(CANONICAL))
    return hashes


def load_tests(loader, tests, pattern):
    selected = os.environ.get("EPLB_POLICY_TEST_DEVICE", "cpu")
    if selected not in ("cpu", "cuda"):
        raise ValueError(f"Unknown policy test device: {selected}")
    return loader.loadTestsFromTestCase(
        PolicyFusedGpuTest if selected == "cuda" else PolicyFallbackTest
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    if sys.argv[1:] == ["--determinism-worker"]:
        print(json.dumps(canonical_hashes(), sort_keys=True))
    else:
        unittest.main()
