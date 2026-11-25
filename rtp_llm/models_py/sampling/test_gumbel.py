"""Set VLLM_GUMBEL_REFERENCE_DIR to a pinned vLLM checkout for exact comparisons."""

import ast
import importlib.util
import os
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.sampling import adapter, vllm_gumbel


def reference_root() -> Path:
    root = os.environ.get("VLLM_GUMBEL_REFERENCE_DIR")
    if not root:
        raise unittest.SkipTest("NOT_RUN: VLLM_GUMBEL_REFERENCE_DIR is unset")
    return Path(root) / "vllm/v1/worker/gpu"


def draft_function(path: Path) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "draft_logits_row"
    )


@lru_cache(maxsize=1)
def reference_module() -> ModuleType:
    """Load upstream code unchanged, shimming only its two import dependencies."""
    root = reference_root()
    draft_path = root / "spec_decode/draft_logits.py"
    draft = ModuleType("_gumbel_reference_draft")
    draft.__dict__.update(triton=triton, tl=tl)
    tree = ast.Module(body=[draft_function(draft_path)], type_ignores=[])
    exec(compile(tree, str(draft_path), "exec"), draft.__dict__)
    triton_utils = ModuleType("_gumbel_reference_triton_utils")
    triton_utils.__dict__.update(triton=triton, tl=tl)
    spec = importlib.util.spec_from_file_location(
        "_gumbel_upstream_reference", root / "sample/gumbel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "vllm.triton_utils": triton_utils,
            "vllm.v1.worker.gpu.spec_decode.draft_logits": draft,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class GumbelSourceTest(unittest.TestCase):
    def test_upstream_source_preserved(self) -> None:
        root = reference_root()
        original = (root / "sample/gumbel.py").read_text()
        expected = original.replace(
            "from vllm.triton_utils import tl, triton",
            "import triton\nimport triton.language as tl",
        ).replace(
            "from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row",
            "from rtp_llm.models_py.sampling.draft_logits import draft_logits_row",
        )
        local = Path(vllm_gumbel.__file__)
        self.assertEqual(local.read_text(), expected)
        self.assertEqual(
            ast.dump(draft_function(local.with_name("draft_logits.py"))),
            ast.dump(draft_function(root / "spec_decode/draft_logits.py")),
        )


@unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA unavailable")
class GumbelTest(unittest.TestCase):
    def test_exact_upstream_fp32_and_fp64(self) -> None:
        upstream = reference_module()
        for batch, vocab in [(1, 1), (7, 257), (32, 4097), (4, 154883)]:
            logits = torch.randn(batch, vocab, device="cuda")
            mapping = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0)
            seeds = torch.arange(batch, device="cuda", dtype=torch.int64) + (1 << 40)
            positions = torch.arange(batch, device="cuda", dtype=torch.int64) * 17 + 123
            temperatures = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cuda").repeat(batch)[:batch]
            for fp64 in [False, True]:
                for drafting in [False, True]:
                    with self.subTest(batch=batch, vocab=vocab, fp64=fp64, drafting=drafting):
                        args = (logits, mapping, temperatures, seeds, positions, True, drafting)
                        expected = upstream.gumbel_sample(*args, use_fp64=fp64)
                        actual = vllm_gumbel.gumbel_sample(*args, use_fp64=fp64)
                        self.assertTrue(torch.equal(actual, expected))

    def test_adapter_exact_upstream_with_preapplied_temperature(self) -> None:
        upstream = reference_module()
        logits = torch.randn(32, 4097, device="cuda") / 0.7
        seeds = torch.arange(32, device="cuda", dtype=torch.int64) + 12345
        positions = seeds * 3
        expected = upstream.gumbel_sample(
            logits, torch.arange(32, device="cuda", dtype=torch.int32),
            torch.ones(32, device="cuda"), seeds, positions, False, False,
        )
        actual = adapter.sample(logits, seeds, positions)
        self.assertTrue(torch.equal(actual, expected))

    def test_temperature_helper_matches_upstream(self) -> None:
        upstream = reference_module()
        logits = torch.randn(4, 8193, device="cuda")
        expected = logits.clone()
        mapping = torch.arange(4, device="cuda", dtype=torch.int32)
        temperatures = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cuda")
        upstream.apply_temperature(expected, mapping, temperatures)
        vllm_gumbel.apply_temperature(logits, mapping, temperatures)
        self.assertTrue(torch.equal(logits, expected))

    def test_greedy_temperature_zero(self) -> None:
        logits = torch.randn(32, 4097, device="cuda")
        mapping = torch.arange(32, device="cuda", dtype=torch.int32)
        seeds = torch.arange(32, device="cuda", dtype=torch.int64)
        actual = vllm_gumbel.gumbel_sample(
            logits, mapping, torch.zeros(32, device="cuda"), seeds, seeds, True, False,
        )
        self.assertTrue(torch.equal(actual, logits.argmax(-1)))

    def test_masked_nonaligned_vocab(self) -> None:
        seeds = torch.arange(3, device="cuda", dtype=torch.int64)
        for vocab in [1, 257, 4097, 154883]:
            with self.subTest(vocab=vocab):
                expected = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
                logits = torch.full((3, vocab), -torch.inf, device="cuda")
                logits[torch.arange(3, device="cuda"), expected] = 1.0
                actual = adapter.sample(logits, seeds, seeds)
                self.assertTrue(torch.equal(actual, expected))

    def test_seed_position_reproducibility(self) -> None:
        logits = torch.zeros(256, 257, device="cuda")
        seeds = torch.arange(256, device="cuda", dtype=torch.int64) + 1234
        positions = torch.arange(256, device="cuda", dtype=torch.int64)
        first = adapter.sample(logits, seeds, positions)
        self.assertTrue(torch.equal(first, adapter.sample(logits, seeds, positions)))
        self.assertFalse(torch.equal(first, adapter.sample(logits, seeds, positions + 1)))
        self.assertFalse(torch.equal(first, adapter.sample(logits, seeds + 9999, positions)))

    def test_categorical_distribution(self) -> None:
        count = 16384
        logits = torch.tensor([6.0, 3.0, 1.0], device="cuda").log().repeat(count, 1)
        varying = torch.arange(count, device="cuda", dtype=torch.int64)
        fixed = torch.full_like(varying, 12345)
        for seeds, positions in [(varying, fixed), (fixed, varying)]:
            actual = adapter.sample(logits, seeds, positions)
            self.assertTrue(bool(((actual >= 0) & (actual < 3)).all()))
            counts = torch.bincount(actual, minlength=3).cpu().tolist()
            for observed, probability in zip(counts, [0.6, 0.3, 0.1]):
                sigma = (count * probability * (1 - probability)) ** 0.5
                self.assertLessEqual(abs(observed - count * probability), 6 * sigma)

    @unittest.skipUnless(os.environ.get("VLLM_GUMBEL_BENCH") == "1", "NOT_RUN: opt-in benchmark")
    def test_benchmark_torch_multinomial_baseline(self) -> None:
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.float32)
        for batch in [1, 32, 64, 128]:
            logits = torch.randn(batch, 154880, device="cuda")
            seeds = torch.arange(batch, device="cuda", dtype=torch.int64)
            timings = []
            for operation in [
                lambda: torch.multinomial(torch.softmax(logits, -1), 1),
                lambda: adapter.sample(logits, seeds, seeds),
            ]:
                for _ in range(10):
                    operation()
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                elapsed = 0.0
                for _ in range(50):
                    flush.zero_()
                    start.record()
                    output = operation()
                    stop.record()
                    stop.synchronize()
                    elapsed += start.elapsed_time(stop) * 1000
                self.assertTrue(bool(((output >= 0) & (output < 154880)).all()))
                timings.append(elapsed / 50)
            print(
                f"[gumbel-benchmark] batch={batch} vocab=154880 "
                f"torch_softmax_multinomial_us={timings[0]:.3f} gumbel_us={timings[1]:.3f} "
                "warmup=10 iterations=50 l2_flush_bytes=536870912 flush_timing=excluded "
                "baseline_is_rtp_flashinfer=false", flush=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
