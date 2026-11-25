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
    def test_upstream_sampling_core_preserved(self) -> None:
        root = reference_root()
        original = (root / "sample/gumbel.py").read_text()
        local = Path(vllm_gumbel.__file__)

        def sampling_core(text: str) -> list[str]:
            # The no-cache launch is covered by the numerical token comparison.
            launch_or_omitted = {
                "_temperature_kernel",
                "apply_temperature",
                "spec_step_seed",
                "gumbel_block_argmax",
                "_gumbel_sample_kernel",
                "gumbel_sample",
            }
            core = []
            for node in ast.parse(text).body:
                if isinstance(node, ast.FunctionDef):
                    if node.name in launch_or_omitted:
                        continue
                    if node.name == "gumbel_noised_argmax" and node.args.args[-1].arg == "RETURN_KEY":
                        # Compare the default dense return with the pinned implementation.
                        node.args.args.pop()
                        self.assertIs(ast.literal_eval(node.args.defaults.pop()), False)
                        branch = node.body[-1]
                        assert isinstance(branch, ast.If)
                        self.assertEqual(ast.unparse(branch.test), "RETURN_KEY")
                        node.body[-1:] = branch.orelse
                    core.append(ast.dump(node))
                elif isinstance(node, ast.Assign):
                    core.append(ast.dump(node))
            return core

        self.assertEqual(sampling_core(local.read_text()), sampling_core(original))
        self.assertEqual(
            ast.dump(draft_function(local.with_name("draft_logits.py"))),
            ast.dump(draft_function(root / "spec_decode/draft_logits.py")),
        )


@triton.jit
def _compact_winner(scores, keys, output, temp, seed, pos):
    lane = tl.arange(0, 4)
    token_ids = tl.load(keys + lane)
    values = tl.load(scores + lane)
    _, token = vllm_gumbel.gumbel_noised_argmax(
        values, token_ids, token_ids >= 0, seed, pos, temp,
        IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False, RETURN_KEY=True,
    )
    tl.store(output, token)


@unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA unavailable")
class GumbelTest(unittest.TestCase):
    def test_compact_winner_matches_original_vocab_after_permutation(self) -> None:
        ids = torch.tensor([191, 33, 290, -1], device="cuda", dtype=torch.int32)
        scores = torch.tensor([4.0, 4.0, 3.0, -torch.inf], device="cuda")
        dense = torch.full((1, 512), -torch.inf, device="cuda")
        dense[0, ids[:3].long()] = scores[:3]
        seed = torch.tensor([11], device="cuda")
        position = torch.tensor([5], device="cuda")
        mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
        result = torch.empty_like(seed)
        for temp in (0.0, 1.0):
            expected = vllm_gumbel.gumbel_sample(
                dense, mapping, torch.tensor([temp], device="cuda"), seed, position, False, False
            )
            self.assertEqual(expected.item(), 33 if temp == 0.0 else 290)
            for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
                _compact_winner[(1,)](scores[order], ids[order], result, temp, 11, 5)
                self.assertTrue(torch.equal(result, expected))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
