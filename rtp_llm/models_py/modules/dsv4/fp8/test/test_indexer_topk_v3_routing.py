import ast
import os
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict
from unittest import mock

import torch


def _source_path() -> Path:
    relative = "rtp_llm/models_py/modules/dsv4/fp8/indexer.py"
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        root = Path(runfiles)
        candidates = [root / relative]
        workspace = os.environ.get("TEST_WORKSPACE")
        if workspace:
            candidates.append(root / workspace / relative)
        candidates.extend(root.glob(f"*/{relative}"))
        matches = list(dict.fromkeys(path for path in candidates if path.exists()))
        if len(matches) != 1:
            raise RuntimeError(f"expected one indexer runfile, found {matches}")
        return matches[0]
    return Path(__file__).parents[1] / "indexer.py"


def _load_source_only_indexer() -> ModuleType:
    tree = ast.parse(_source_path().read_text())
    assignment_names = {
        "_FAST_PREFILL_TOPK_OK",
        "_FAST_PREFILL_TOPK_MAX_INPUT_TOKENS",
        "_DECODE_TOPK_CANDIDATES",
        "_MISSING_DECODE_TOPK_CANDIDATE",
        "_DECODE_TOPK_WORKSPACE_SIZE",
        "_decode_topk_workspace_cache",
    }
    function_names = {
        "_fp8_prefill_fast_topk_enabled",
        "_fp8_prefill_topk_force_radix_sort",
        "_fp8_prefill_topk_use_torch",
        "_fp8_prefill_topk_canonicalize",
        "_run_prefill_topk_torch",
        "_run_prefill_topk",
        "_topk_v3_enabled",
        "_decode_topk_capture_active",
        "_get_decode_topk_workspace",
        "_run_decode_topk",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & assignment_names:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in function_names:
                selected.append(node)
    module = ModuleType("dsv4_indexer_topk_source_only")
    module.__dict__.update(
        Dict=Dict,
        os=os,
        rtp_llm_ops=SimpleNamespace(),
        torch=torch,
    )
    code = compile(
        ast.Module(body=selected, type_ignores=[]), str(_source_path()), "exec"
    )
    exec(code, module.__dict__)
    missing = (assignment_names | function_names) - set(module.__dict__)
    if missing:
        raise RuntimeError(f"missing extracted indexer symbols: {sorted(missing)}")
    return module


if os.environ.get("DSV4_INDEXER_TOPK_SOURCE_ONLY") == "1":
    indexer = _load_source_only_indexer()
else:
    from rtp_llm.models_py.modules.dsv4.fp8 import indexer


def _canonicalize_ties(logits: torch.Tensor) -> torch.Tensor:
    """Give equal values a deterministic lower-index-first ordering."""
    width = logits.shape[1]
    offsets = torch.arange(width, dtype=torch.float64).mul_(-1.0e-12)
    return logits.to(torch.float64) + offsets


def _oracle(logits: torch.Tensor, lengths: torch.Tensor, k: int) -> list[set[int]]:
    canonical = _canonicalize_ties(logits)
    expected = []
    for row, length in enumerate(lengths.tolist()):
        keep = min(k, length)
        indices = canonical[row, :length].topk(keep, sorted=False).indices
        expected.append(set(indices.tolist()))
    return expected


def _assert_output_contract(
    test: unittest.TestCase,
    output: torch.Tensor,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> None:
    test.assertEqual(output.shape, (logits.shape[0], k))
    test.assertEqual(output.dtype, torch.int32)
    expected = _oracle(logits, lengths, k)
    for row, length in enumerate(lengths.tolist()):
        keep = min(k, length)
        valid = output[row, :keep]
        padding = output[row, keep:]
        test.assertTrue(bool(((valid >= 0) & (valid < length)).all()))
        test.assertEqual(valid.unique().numel(), keep)
        test.assertEqual(set(valid.tolist()), expected[row])
        test.assertTrue(bool((padding == -1).all()))


class _UnorderedTopK:
    """CPU ABI double with native set semantics and deliberately reversed order."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, logits, lengths, output, workspace, k, max_seq_len):
        self.calls.append((logits, lengths, output, workspace, k, max_seq_len))
        assert logits.ndim == 2 and logits.dtype == torch.float32
        assert lengths.ndim == 1 and lengths.dtype == torch.int32
        assert output.shape == (logits.shape[0], k) and output.dtype == torch.int32
        assert workspace.dtype == torch.uint8 and workspace.is_contiguous()
        assert max_seq_len == logits.shape[1]
        output.fill_(-1)
        canonical = _canonicalize_ties(logits)
        for row, length in enumerate(lengths.tolist()):
            keep = min(k, length)
            selected = canonical[row, :length].topk(keep, sorted=False).indices
            output[row, :keep].copy_(selected.flip(0).to(torch.int32))


class IndexerPrefillTopKOrderingTest(unittest.TestCase):
    def test_native_backends_canonicalize_without_changing_selected_set(self):
        logits = torch.zeros((3, 1024), dtype=torch.float32)
        starts = torch.tensor([0, 11, 30], dtype=torch.int32)
        ends = torch.tensor([700, 14, 30], dtype=torch.int32)
        unordered = torch.full((3, 512), -1, dtype=torch.int32)
        unordered[0] = torch.arange(600, 88, -1, dtype=torch.int32)
        unordered[1, :3] = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected = unordered.clone()
        expected[0] = torch.arange(89, 601, dtype=torch.int32)
        expected[1, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
        for fast in (True, False):
            for canonicalize in ("0", "1"):
                with self.subTest(fast=fast, canonicalize=canonicalize):
                    fast_op = mock.Mock(side_effect=lambda *args: args[1].copy_(unordered))
                    general_op = mock.Mock(side_effect=lambda *args: args[3].copy_(unordered))
                    out = torch.empty_like(unordered)
                    with (
                        mock.patch.object(indexer, "_FAST_PREFILL_TOPK_OK", True),
                        mock.patch.object(indexer, "rtp_llm_ops", SimpleNamespace(
                            fast_topk_v2_variable=fast_op,
                            dsv4_top_k_per_row_prefill=general_op)),
                        mock.patch.dict(os.environ, {
                            "DSV4_INDEXER_TOPK_BACKEND": "auto",
                            "DSV4_PREFILL_FAST_TOPK": "1" if fast else "0",
                            "DSV4_INDEXER_TOPK_CANONICALIZE": canonicalize,
                        }),
                    ):
                        indexer._run_prefill_topk(logits, starts, ends, out, 512, 4)
                    self.assertTrue(torch.equal(out, expected if canonicalize == "1" else unordered))
                    self.assertEqual(fast_op.call_count, int(fast))
                    self.assertEqual(general_op.call_count, int(not fast))
                    self.assertTrue(torch.equal(out.sort().values, unordered.sort().values))

    def test_torch_canonicalization_keeps_relative_indices_and_padding(self):
        logits = torch.tensor([[99., 98., 3., 1., 2., 97.], [99., 98., 3., 1., 2., 97.]])
        starts = torch.tensor([2, 4], dtype=torch.int32)
        ends = torch.tensor([5, 4], dtype=torch.int32)
        out = torch.empty((2, 8), dtype=torch.int32)
        with mock.patch.dict(os.environ, {
            "DSV4_INDEXER_TOPK_BACKEND": "torch",
            "DSV4_INDEXER_TOPK_CANONICALIZE": "1",
        }):
            indexer._run_prefill_topk(logits, starts, ends, out, 8, 4)
        expected = torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1], [-1] * 8], dtype=torch.int32)
        self.assertTrue(torch.equal(out, expected))


class IndexerDecodeTopKRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        indexer._decode_topk_workspace_cache.clear()

    def _run(self, ops, logits, lengths, output, k, max_seq_len):
        with (
            mock.patch.object(indexer, "rtp_llm_ops", ops),
            mock.patch.dict(os.environ, {"DSV4_TOPK_V3": "1"}),
        ):
            return indexer._run_decode_topk(
                logits, lengths, output, k, max_seq_len
            )

    def test_candidate_priority_prefers_topk_v3(self) -> None:
        preferred = mock.Mock()
        secondary = mock.Mock()
        logits = torch.empty((3, 2048), dtype=torch.float32)
        lengths = torch.tensor([2048, 1024, 512], dtype=torch.int32)
        output = torch.empty((3, 512), dtype=torch.int32)

        self.assertTrue(
            self._run(
                SimpleNamespace(
                    topk_v3=preferred, dsv4_persistent_topk=secondary
                ),
                logits,
                lengths,
                output,
                512,
                2048,
            )
        )

        preferred.assert_called_once_with(
            logits,
            lengths,
            output,
            indexer._decode_topk_workspace_cache[logits.device],
            512,
            2048,
        )
        secondary.assert_not_called()

    def test_topk_v3_debug_disable_still_uses_secondary(self) -> None:
        preferred = mock.Mock()
        secondary = mock.Mock()
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([512], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        with (
            mock.patch.object(
                indexer,
                "rtp_llm_ops",
                SimpleNamespace(
                    topk_v3=preferred, dsv4_persistent_topk=secondary
                ),
            ),
            mock.patch.dict(os.environ, {"DSV4_TOPK_V3": "0"}),
        ):
            self.assertTrue(
                indexer._run_decode_topk(tensor, lengths, output, 512, 512)
            )

        preferred.assert_not_called()
        secondary.assert_called_once()

    def test_missing_symbols_signal_existing_torch_fallback(self) -> None:
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([512], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)
        self.assertFalse(
            self._run(SimpleNamespace(), tensor, lengths, output, 512, 512)
        )
        self.assertEqual(indexer._decode_topk_workspace_cache, {})

    def test_legacy_persistent_topk_is_not_a_decode_candidate(self) -> None:
        legacy = mock.Mock()
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([511], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        self.assertFalse(
            self._run(
                SimpleNamespace(persistent_topk=legacy),
                tensor,
                lengths,
                output,
                512,
                512,
            )
        )
        legacy.assert_not_called()
        self.assertEqual(indexer._decode_topk_workspace_cache, {})

    def test_late_binding_availability_is_observed(self) -> None:
        ops = SimpleNamespace()
        candidate = mock.Mock()
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([512], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        self.assertFalse(self._run(ops, tensor, lengths, output, 512, 512))
        ops.dsv4_persistent_topk = candidate
        self.assertTrue(self._run(ops, tensor, lengths, output, 512, 512))
        candidate.assert_called_once()

    def test_unsupported_k_signals_existing_torch_fallback(self) -> None:
        candidate = mock.Mock()
        tensor = torch.empty((1, 256), dtype=torch.float32)
        lengths = torch.tensor([256], dtype=torch.int32)
        output = torch.empty((1, 256), dtype=torch.int32)
        self.assertFalse(
            self._run(
                SimpleNamespace(topk_v3=candidate),
                tensor,
                lengths,
                output,
                256,
                256,
            )
        )
        candidate.assert_not_called()

    def test_noncallable_wrong_abi_and_runtime_errors_fail_fast(self) -> None:
        secondary = mock.Mock()
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([512], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        with self.assertRaisesRegex(TypeError, "not callable"):
            self._run(
                SimpleNamespace(topk_v3=object(), dsv4_persistent_topk=secondary),
                tensor,
                lengths,
                output,
                512,
                512,
            )
        secondary.assert_not_called()

        with self.assertRaises(TypeError):
            self._run(
                SimpleNamespace(
                    topk_v3=lambda only_one_argument: None,
                    dsv4_persistent_topk=secondary,
                ),
                tensor,
                lengths,
                output,
                512,
                512,
            )
        secondary.assert_not_called()

        primary = mock.Mock(side_effect=RuntimeError("kernel failure"))
        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            self._run(
                SimpleNamespace(
                    topk_v3=primary, dsv4_persistent_topk=secondary
                ),
                tensor,
                lengths,
                output,
                512,
                512,
            )
        secondary.assert_not_called()

    def test_workspace_is_reused_and_capture_requires_warmup(self) -> None:
        candidate = _UnorderedTopK()
        ops = SimpleNamespace(dsv4_persistent_topk=candidate)
        logits = torch.randn((1, 768), dtype=torch.float32)
        lengths = torch.tensor([700], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        with mock.patch.object(
            indexer, "_decode_topk_capture_active", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "warmed before graph capture"):
                self._run(ops, logits, lengths, output, 512, 768)
        self.assertEqual(candidate.calls, [])

        self.assertTrue(self._run(ops, logits, lengths, output, 512, 768))
        workspace = candidate.calls[-1][3]
        pointer = workspace.data_ptr()
        self.assertEqual(workspace.dtype, torch.uint8)
        self.assertEqual(workspace.numel(), 1024 * 1024)
        self.assertTrue(workspace.is_contiguous())

        batched_logits = torch.randn((4, 768), dtype=torch.float32)
        batched_lengths = torch.tensor([1, 511, 512, 768], dtype=torch.int32)
        batched_output = torch.empty((4, 512), dtype=torch.int32)
        with mock.patch.object(
            indexer, "_decode_topk_capture_active", return_value=True
        ):
            self.assertTrue(
                self._run(
                    ops,
                    batched_logits,
                    batched_lengths,
                    batched_output,
                    512,
                    768,
                )
            )
        replay_workspace = candidate.calls[-1][3]
        self.assertIs(replay_workspace, workspace)
        self.assertEqual(replay_workspace.data_ptr(), pointer)

    def test_independent_oracle_covers_k_geometry_batch_and_padding(self) -> None:
        candidate = _UnorderedTopK()
        ops = SimpleNamespace(dsv4_persistent_topk=candidate)
        cases = (
            (512, 257, [0, 1, 256, 257]),
            (512, 512, [1, 511, 512]),
            (512, 777, [7, 512, 777]),
            (1024, 1024, [513, 1024]),
            (2048, 2305, [2047, 2048, 2305]),
        )
        generator = torch.Generator().manual_seed(17)
        for k, width, row_lengths in cases:
            with self.subTest(k=k, width=width):
                logits = torch.randn(
                    (len(row_lengths), width), generator=generator, dtype=torch.float32
                )
                # Explicit ties are normalized only inside the independent oracle
                # and ABI double, never by the production route.
                logits[:, : min(8, width)] = 3.0
                lengths = torch.tensor(row_lengths, dtype=torch.int32)
                output = torch.empty((len(row_lengths), k), dtype=torch.int32)
                self.assertTrue(
                    self._run(ops, logits, lengths, output, k, width)
                )
                _assert_output_contract(self, output, logits, lengths, k)


if __name__ == "__main__":
    unittest.main()
