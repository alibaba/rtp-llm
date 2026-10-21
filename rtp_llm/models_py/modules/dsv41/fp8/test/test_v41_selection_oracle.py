"""Deterministic eager test hooks; not for timing or CUDA graph capture.

Equal scores choose ascending column IDs. CPU stable sorting is intentional:
the oracle must not inherit the native selectors' tie nondeterminism. Chunking
bounds scratch without materializing another whole long-context logits matrix.
"""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch


def deterministic_topk(logits, lengths, k=512, *, mode="prefill", out=None):
    """Preserve each selector's short-row/nonfinite contract, fixing only ties."""
    if mode not in ("prefill", "decode", "sparse"):
        raise ValueError("unknown selector contract")
    if logits.ndim != 2 or not 0 < k <= logits.shape[1]:
        raise ValueError("expected [rows, width] logits with width >= k")
    rows, width = logits.shape
    ends = lengths.detach().to(device="cpu", dtype=torch.int64).clamp(0, width)
    if ends.shape != (rows,):
        raise ValueError("one visible length is required per query")
    result = torch.full((rows, k), -1, dtype=torch.int32)
    chunk_rows = max(1, min(rows, (1 << 20) // width))
    columns = torch.arange(width)
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        scores = logits[begin:end].detach().to(device="cpu", dtype=torch.float32)
        visible = ends[begin:end]
        scores = scores.masked_fill(columns[None] >= visible[:, None], -torch.inf)
        selected = scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
        selected_scores = scores.gather(1, selected)
        if mode != "sparse":
            selected = selected.masked_fill(~selected_scores.isfinite(), -1)
        if mode in ("decode", "sparse"):
            short = visible <= k
            prefix = torch.arange(k).expand(end - begin, k)
            prefix = prefix.masked_fill(prefix >= visible[:, None], -1)
            selected = torch.where(short[:, None], prefix, selected)
        if mode == "sparse":
            bad_long_row = (visible > k) & scores.isnan().any(dim=-1)
            selected = selected.masked_fill(bad_long_row[:, None], -1)
        result[begin:end].copy_(selected)
    if out is None:
        return result.to(device=logits.device)
    if (
        out.shape != (rows, k)
        or out.dtype != torch.int32
        or out.device != logits.device
    ):
        raise ValueError("output must match selector shape, dtype and device")
    out.copy_(result)
    return out


@contextmanager
def deterministic_selection_hooks(*, prefill=None, decode=None, sparse=None):
    """Patch supplied V4.1 selector modules, restoring them even after errors.

    Modules are explicit so importing this test helper never loads production
    native bindings. Unsupported inputs retain each production support gate.
    """
    with ExitStack() as stack:
        if prefill is not None:

            def select_prefill(logits, visible, topk=512, *, bounds=None, out=None):
                if not prefill.is_supported(logits, visible, topk):
                    return None
                return deterministic_topk(
                    logits, visible if bounds is None else bounds[1], topk, out=out
                )

            stack.enter_context(
                patch.object(prefill, "try_select_tokens", select_prefill)
            )
        if decode is not None:

            def select_decode(logits, lengths, topk):
                return deterministic_topk(logits, lengths, topk, mode="decode")

            stack.enter_context(patch.object(decode, "select_tokens", select_decode))
        if sparse is not None:

            def select_sparse(logits, end, out=None):
                if not sparse.is_supported(logits, end, out):
                    return None
                return deterministic_topk(logits, end, mode="sparse", out=out)

            stack.enter_context(
                patch.object(sparse, "try_select_sparse_tokens", select_sparse)
            )
        yield


class SelectionOracleTest(unittest.TestCase):
    def test_ties_use_column_order_for_float32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.zeros(3, 1024, dtype=dtype)
            ends = torch.tensor([0, 513, 2048])
            for mode in ("prefill", "decode", "sparse"):
                with self.subTest(dtype=dtype, mode=mode):
                    got = deterministic_topk(logits, ends, mode=mode)
                    self.assertTrue((got[0] == -1).all())
                    torch.testing.assert_close(got[1], torch.arange(512).int())
                    torch.testing.assert_close(got[2], got[1])
                    torch.testing.assert_close(
                        deterministic_topk(logits, ends, mode=mode), got
                    )

    def test_nonfinite_contracts_differ_for_short_and_long_rows(self):
        logits = torch.zeros(3, 1024)
        logits[:, 0] = torch.nan
        logits[:, 1] = torch.inf
        logits[:, 2] = -torch.inf
        ends = torch.tensor([3, 1024, 0])
        prefill = deterministic_topk(logits, ends)
        self.assertTrue((prefill[0] == -1).all())
        self.assertEqual(prefill[1, :3].tolist(), [-1, -1, 3])
        for mode in ("decode", "sparse"):
            got = deterministic_topk(logits, ends, mode=mode)
            self.assertEqual(got[0, :4].tolist(), [0, 1, 2, -1])
            self.assertTrue((got[2] == -1).all())
            if mode == "sparse":
                self.assertTrue((got[1] == -1).all())

    def test_strides_clamped_bounds_output_alias_and_hidden_nan(self):
        logits = torch.arange(2048).float().reshape(2, 1024)[:, ::2]
        logits[1, -1] = torch.nan
        ends = torch.tensor([2**40, 17])
        storage = torch.full((2, 520), -7, dtype=torch.int32)
        out = storage[:, :512]
        self.assertIs(deterministic_topk(logits, ends, out=out), out)
        self.assertEqual(out[0, :3].tolist(), [511, 510, 509])
        self.assertEqual(out[1, :3].tolist(), [16, 15, 14])
        self.assertTrue((storage[:, 512:] == -7).all())

    def test_hooks_restore_on_exception_and_honor_support_gate(self):
        original = lambda *args, **kwargs: "native"
        prefill = SimpleNamespace(
            try_select_tokens=original, is_supported=lambda *a: True
        )
        decode = SimpleNamespace(select_tokens=original)
        sparse = SimpleNamespace(
            try_select_sparse_tokens=original, is_supported=lambda *a: True
        )
        logits, ends = torch.ones(1, 1024), torch.tensor([1024])
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with deterministic_selection_hooks(
                prefill=prefill, decode=decode, sparse=sparse
            ):
                for selected in (
                    prefill.try_select_tokens(logits, ends),
                    decode.select_tokens(logits, ends, 512),
                    sparse.try_select_sparse_tokens(logits.bfloat16(), ends),
                ):
                    torch.testing.assert_close(selected[0], torch.arange(512).int())
                prefill.is_supported = lambda *a: False
                self.assertIsNone(prefill.try_select_tokens(logits, ends))
                raise RuntimeError("probe")
        self.assertIs(prefill.try_select_tokens, original)
        self.assertIs(decode.select_tokens, original)
        self.assertIs(sparse.try_select_sparse_tokens, original)


if __name__ == "__main__":
    unittest.main()
