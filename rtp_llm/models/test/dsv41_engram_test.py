"""Real small host-table composition, chunk ownership and failure contracts."""

import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_engram_host_test import host, scalar_row, write_fixture
from rtp_llm.models.test.dsv41_engram_inject_test import checkpoint_reference, inject
from rtp_llm.models.test.dsv41_mxfp8_test import (
    grouped_reference,
    linear,
    quantization_reference,
    weights,
)

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_engram_host": host,
        "rtp_llm.models_py.modules.dsv4.v41_engram_inject": inject,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    engram = load_codec_source("_dsv41_engram_composition_test", "v41_engram.py")


class RecordingEmbedding(host.HostEngramEmbedding):
    def __init__(self, table):
        super().__init__(table)
        self.calls = []

    def __call__(self, indices, valid_rows, **kwargs):
        self.calls.append(indices.shape[0])
        return super().__call__(indices, valid_rows, **kwargs)


class V41EngramTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        raw, scales = write_fixture(root)
        source = host.V41EngramTableSource.from_checkpoint(root, 1, 7, 64, "a" * 64)
        self.table = host._PinnedSharedTable(source, max_bytes=4096, directory=root)
        self.addCleanup(self.table.close)
        self.embedding = RecordingEmbedding(self.table)
        self.decoded = torch.stack(
            [
                scalar_row(raw[i * 64 : (i + 1) * 64], scales[i * 2 : (i + 1) * 2])
                for i in range(7)
            ]
        )
        self.projection = linear.V41MXFP8Linear(*weights(n=160, k=128), backend="eager")
        self.q = torch.ones(4, 32, dtype=torch.bfloat16)
        self.k = torch.linspace(-1, 1, 128).reshape(4, 32).bfloat16()

    def component(self, chunk=3):
        return engram.V41Engram(
            self.embedding,
            self.projection,
            self.q,
            self.k,
            eps=1e-20,
            chunk_tokens=chunk,
        )

    @staticmethod
    def inputs(tokens=11):
        hidden = torch.randn(
            tokens, 4, 32, generator=torch.Generator().manual_seed(41)
        ).bfloat16()
        ids = (torch.arange(tokens * 4).reshape(tokens, 4) % 9 - 1)[:, ::2]
        valid = torch.arange(tokens) % 3 != 0
        return hidden, ids, valid

    def reference(self, hidden, ids, valid):
        lookup = torch.zeros((*ids.shape, 64), dtype=torch.bfloat16)
        live = (ids >= 0) & (ids < 7) & valid[:, None]
        lookup[live] = self.decoded[ids[live]]
        raw, scales = quantization_reference(lookup.flatten(1))
        projected = grouped_reference(
            raw.view(torch.float8_e4m3fn),
            scales,
            self.projection.weight,
            self.projection.scales,
        )
        return checkpoint_reference(hidden, projected, self.q, self.k, valid, 1e-20)

    def test_matches_complete_checkpoint_equations_and_bounds_chunks(self):
        args = self.inputs()
        expected = self.reference(*args)
        for chunk in (1, 3, 11, 128):
            self.embedding.calls.clear()
            output = self.component(chunk)(*args)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertEqual(sum(self.embedding.calls), 11)
            self.assertLessEqual(max(self.embedding.calls), chunk)

    def test_alias_dead_rows_and_empty_batches(self):
        args = self.inputs()
        args[0][~args[2]] = -0.0
        args[0][0].view(torch.int16).fill_(32705)
        expected = self.reference(*args)
        before = args[0].clone()
        op = self.component()
        with torch.inference_mode():
            self.assertIs(op(*args, out=args[0]), args[0])
        torch.testing.assert_close(args[0], expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            args[0][~args[2]].view(torch.uint8), before[~args[2]].view(torch.uint8)
        )
        self.embedding.calls.clear()
        empty = op(*self.inputs(0))
        self.assertEqual(empty.shape, (0, 4, 32))
        self.assertEqual(self.embedding.calls, [])

    def test_cp_permutation_and_repeated_forwards_have_no_retained_token_state(self):
        args = self.inputs()
        expected = self.reference(*args)
        op = self.component()
        order = torch.tensor([8, 0, 3, 10, 1, 5, 4, 2, 9, 7, 6])
        torch.testing.assert_close(
            op(*(x[order] for x in args)), expected[order], rtol=0, atol=0
        )
        op(*self.inputs(2))
        torch.testing.assert_close(op(*args), expected, rtol=0, atol=0)
        self.assertEqual(list(op.buffers()), [])
        self.assertEqual(list(op.parameters()), [])
        self.assertEqual(op.layer_id, 1)

    def test_invalid_metadata_and_gradients_fail_before_lookup_or_output_write(self):
        args = self.inputs()
        output = torch.full_like(args[0], 42)
        op = self.component()
        for index, bad in (
            (0, args[0].float()),
            (0, args[0].transpose(1, 2)),
            (0, args[0].clone().requires_grad_()),
            (1, args[1].float()),
            (1, args[1][:, :1]),
            (2, args[2].long()),
            (2, args[2][:-1]),
        ):
            changed = list(args)
            changed[index] = bad
            with self.assertRaises(ValueError):
                op(*changed, out=output)
        self.assertEqual(self.embedding.calls, [])
        torch.testing.assert_close(output, torch.full_like(output, 42))

    def test_partial_hidden_and_weight_output_aliases_are_rejected(self):
        op = self.component()
        _, ids, valid = self.inputs(1)
        storage = torch.zeros(129, dtype=torch.bfloat16)
        hidden, output = storage[:-1].view(1, 4, 32), storage[1:].view(1, 4, 32)
        with self.assertRaisesRegex(ValueError, "exact"):
            op(hidden, ids, valid, out=output)
        for alias in (
            self.q.view(1, 4, 32),
            self.projection.weight.view(torch.bfloat16).flatten()[:128].view(1, 4, 32),
        ):
            with self.assertRaisesRegex(ValueError, "share storage"):
                op(hidden, ids, valid, out=alias)
        self.assertEqual(self.embedding.calls, [])

    def test_constructor_rejects_incompatible_geometry_and_chunk_budget(self):
        for chunk in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "chunk_tokens"):
                self.component(chunk)
        for eps in (0, -1, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "epsilon"):
                engram.V41Engram(
                    self.embedding, self.projection, self.q, self.k, eps=eps
                )
        wrong = linear.V41MXFP8Linear(*weights(n=128, k=128), backend="eager")
        with self.assertRaisesRegex(ValueError, "geometry"):
            engram.V41Engram(self.embedding, wrong, self.q, self.k, eps=1e-20)
        with self.assertRaisesRegex(ValueError, "BF16"):
            engram.V41Engram(
                self.embedding, self.projection, self.q.float(), self.k, eps=1e-20
            )

    def test_provider_failure_propagates_with_explicit_partial_output_semantics(self):
        op = self.component(3)
        args = self.inputs()
        output = torch.full_like(args[0], 42)
        original = self.projection.forward
        count = 0

        def fail_second_chunk(*a, **kw):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("GEMM failure")
            return original(*a, **kw)

        with patch.object(self.projection, "forward", side_effect=fail_second_chunk):
            with self.assertRaisesRegex(RuntimeError, "GEMM failure"):
                op(*args, out=output)
        torch.testing.assert_close(
            output[:3], self.reference(*(x[:3] for x in args)), rtol=0, atol=0
        )
        torch.testing.assert_close(output[3:], torch.full_like(output[3:], 42))
        self.table._check_open()

    def test_closed_owner_fails_before_any_output_write(self):
        op = self.component()
        args = self.inputs()
        output = torch.full_like(args[0], 42)
        self.table.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            op(*args, out=output)
        torch.testing.assert_close(output, torch.full_like(output, 42))

    def test_registration_and_capture_admission_precede_lookup(self):
        op = self.component()
        op.backend = "triton"
        args = self.inputs()
        output = torch.full_like(args[0], 42)
        with self.assertRaisesRegex(ValueError, "registered"):
            op(*args, out=output)
        # Only the rejection contract is mocked; this does not exercise CUDA.
        with patch.object(self.table, "_runtime", object()), patch.object(
            self.table, "_device", args[0].device
        ), patch.object(torch.cuda, "device", return_value=nullcontext()), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "worker lifecycle"):
                op(*args, out=output)
        self.assertEqual(self.embedding.calls, [])
        torch.testing.assert_close(output, torch.full_like(output, 42))


if __name__ == "__main__":
    unittest.main()
