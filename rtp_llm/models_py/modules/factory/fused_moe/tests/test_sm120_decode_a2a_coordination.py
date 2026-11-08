"""Eager SM120 EP all-to-all coordination regressions.

``Sm120DecodeExecutor._forward_all_to_all`` is the path fused/speculative
PDFUSION decode takes when CUDA graphs are off. These tests drive the real
executor body with a scripted ``torch.distributed`` so the collective
sequence and split extents a rank actually issues are asserted directly,
without needing GPUs: uneven batches, idle ranks, oversized rounds, and
count/order mismatches. The fake collective backend validates every call's
split sums against the real tensor extents, and the mocked compute leaves
produce correctly shaped payloads, so invalid extents cannot pass
silently. They prove CPU-level coordination invariants only; multi-GPU
correctness stays a hardware gate.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    sm120_decode as sm120_decode_mod,
)

DIM = 128  # scale_end = DIM + DIM // 32 = 132 (4-byte aligned slices)
TOPK = 2


def _fake_flashinfer():
    fake = types.ModuleType("flashinfer")

    # Keep the real byte layout: activation payload stays fp8-sized and the
    # scale tail is (rows, cols // 32) uint8, as the executor expects.
    def mxfp8_quantize(x, is_sf_swizzled_layout=False):
        return (
            x.to(torch.float8_e4m3fn),
            torch.zeros((x.size(0), max(1, x.size(1) // 32)), dtype=torch.uint8),
        )

    fake.mxfp8_quantize = mxfp8_quantize
    return fake


class _FakeDist:
    """Minimal dist backend that replays a fixed group's send counts and
    rejects collectives whose split sizes do not match the real extents."""

    def __init__(self, world, rank, send_counts_by_rank):
        self.group = SimpleNamespace(WORLD=object())
        self.world = world
        self.rank = rank
        self.send_counts_by_rank = send_counts_by_rank
        self.calls = []

    def get_world_size(self, group=None):
        return self.world

    def get_rank(self, group=None):
        return self.rank

    def all_gather_into_tensor(self, out, inp, group=None):
        self.calls.append(("gather", tuple(int(v) for v in inp.view(-1).tolist())))
        out.copy_(
            torch.tensor(self.send_counts_by_rank, dtype=out.dtype).view(out.shape)
        )

    def all_to_all_single(
        self, out, inp, output_split_sizes=None, input_split_sizes=None, group=None
    ):
        assert output_split_sizes is not None and input_split_sizes is not None
        assert len(output_split_sizes) == self.world
        assert len(input_split_sizes) == self.world
        assert sum(output_split_sizes) == out.size(0), (
            output_split_sizes,
            tuple(out.shape),
        )
        assert sum(input_split_sizes) == inp.size(0), (
            input_split_sizes,
            tuple(inp.shape),
        )
        assert out.size(1) == inp.size(1)
        self.calls.append(
            (
                "a2a",
                list(output_split_sizes),
                list(input_split_sizes),
            )
        )
        if out.numel():
            out.zero_()


def _executor(world):
    executor = sm120_decode_mod.Sm120DecodeExecutor.__new__(
        sm120_decode_mod.Sm120DecodeExecutor
    )
    # Bypass the real constructor but keep nn.Module bookkeeping, so
    # patch.object can delete its instance attribute on exit.
    torch.nn.Module.__init__(executor)
    executor.cfg = SimpleNamespace(n_routed_experts=world * 2)
    executor._chunk_tokens = 1024
    return executor


def _combine_mock(dim):
    def fake_compute(recv_buf, n_rows, x_arg, x_end, scale_end, weight_end):
        # The real return path sends one combine row per received row, with
        # the activation width restored to fp32.
        return torch.zeros((n_rows, dim), dtype=torch.float32)

    return fake_compute


def _run(executor, fake, x, weights, indices):
    with (
        patch("torch.distributed", fake),
        patch.dict(sys.modules, {"flashinfer": _fake_flashinfer()}),
        patch.object(
            executor, "_compute_received", side_effect=_combine_mock(x.size(1))
        ),
        patch.object(
            sm120_decode_mod,
            "fp32_peer_sum",
            side_effect=lambda payload, world, output: output,
        ),
    ):
        return executor._forward_all_to_all(x, weights, indices)


def _inputs(rows, dim=DIM, topk=TOPK):
    x = torch.zeros((rows, dim), dtype=torch.float32)
    weights = torch.ones((rows, topk), dtype=torch.float32)
    indices = torch.zeros((rows, topk), dtype=torch.int64)
    return x, weights, indices


class Sm120DecodeA2ACoordinationTest(unittest.TestCase):
    def test_count_gather_precedes_dispatch_and_return(self):
        world, rank = 4, 1
        counts = [0, 3, 2, 0]
        fake = _FakeDist(world, rank, counts)
        executor = _executor(world)
        x, weights, indices = _inputs(counts[rank])
        _run(executor, fake, x, weights, indices)

        kinds = [call[0] for call in fake.calls]
        self.assertEqual(kinds, ["gather", "a2a", "a2a"])
        _, out_splits, in_splits = fake.calls[1]
        # Dispatch: receive each rank's advertised rows; send local rows to
        # every peer (fixed per-peer extent).
        self.assertEqual(out_splits, counts)
        self.assertEqual(in_splits, [counts[rank]] * world)
        _, out_splits, in_splits = fake.calls[2]
        # Combine: the same extents swap direction, no extra count exchange.
        self.assertEqual(out_splits, [counts[rank]] * world)
        self.assertEqual(in_splits, counts)

    def test_idle_rank_still_joins_collectives_with_zero_rows(self):
        world, rank = 4, 0
        counts = [0, 5, 2, 0]
        fake = _FakeDist(world, rank, counts)
        executor = _executor(world)
        x, weights, indices = _inputs(0)
        out = _run(executor, fake, x, weights, indices)

        self.assertEqual([call[0] for call in fake.calls], ["gather", "a2a", "a2a"])
        self.assertEqual(fake.calls[1][2], [0] * world)
        self.assertEqual(fake.calls[1][1], counts)
        self.assertEqual(out.shape, (0, x.size(1)))

    def test_all_idle_round_performs_only_the_count_gather(self):
        world, rank = 4, 2
        fake = _FakeDist(world, rank, [0] * world)
        executor = _executor(world)
        x, weights, indices = _inputs(0)
        out = _run(executor, fake, x, weights, indices)

        self.assertEqual([call[0] for call in fake.calls], ["gather"])
        self.assertEqual(out.shape, (0, x.size(1)))

    def test_oversize_rounds_are_uniform_for_busy_and_idle_ranks(self):
        world = 4
        counts = [0, 5, 2, 0]
        for rank in (0, 2):
            fake = _FakeDist(world, rank, counts)
            executor = _executor(world)
            x, weights, indices = _inputs(counts[rank])
            with (
                patch.object(sm120_decode_mod, "_A2A_PAYLOAD_TOKEN_BOUND", 4),
                patch.object(executor, "_chunk_tokens", 1),
                patch("torch.distributed", fake),
                patch.dict(sys.modules, {"flashinfer": _fake_flashinfer()}),
                patch.object(
                    executor,
                    "_compute_received",
                    side_effect=_combine_mock(x.size(1)),
                ),
                patch.object(
                    sm120_decode_mod,
                    "fp32_peer_sum",
                    side_effect=lambda payload, w, output: output,
                ),
            ):
                executor._forward_all_to_all(x, weights, indices)

            # One count gather, then ceil(max(counts)/chunk)=5 coordinated
            # rounds of dispatch+combine, identical on every rank.
            expected_rounds = [
                [max(0, min(c - begin, 1)) for c in counts] for begin in range(5)
            ]
            a2a_calls = [call for call in fake.calls if call[0] == "a2a"]
            self.assertEqual(len(a2a_calls), 2 * len(expected_rounds))
            for i, chunk in enumerate(expected_rounds):
                _, out_splits, in_splits = a2a_calls[2 * i]
                self.assertEqual(out_splits, chunk)
                self.assertEqual(in_splits, [chunk[rank]] * world)

    def test_count_mismatch_is_rejected_before_any_dispatch(self):
        world, rank = 4, 3
        fake = _FakeDist(world, rank, [1, 2, 2, 9])  # rank 3 reports 9, has 2
        executor = _executor(world)
        x, weights, indices = _inputs(2)
        with self.assertRaisesRegex(RuntimeError, "collective protocol"):
            _run(executor, fake, x, weights, indices)
        self.assertEqual([call[0] for call in fake.calls], ["gather"])


if __name__ == "__main__":
    unittest.main()
