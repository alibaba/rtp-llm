"""UT for the fused DSpARK draft decode metadata Triton kernel.

``_dspark_meta_triton.try_compute_dspark_decode_meta`` must reproduce the
production torch chains bit-for-bit: the query pool slots from
``DeepSeekV4DSparkModel._global_pool_slots`` and the packed-left
non-causal topk rows + lengths from ``_build_noncausal_indices``.  The
reference is the model module itself (not a re-implementation), so any
drift in the production math fails here.  Coverage includes the ``-1``
sentinels (unallocated / negative block ids, block-table overflow,
negative positions, inactive requests), the identical non-causal gamma
rows, the FlashMLA 128-aligned topk width, and CUDA-graph replay picking
up fresh device-side inputs.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DeepSeekV4DSparkModel,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _dspark_meta_triton

_CUDA_OK = torch.cuda.is_available() and _dspark_meta_triton._TRITON_AVAILABLE
_DEVICE = "cuda" if _CUDA_OK else "cpu"


def _reference_model(gamma: int, window: int) -> DeepSeekV4DSparkModel:
    """Shell instance carrying only the fields the reference methods read."""
    model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
    model._gen_num_per_cycle = gamma
    model._v4_args = SimpleNamespace(window_size=window)
    return model


def _torch_reference(
    gamma: int,
    window: int,
    entries_per_block: int,
    tokens_per_block: int,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    prefix_lengths: torch.Tensor,
    active_requests: torch.Tensor,
):
    """Run the production torch chains (the model's fallback path)."""
    batch_size = int(prefix_lengths.numel())
    model = _reference_model(gamma, window)
    query_req_ids = torch.arange(
        batch_size, dtype=torch.long, device=block_table.device
    ).repeat_interleave(gamma)
    query_slots = DeepSeekV4DSparkModel._global_pool_slots(
        block_table,
        query_req_ids,
        positions.reshape(-1),
        entries_per_block,
        tokens_per_block,
    )
    global_indices, topk_length = model._build_noncausal_indices(
        prefix_lengths,
        active_requests,
        block_table,
        entries_per_block,
        tokens_per_block,
    )
    return query_slots, global_indices, topk_length


def _aligned_topk(window: int, gamma: int) -> int:
    return ((window + gamma + 127) // 128) * 128


class DSparkMetaGateTest(unittest.TestCase):
    """Unsupported inputs must signal the torch fallback (return None)."""

    def test_cpu_inputs_rejected(self):
        block_table = torch.ones((2, 3), dtype=torch.int32)
        positions = torch.zeros((2, 4), dtype=torch.int64)
        prefix = torch.tensor([5, 0], dtype=torch.int32)
        active = torch.tensor([True, False])
        self.assertIsNone(
            _dspark_meta_triton.try_compute_dspark_decode_meta(
                block_table,
                positions,
                prefix,
                active,
                gamma=4,
                window=64,
                entries_per_block=16,
                tokens_per_block=32,
            )
        )

    def test_wrong_dtypes_rejected(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prefix = torch.tensor([5, 0], dtype=torch.int32, device=device)
        active = torch.tensor([True, False], device=device)
        positions = torch.zeros((2, 4), dtype=torch.int64, device=device)
        int64_block_table = torch.ones((2, 3), dtype=torch.int64, device=device)
        self.assertIsNone(
            _dspark_meta_triton.try_compute_dspark_decode_meta(
                int64_block_table,
                positions,
                prefix,
                active,
                gamma=4,
                window=64,
                entries_per_block=16,
                tokens_per_block=32,
            )
        )

    def test_noncontiguous_inputs_rejected(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prefix = torch.tensor([5, 0], dtype=torch.int32, device=device)
        active = torch.tensor([True, False], device=device)
        block_table = torch.ones((2, 6), dtype=torch.int32, device=device)[:, ::2]
        positions = torch.zeros((2, 8), dtype=torch.int64, device=device)[:, ::2]
        self.assertFalse(block_table.is_contiguous())
        self.assertFalse(positions.is_contiguous())
        self.assertIsNone(
            _dspark_meta_triton.try_compute_dspark_decode_meta(
                block_table,
                positions,
                prefix,
                active,
                gamma=4,
                window=64,
                entries_per_block=16,
                tokens_per_block=32,
            )
        )

    def test_shape_mismatch_rejected(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block_table = torch.ones((2, 3), dtype=torch.int32, device=device)
        prefix = torch.tensor([5, 0], dtype=torch.int32, device=device)
        active = torch.tensor([True, False], device=device)
        wrong_gamma = torch.zeros((2, 5), dtype=torch.int64, device=device)
        self.assertIsNone(
            _dspark_meta_triton.try_compute_dspark_decode_meta(
                block_table,
                wrong_gamma,
                prefix,
                active,
                gamma=4,
                window=64,
                entries_per_block=16,
                tokens_per_block=32,
            )
        )


@unittest.skipUnless(_CUDA_OK, "CUDA/Triton unavailable")
class DSparkMetaTritonTest(unittest.TestCase):
    """Bit-exact comparison against the production torch chains."""

    def _compare(
        self,
        *,
        gamma: int,
        window: int,
        entries_per_block: int,
        tokens_per_block: int,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        active_requests: torch.Tensor,
    ):
        batch_size = int(prefix_lengths.numel())
        ref_slots, ref_indices, ref_lengths = _torch_reference(
            gamma,
            window,
            entries_per_block,
            tokens_per_block,
            block_table,
            positions,
            prefix_lengths,
            active_requests,
        )
        fused = _dspark_meta_triton.try_compute_dspark_decode_meta(
            block_table,
            positions,
            prefix_lengths,
            active_requests,
            gamma=gamma,
            window=window,
            entries_per_block=entries_per_block,
            tokens_per_block=tokens_per_block,
        )
        self.assertIsNotNone(fused)
        got_slots, got_indices, got_lengths = fused
        self.assertEqual(got_slots.dtype, torch.int64)
        self.assertEqual(got_indices.dtype, torch.int32)
        self.assertEqual(got_lengths.dtype, torch.int32)
        self.assertEqual(
            tuple(got_indices.shape),
            (batch_size * gamma, _aligned_topk(window, gamma)),
        )
        self.assertTrue(torch.equal(got_slots, ref_slots))
        self.assertTrue(torch.equal(got_indices, ref_indices))
        self.assertTrue(torch.equal(got_lengths, ref_lengths))
        return fused

    def _positions_from_prefix(
        self, prefix_lengths: torch.Tensor, gamma: int
    ) -> torch.Tensor:
        return prefix_lengths.to(torch.long).view(-1, 1) + torch.arange(
            gamma, device=prefix_lengths.device
        ).view(1, gamma)

    def test_matches_reference_mixed_sentinels(self):
        gamma, window, epb, tpb = 8, 200, 134, 256
        block_table = torch.tensor(
            [
                [3, 4, 0, 6],
                [0, 0, 0, 0],
                [7, 8, 9, 10],
                [-1, 12, 13, 14],
            ],
            dtype=torch.int32,
            device=_DEVICE,
        )
        prefix_lengths = torch.tensor(
            [500, 0, 130, 40], dtype=torch.int32, device=_DEVICE
        )
        positions = self._positions_from_prefix(prefix_lengths, gamma)
        active_requests = prefix_lengths > 0
        self._compare(
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
            block_table=block_table,
            positions=positions,
            prefix_lengths=prefix_lengths,
            active_requests=active_requests,
        )

    def test_matches_reference_nonpow2_gamma(self):
        gamma, window, epb, tpb = 5, 64, 16, 32
        block_table = torch.tensor(
            [[3, 0, 5], [6, 7, -2], [0, 9, 10]],
            dtype=torch.int32,
            device=_DEVICE,
        )
        prefix_lengths = torch.tensor([33, 200, 7], dtype=torch.int32, device=_DEVICE)
        positions = self._positions_from_prefix(prefix_lengths, gamma)
        active_requests = prefix_lengths > 0
        self._compare(
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
            block_table=block_table,
            positions=positions,
            prefix_lengths=prefix_lengths,
            active_requests=active_requests,
        )

    def test_matches_reference_block_capacity_overflow(self):
        gamma, window, epb, tpb = 4, 4096, 64, 256
        block_table = torch.tensor([[9, 10]], dtype=torch.int32, device=_DEVICE)
        prefix_lengths = torch.tensor([800], dtype=torch.int32, device=_DEVICE)
        positions = self._positions_from_prefix(prefix_lengths, gamma)
        active_requests = prefix_lengths > 0
        got_slots, got_indices, _ = self._compare(
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
            block_table=block_table,
            positions=positions,
            prefix_lengths=prefix_lengths,
            active_requests=active_requests,
        )
        # Context positions < 512 stay in block 0/1; everything from column
        # 512 on (context tail and the whole query block) exceeds the
        # two-row table and must be a -1 sentinel, including all query slots.
        row = got_indices[0]
        self.assertTrue(torch.all(row[:512] >= 0))
        self.assertTrue(torch.all(row[512:] == -1))
        self.assertTrue(torch.all(got_slots == -1))

    def test_matches_reference_negative_query_positions(self):
        gamma, window, epb, tpb = 4, 64, 16, 32
        block_table = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32, device=_DEVICE)
        prefix_lengths = torch.tensor([40, 33], dtype=torch.int32, device=_DEVICE)
        positions = self._positions_from_prefix(prefix_lengths, gamma)
        positions[1, 2:] = -1  # padded / hole tail
        active_requests = prefix_lengths > 0
        got_slots, _, _ = self._compare(
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
            block_table=block_table,
            positions=positions,
            prefix_lengths=prefix_lengths,
            active_requests=active_requests,
        )
        self.assertTrue(torch.all(got_slots[6:8] == -1))

    def test_packed_left_layout_and_identical_gamma_rows(self):
        gamma, window, epb, tpb = 8, 192, 134, 256
        block_table = torch.tensor(
            [[5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32, device=_DEVICE
        )
        prefix = 431  # > window
        prefix_lengths = torch.tensor([prefix, 0], dtype=torch.int32, device=_DEVICE)
        positions = self._positions_from_prefix(prefix_lengths, gamma)
        active_requests = prefix_lengths > 0
        got_slots, got_indices, got_lengths = self._compare(
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
            block_table=block_table,
            positions=positions,
            prefix_lengths=prefix_lengths,
            active_requests=active_requests,
        )
        committed = min(prefix, window)
        self.assertEqual(int(got_lengths[0]), committed + gamma)
        self.assertEqual(int(got_lengths[1]), 0)
        row = got_indices[0]
        self.assertTrue(torch.all(row[: committed + gamma] >= 0))
        self.assertTrue(torch.all(row[committed + gamma :] == -1))
        for g in range(1, gamma):
            self.assertTrue(torch.equal(got_indices[g], row))
        self.assertTrue(torch.all(got_indices[gamma:] == -1))
        # Row-major query slots: request 0 positions 431..438 all live in
        # block 1 (block id 6); the zeroed inactive row maps to -1.
        expected_q0 = [6 * epb + (prefix + g) % epb for g in range(gamma)]
        self.assertEqual(got_slots[:gamma].tolist(), expected_q0)
        self.assertTrue(torch.all(got_slots[gamma:] == -1))

    def test_matches_reference_randomized_shapes(self):
        torch.manual_seed(20261126)
        for gamma in (1, 3, 8, 16):
            for window in (7, 128, 511):
                for epb, tpb in ((134, 256), (16, 16), (256, 64)):
                    block_table = torch.randint(
                        -1, 30, (5, 7), dtype=torch.int32, device=_DEVICE
                    )
                    # Keep one fully-allocated row so the valid path runs.
                    block_table[0] = torch.randint(
                        1, 30, (7,), dtype=torch.int32, device=_DEVICE
                    )
                    prefix_lengths = torch.randint(
                        0, 1500, (5,), dtype=torch.int32, device=_DEVICE
                    )
                    active_requests = prefix_lengths > 0
                    positions = self._positions_from_prefix(prefix_lengths, gamma)
                    self._compare(
                        gamma=gamma,
                        window=window,
                        entries_per_block=epb,
                        tokens_per_block=tpb,
                        block_table=block_table,
                        positions=positions,
                        prefix_lengths=prefix_lengths,
                        active_requests=active_requests,
                    )


@unittest.skipUnless(_CUDA_OK, "CUDA/Triton unavailable")
class DSparkMetaCudaGraphTest(unittest.TestCase):
    """The captured launch must replay with fresh device-side inputs."""

    def test_replay_tracks_device_side_inputs(self):
        gamma, window, epb, tpb = 8, 192, 134, 256
        batch_size, max_blocks = 4, 6
        torch.manual_seed(7)
        block_table = torch.randint(
            1, 50, (batch_size, max_blocks), dtype=torch.int32, device=_DEVICE
        )
        block_table[3, :] = 0  # CUDA-graph padding row
        prefix_lengths = torch.tensor(
            [431, 17, 1000, 0], dtype=torch.int32, device=_DEVICE
        )
        positions = prefix_lengths.to(torch.long).view(batch_size, 1) + torch.arange(
            gamma, device=_DEVICE
        ).view(1, gamma)
        active_requests = prefix_lengths > 0

        def launch():
            return _dspark_meta_triton.try_compute_dspark_decode_meta(
                block_table,
                positions,
                prefix_lengths,
                active_requests,
                gamma=gamma,
                window=window,
                entries_per_block=epb,
                tokens_per_block=tpb,
            )

        # Warm up (JIT compile) on a side stream, then capture.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            warm = launch()
        torch.cuda.current_stream().wait_stream(side_stream)
        del warm
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = launch()
        self.assertIsNotNone(static_out)

        # Refresh every device-resident input and replay: the captured
        # launch must read the new values from device memory.
        new_prefix = torch.tensor([500, 0, 130, 40], dtype=torch.int32, device=_DEVICE)
        new_block_table = torch.randint(
            -1, 60, (batch_size, max_blocks), dtype=torch.int32, device=_DEVICE
        )
        new_block_table[0] = torch.randint(
            1, 60, (max_blocks,), dtype=torch.int32, device=_DEVICE
        )
        prefix_lengths.copy_(new_prefix)
        block_table.copy_(new_block_table)
        active_requests.copy_(new_prefix > 0)
        positions.copy_(
            new_prefix.to(torch.long).view(batch_size, 1)
            + torch.arange(gamma, device=_DEVICE).view(1, gamma)
        )
        graph.replay()
        torch.cuda.synchronize()

        ref_slots, ref_indices, ref_lengths = _torch_reference(
            gamma,
            window,
            epb,
            tpb,
            block_table,
            positions,
            prefix_lengths,
            active_requests,
        )
        got_slots, got_indices, got_lengths = static_out
        self.assertTrue(torch.equal(got_slots, ref_slots))
        self.assertTrue(torch.equal(got_indices, ref_indices))
        self.assertTrue(torch.equal(got_lengths, ref_lengths))

    def test_int64_active_flag_above_32bit_stays_active(self):
        # Regression: active_requests values with bit >= 32 (e.g. 1 << 32)
        # must not be truncated by .to(tl.int32) before != 0 check.
        batch_size, max_blocks = 2, 16
        gamma, window, epb, tpb = 4, 64, 16, 32
        block_table = torch.randint(
            1, 40, (batch_size, max_blocks), dtype=torch.int32, device=_DEVICE
        )
        prefix_lengths = torch.tensor([120, 90], dtype=torch.int32, device=_DEVICE)
        positions = prefix_lengths.to(torch.long).view(batch_size, 1) + torch.arange(
            gamma, device=_DEVICE
        ).view(1, gamma)
        active_requests = torch.tensor([1 << 32, 0], dtype=torch.int64, device=_DEVICE)

        got = _dspark_meta_triton.try_compute_dspark_decode_meta(
            block_table,
            positions,
            prefix_lengths,
            active_requests,
            gamma=gamma,
            window=window,
            entries_per_block=epb,
            tokens_per_block=tpb,
        )
        self.assertIsNotNone(got)
        ref_slots, ref_indices, ref_lengths = _torch_reference(
            gamma,
            window,
            epb,
            tpb,
            block_table,
            positions,
            prefix_lengths,
            active_requests,
        )
        got_slots, got_indices, got_lengths = got
        self.assertTrue(torch.equal(got_slots, ref_slots))
        self.assertTrue(torch.equal(got_indices, ref_indices))
        self.assertTrue(torch.equal(got_lengths, ref_lengths))


if __name__ == "__main__":
    unittest.main()
