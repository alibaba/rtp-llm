"""UT for prefill ring write mask in state slot mapping.

The ring buffer layout stores only R = state_eb entries per block. During
prefill, multiple tokens in the same block are processed in parallel, but
only the last R positions before each block boundary (or sequence end)
should actually write to the state pool — earlier positions would be
overwritten by later ones in the ring.

The mask formula: should_write = (pos + R) >= min((block_idx+1)*TPB, seq_end)
where:
  - R = state_eb (ring entries per block)
  - TPB = state_tokens_per_block (block table stride, typically 256)
  - block_idx = pos // TPB
  - seq_end = prefix_length + input_length per request

Tests cover:
  - CSA (ratio=4, overlap=1 → R=8 at gen_num=0)
  - HCA (ratio=128, overlap=0 → R=128 at gen_num=0)
  - Various gen_num_per_cycle: 0, 1, 3, 5
  - Different prefill sequence lengths spanning multiple blocks

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_ring_write_mask \
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import unittest
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8, CompressorMeta


def _compute_ring_size(
    compress_ratio: int, overlap: int, gen_num_per_cycle: int
) -> int:
    window = (1 + overlap) * compress_ratio
    G = max(0, gen_num_per_cycle)
    raw = window + G
    return (raw + 1) & ~1  # ceil to even


class _StubCompressor:
    """Minimal stub for testing prepare_metadata ring write mask."""

    def __init__(
        self,
        compress_ratio: int,
        state_block_table: torch.Tensor,
        state_eb: int,
        state_tokens_per_block: int,
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
    ) -> None:
        self.compress_ratio = compress_ratio
        self._state_block_table = state_block_table
        self._state_eb = state_eb
        self._state_tokens_per_block = state_tokens_per_block
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._kv_cache_sharded = False
        self._cp_ctx = None
        self._kv_pool_3d = None

    _compute_state_slot_mapping = CompressorFP8._compute_state_slot_mapping
    _compute_kv_slot_mapping = CompressorFP8._compute_kv_slot_mapping
    prepare_metadata = CompressorFP8.prepare_metadata


def _make_stub(
    device: torch.device,
    *,
    ratio: int,
    overlap: int,
    gen_num: int,
    n_reqs: int = 1,
    blocks_per_req: int = 8,
    tpb: int = 256,
) -> _StubCompressor:
    eb = _compute_ring_size(ratio, overlap, gen_num)
    state_bt = torch.arange(
        1, n_reqs * blocks_per_req + 1, dtype=torch.int32, device=device
    ).view(n_reqs, blocks_per_req)
    kv_eb = tpb // ratio
    kv_bt = state_bt + 1000
    return _StubCompressor(
        compress_ratio=ratio,
        state_block_table=state_bt,
        state_eb=eb,
        state_tokens_per_block=tpb,
        kv_block_table=kv_bt,
        kv_eb=kv_eb,
    )


def _reference_ring_mask(
    positions: torch.Tensor,
    b_idx: torch.Tensor,
    seq_end_per_req: torch.Tensor,
    tpb: int,
    eb: int,
) -> torch.Tensor:
    """Python reference for the ring write mask (independent of slot mapping)."""
    block_in_seq = positions // tpb
    block_end = (block_in_seq + 1) * tpb
    seq_end = seq_end_per_req[b_idx]
    effective_end = torch.minimum(block_end, seq_end)
    return (positions + eb) >= effective_end


class RingWriteMaskTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def _run_single_request(
        self,
        ratio: int,
        overlap: int,
        gen_num: int,
        prefix_len: int,
        input_len: int,
    ) -> None:
        """Verify state_slots mask for a single-request prefill."""
        tpb = 256
        eb = _compute_ring_size(ratio, overlap, gen_num)
        total_blocks = (prefix_len + input_len + tpb - 1) // tpb + 1
        stub = _make_stub(
            self.device,
            ratio=ratio,
            overlap=overlap,
            gen_num=gen_num,
            n_reqs=1,
            blocks_per_req=total_blocks,
        )

        positions = torch.arange(
            prefix_len, prefix_len + input_len, dtype=torch.int64, device=self.device
        )
        b_idx = torch.zeros(input_len, dtype=torch.int64, device=self.device)
        seq_end = prefix_len + input_len

        meta = stub.prepare_metadata(
            positions,
            b_idx,
            is_batched=True,
            seq_start_per_req=torch.tensor(
                [prefix_len], dtype=torch.int32, device=self.device
            ),
            cu_seq_per_req=torch.tensor(
                [0, input_len], dtype=torch.int32, device=self.device
            ),
        )

        # Compute expected mask
        expected_mask = _reference_ring_mask(
            positions,
            b_idx,
            torch.tensor([seq_end], dtype=torch.int64, device=self.device),
            tpb,
            eb,
        )

        # Verify: state_slots == -1 exactly where mask is False
        state_slots = meta.state_slots
        actual_valid = state_slots >= 0
        self.assertTrue(
            torch.equal(actual_valid, expected_mask),
            f"Ring mask mismatch for ratio={ratio}, overlap={overlap}, "
            f"gen_num={gen_num}, prefix={prefix_len}, input={input_len}, "
            f"eb={eb}\n"
            f"  Expected valid count: {expected_mask.sum().item()}\n"
            f"  Actual valid count: {actual_valid.sum().item()}\n"
            f"  First mismatch at index: "
            f"{int((actual_valid != expected_mask).nonzero(as_tuple=True)[0][0].item()) if not torch.equal(actual_valid, expected_mask) else 'N/A'}",
        )

        # Verify: valid slots have correct values
        for t in range(input_len):
            if not expected_mask[t]:
                self.assertEqual(
                    int(state_slots[t].item()),
                    -1,
                    f"pos={int(positions[t].item())} should be masked but got slot={int(state_slots[t].item())}",
                )
            else:
                pos = int(positions[t].item())
                block_idx = pos // tpb
                block_id = block_idx + 1  # state_bt row 0 = [1, 2, 3, ...]
                in_ring = pos % eb
                expected_slot = block_id * eb + in_ring
                self.assertEqual(
                    int(state_slots[t].item()),
                    expected_slot,
                    f"pos={pos}: expected slot={expected_slot}, got {int(state_slots[t].item())}",
                )

    def _run_batched(
        self,
        ratio: int,
        overlap: int,
        gen_num: int,
        prefix_lengths: list[int],
        input_lengths: list[int],
    ) -> None:
        """Verify state_slots mask for batched prefill."""
        tpb = 256
        eb = _compute_ring_size(ratio, overlap, gen_num)
        n_reqs = len(prefix_lengths)
        max_blocks = max(
            (p + il + tpb - 1) // tpb + 1
            for p, il in zip(prefix_lengths, input_lengths)
        )
        stub = _make_stub(
            self.device,
            ratio=ratio,
            overlap=overlap,
            gen_num=gen_num,
            n_reqs=n_reqs,
            blocks_per_req=max_blocks,
        )

        positions = torch.cat(
            [
                torch.arange(p, p + il, dtype=torch.int64, device=self.device)
                for p, il in zip(prefix_lengths, input_lengths)
            ]
        )
        b_idx = torch.cat(
            [
                torch.full((il,), b, dtype=torch.int64, device=self.device)
                for b, il in enumerate(input_lengths)
            ]
        )
        seq_end_per_req = torch.tensor(
            [p + il for p, il in zip(prefix_lengths, input_lengths)],
            dtype=torch.int64,
            device=self.device,
        )
        cu_seqlens = torch.tensor(
            [0] + [sum(input_lengths[: i + 1]) for i in range(n_reqs)],
            dtype=torch.int32,
            device=self.device,
        )
        seq_start = torch.tensor(prefix_lengths, dtype=torch.int32, device=self.device)

        meta = stub.prepare_metadata(
            positions,
            b_idx,
            is_batched=True,
            seq_start_per_req=seq_start,
            cu_seq_per_req=cu_seqlens,
        )

        expected_mask = _reference_ring_mask(positions, b_idx, seq_end_per_req, tpb, eb)

        state_slots = meta.state_slots
        actual_valid = state_slots >= 0
        self.assertTrue(
            torch.equal(actual_valid, expected_mask),
            f"Batched ring mask mismatch for ratio={ratio}, overlap={overlap}, "
            f"gen_num={gen_num}, prefixes={prefix_lengths}, inputs={input_lengths}, "
            f"eb={eb}\n"
            f"  Expected valid count: {expected_mask.sum().item()}\n"
            f"  Actual valid count: {actual_valid.sum().item()}",
        )

    # ------------------------------------------------------------------
    # CSA tests: ratio=4, overlap=1
    # ------------------------------------------------------------------
    def test_csa_gen0_short_seq(self) -> None:
        """CSA R=8, seq fits in one block."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=20
        )

    def test_csa_gen0_full_block(self) -> None:
        """CSA R=8, exactly one block."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=256
        )

    def test_csa_gen0_multi_block(self) -> None:
        """CSA R=8, spans 3+ blocks."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=700
        )

    def test_csa_gen0_with_prefix(self) -> None:
        """CSA R=8, prefix crosses block boundary."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=200, input_len=300
        )

    def test_csa_gen1_multi_block(self) -> None:
        """CSA R=10 (gen_num=1), multi-block."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=1, prefix_len=0, input_len=600
        )

    def test_csa_gen3_multi_block(self) -> None:
        """CSA R=12 (gen_num=3), multi-block."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=3, prefix_len=100, input_len=500
        )

    def test_csa_gen5_multi_block(self) -> None:
        """CSA R=14 (gen_num=5), multi-block."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=5, prefix_len=0, input_len=1000
        )

    def test_csa_gen0_partial_last_block(self) -> None:
        """CSA R=8, last block is partial (< 256 tokens)."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=260
        )

    # ------------------------------------------------------------------
    # HCA tests: ratio=128, overlap=0
    # ------------------------------------------------------------------
    def test_hca_gen0_short_seq(self) -> None:
        """HCA R=128, seq fits in one block."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=0, input_len=100
        )

    def test_hca_gen0_full_block(self) -> None:
        """HCA R=128, exactly one block."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=0, input_len=256
        )

    def test_hca_gen0_multi_block(self) -> None:
        """HCA R=128, spans 3+ blocks."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=0, input_len=700
        )

    def test_hca_gen0_with_prefix(self) -> None:
        """HCA R=128, prefix crosses block boundary."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=200, input_len=400
        )

    def test_hca_gen1_multi_block(self) -> None:
        """HCA R=130 (gen_num=1), multi-block."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=1, prefix_len=0, input_len=600
        )

    def test_hca_gen3_multi_block(self) -> None:
        """HCA R=132 (gen_num=3), multi-block."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=3, prefix_len=50, input_len=700
        )

    def test_hca_gen5_multi_block(self) -> None:
        """HCA R=134 (gen_num=5), multi-block."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=5, prefix_len=0, input_len=1000
        )

    def test_hca_gen0_partial_last_block(self) -> None:
        """HCA R=128, last block partial — all tokens valid (< R remain)."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=0, input_len=300
        )

    # ------------------------------------------------------------------
    # Batched (B=2) tests
    # ------------------------------------------------------------------
    def test_csa_gen0_batched(self) -> None:
        """CSA R=8, two requests with different lengths."""
        self._run_batched(
            ratio=4,
            overlap=1,
            gen_num=0,
            prefix_lengths=[0, 100],
            input_lengths=[300, 500],
        )

    def test_csa_gen3_batched(self) -> None:
        """CSA R=12, two requests."""
        self._run_batched(
            ratio=4,
            overlap=1,
            gen_num=3,
            prefix_lengths=[50, 200],
            input_lengths=[400, 300],
        )

    def test_hca_gen0_batched(self) -> None:
        """HCA R=128, two requests with different lengths."""
        self._run_batched(
            ratio=128,
            overlap=0,
            gen_num=0,
            prefix_lengths=[0, 128],
            input_lengths=[400, 300],
        )

    def test_hca_gen5_batched(self) -> None:
        """HCA R=134, two requests."""
        self._run_batched(
            ratio=128,
            overlap=0,
            gen_num=5,
            prefix_lengths=[0, 50],
            input_lengths=[600, 800],
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    def test_csa_gen0_exactly_R_tokens(self) -> None:
        """CSA R=8, input_len == R — all tokens should write."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=8
        )

    def test_csa_gen0_less_than_R_tokens(self) -> None:
        """CSA R=8, input_len < R — all tokens should write."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=0, input_len=5
        )

    def test_hca_gen0_seq_end_at_block_boundary(self) -> None:
        """HCA R=128, seq_end exactly at block boundary (512)."""
        self._run_single_request(
            ratio=128, overlap=0, gen_num=0, prefix_len=0, input_len=512
        )

    def test_csa_gen0_prefix_at_block_boundary(self) -> None:
        """CSA R=8, prefix starts at block boundary."""
        self._run_single_request(
            ratio=4, overlap=1, gen_num=0, prefix_len=256, input_len=300
        )


if __name__ == "__main__":
    unittest.main()
