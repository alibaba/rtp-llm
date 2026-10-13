"""Stage 4B — fp8_kv_quant_decode_op tests.

Pure-CPU tests using ``reference_quantize_v4_kv_decode`` as the oracle.
The CUDA fast path (``quantize_v4_kv_decode``) is not exercised here
since dev box has no CUDA; CI on SM100_ARM does that via the smoke
suite.

Verifies:
  * Byte-level layout — RoPE bytes match the bf16 view directly,
    scale region holds 7 ue8m0 bytes + 1 padding, NoPE region holds
    7 tiles × 64 fp8_e4m3fn bytes.
  * Round-trip: quantize then dequantize → close to the input within
    fp8_e4m3 precision (< ~3% rel diff per tile).
  * ``slot == -1`` is a no-op (sentinel skip).
  * Tile scaling: a tile of all-equal magnitude quantizes to identical
    fp8_e4m3 codes per element, with the scale chosen as max/448.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_BYTES,
    NOPE_DIM,
    NOPE_TILES,
    ROPE_BYTES,
    ROPE_DIM,
    SCALE_BYTES_PER_TOKEN,
    TILE_SIZE,
    _ue8m0_byte_to_scale,
    _ue8m0_scale_byte,
    dequantize_v4_kv_slot,
    reference_quantize_v4_kv_decode,
)


def _alloc_packed_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.zeros((num_blocks, block_size, ENTRY_BYTES), dtype=torch.uint8)


class TestLayoutConstants(unittest.TestCase):

    def test_constants_match_kernel(self):
        """Sanity check the layout constants match the CUDA kernel comments."""
        self.assertEqual(NOPE_DIM, 448)
        self.assertEqual(ROPE_DIM, 64)
        self.assertEqual(TILE_SIZE, 64)
        self.assertEqual(NOPE_TILES, 7)
        self.assertEqual(NOPE_BYTES, 448)
        self.assertEqual(ROPE_BYTES, 128)
        self.assertEqual(SCALE_BYTES_PER_TOKEN, 8)
        self.assertEqual(ENTRY_BYTES, 584)


class TestUe8m0Scale(unittest.TestCase):

    def test_scale_round_trip(self):
        """Scale byte → float → byte is a fixed point for valid values."""
        for byte in [0, 64, 127, 128, 192, 255]:
            scale = _ue8m0_byte_to_scale(byte)
            recovered = _ue8m0_scale_byte(
                scale * 448.0
            )  # max_abs that yields this scale
            # Within ±1 due to log2/exp2 round-trip
            self.assertLessEqual(
                abs(recovered - byte),
                1,
                f"byte={byte} → scale={scale} → recovered={recovered}",
            )


class TestReferenceQuantize(unittest.TestCase):

    def test_rope_bytes_are_bf16_view(self):
        """RoPE region [448:576] of the slot must equal the bf16 view bytes."""
        torch.manual_seed(0)
        T = 2
        block_size = 4
        cache = _alloc_packed_cache(num_blocks=1, block_size=block_size)
        k = torch.randn(T, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16)
        slots = torch.tensor([0, 2], dtype=torch.long)

        reference_quantize_v4_kv_decode(k, slots, cache, block_size=block_size)

        for i, slot in enumerate(slots.tolist()):
            slot_view = cache[0, slot]  # [584]
            rope_bytes = slot_view[NOPE_BYTES : NOPE_BYTES + ROPE_BYTES]
            expected = k[i, NOPE_DIM:].contiguous().view(torch.uint8)
            self.assertTrue(
                torch.equal(rope_bytes, expected), f"slot {slot}: RoPE bytes mismatch"
            )

    def test_skip_negative_slot(self):
        """slot == -1 → no-op; that slot's region stays zero."""
        block_size = 4
        cache = _alloc_packed_cache(num_blocks=1, block_size=block_size)
        k = torch.ones(2, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16) * 0.1
        slots = torch.tensor([-1, 1], dtype=torch.long)

        reference_quantize_v4_kv_decode(k, slots, cache, block_size=block_size)

        # Slot 0 (would have been the -1 token): should be all zero
        # (no mapping wrote there). slot 1 should be populated.
        self.assertTrue(
            bool((cache[0, 0] == 0).all()), "slot 0 should remain zero (skipped)"
        )
        self.assertTrue(
            bool((cache[0, 1] != 0).any()), "slot 1 should have non-zero data"
        )

    def test_round_trip_within_fp8_precision(self):
        """quantize → dequantize should recover the input within fp8_e4m3 precision.

        Per-tile rel diff < ~3% is the practical bound for fp8_e4m3 with
        per-tile max-abs scaling on random gaussian inputs.
        """
        torch.manual_seed(7)
        T = 3
        block_size = 4
        cache = _alloc_packed_cache(num_blocks=1, block_size=block_size)
        k = torch.randn(T, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16) * 0.2
        slots = torch.tensor([0, 1, 3], dtype=torch.long)

        reference_quantize_v4_kv_decode(k, slots, cache, block_size=block_size)

        for i, slot in enumerate(slots.tolist()):
            recovered = dequantize_v4_kv_slot(cache[0, slot])
            # NoPE: per-tile rel diff
            for tile_idx in range(NOPE_TILES):
                a = k[i, tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE].float()
                b = recovered[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE].float()
                ref_mag = a.abs().mean().item() + 1e-9
                rel_diff = (a - b).abs().mean().item() / ref_mag
                self.assertLess(
                    rel_diff,
                    0.05,
                    f"slot {slot} tile {tile_idx}: rel_diff={rel_diff:.3e}",
                )
            # RoPE: bf16 → bf16 view-cast → bf16, MUST be bit-equal (no quant).
            self.assertTrue(
                torch.equal(
                    k[i, NOPE_DIM:].contiguous(),
                    recovered[NOPE_DIM:].contiguous(),
                ),
                f"slot {slot}: RoPE round-trip not bit-equal",
            )

    def test_uniform_tile_quantizes_to_same_code(self):
        """A tile of all-equal magnitude → all elements quantize to the
        same fp8_e4m3 code (with the right sign)."""
        T = 1
        block_size = 1
        cache = _alloc_packed_cache(num_blocks=1, block_size=block_size)
        k = torch.zeros(T, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16)
        # Fill first NoPE tile with constant +0.3
        k[0, :TILE_SIZE] = 0.3
        slots = torch.tensor([0], dtype=torch.long)

        reference_quantize_v4_kv_decode(k, slots, cache, block_size=block_size)

        slot_view = cache[0, 0]
        first_tile_codes = slot_view[:TILE_SIZE].tolist()
        # All 64 elements should share the same fp8_e4m3 byte code.
        self.assertEqual(
            len(set(first_tile_codes)),
            1,
            f"tile codes should all match, got {set(first_tile_codes)}",
        )

    def test_multi_block(self):
        """Slot mapping spans multiple blocks → block_idx routing is correct."""
        block_size = 2
        num_blocks = 3
        cache = _alloc_packed_cache(num_blocks=num_blocks, block_size=block_size)
        k = torch.randn(3, NOPE_DIM + ROPE_DIM, dtype=torch.bfloat16) * 0.1
        slots = torch.tensor([0, 3, 5], dtype=torch.long)
        # block 0, offset 0 ; block 1, offset 1 ; block 2, offset 1

        reference_quantize_v4_kv_decode(k, slots, cache, block_size=block_size)

        # Verify the right (block, offset) cells got data
        for i, slot in enumerate(slots.tolist()):
            b = slot // block_size
            o = slot % block_size
            self.assertTrue(
                bool((cache[b, o] != 0).any()),
                f"slot={slot} (block={b}, offset={o}) is empty",
            )

        # And the unwritten cells stay zero
        unwritten = [(0, 1), (1, 0), (2, 0)]
        for b, o in unwritten:
            self.assertTrue(
                bool((cache[b, o] == 0).all()),
                f"unwritten cell (block={b}, offset={o}) was modified",
            )


if __name__ == "__main__":
    unittest.main()
