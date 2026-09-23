"""Precision UT for ``pack_slots_to_paged``.

Checked against a vectorized torch footer gather. Invalid slots (``-1``)
become zeros and remap to 0; the unused tail of the last dest page stays
zero.

Run:
  CUDA_VISIBLE_DEVICES=1 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.fp8.test.test_pack_slots_to_paged
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    TOKEN_DATA_SIZE,
    pack_slots_to_paged,
)

TMA_ALIGN = 576
SCALE_BYTES = 8


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _tma_stride(page: int) -> int:
    return _align_up(page * ENTRY_BYTES, TMA_ALIGN)


def _make_padded_cache(
    num_blocks: int, page: int, device: torch.device
) -> torch.Tensor:
    stride = _tma_stride(page)
    backing = torch.zeros((num_blocks, stride), dtype=torch.uint8, device=device)
    cache = backing.as_strided(
        (num_blocks, page, ENTRY_BYTES),
        (stride, ENTRY_BYTES, 1),
    )
    assert int(cache.stride(0)) == stride
    return cache


def _as_block_bytes(cache: torch.Tensor) -> torch.Tensor:
    num_blocks = int(cache.shape[0])
    stride = int(cache.stride(0))
    return torch.as_strided(cache, (num_blocks, stride), (stride, 1))


def _pack_slots_to_paged_torch(
    k_cache: torch.Tensor,
    slot_indices: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized footer gather matching ``pack_slots_to_paged``."""
    slots = slot_indices.reshape(-1).to(dtype=torch.int64)
    src_page = int(k_cache.shape[1])
    src = _as_block_bytes(k_cache)
    slot_count = int(slots.numel())
    dst_pages = max((slot_count + page_size - 1) // page_size, 1)
    paged = torch.zeros(
        (dst_pages, page_size, ENTRY_BYTES),
        dtype=torch.uint8,
        device=k_cache.device,
    )
    dst = _as_block_bytes(paged)

    rows = torch.arange(slot_count, device=k_cache.device, dtype=torch.int64)
    valid = slots >= 0
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    src_block = safe // src_page
    src_pos = safe - src_block * src_page
    dst_block = rows // page_size
    dst_pos = rows - dst_block * page_size

    data_off = torch.arange(TOKEN_DATA_SIZE, device=k_cache.device, dtype=torch.int64)
    src_data = src[src_block.unsqueeze(1), src_pos.unsqueeze(1) * TOKEN_DATA_SIZE + data_off]
    src_data = torch.where(valid.unsqueeze(1), src_data, torch.zeros_like(src_data))
    dst[dst_block.unsqueeze(1), dst_pos.unsqueeze(1) * TOKEN_DATA_SIZE + data_off] = src_data

    scale_off = torch.arange(SCALE_BYTES, device=k_cache.device, dtype=torch.int64)
    src_scale = src[
        src_block.unsqueeze(1),
        src_page * TOKEN_DATA_SIZE + src_pos.unsqueeze(1) * SCALE_BYTES + scale_off,
    ]
    src_scale = torch.where(valid.unsqueeze(1), src_scale, torch.zeros_like(src_scale))
    dst[
        dst_block.unsqueeze(1),
        page_size * TOKEN_DATA_SIZE + dst_pos.unsqueeze(1) * SCALE_BYTES + scale_off,
    ] = src_scale

    remapped = torch.where(
        valid, rows.to(torch.int32), torch.zeros((), dtype=torch.int32, device=k_cache.device)
    )
    return paged, remapped.view(slot_indices.shape)


def _fill_unique_footer(cache: torch.Tensor) -> None:
    """Paint every footer slot with a unique, pad-visible pattern."""
    num_blocks, page, _ = cache.shape
    backing = _as_block_bytes(cache)
    backing.fill_(0xFF)
    slots = torch.arange(num_blocks * page, device=cache.device, dtype=torch.int64)
    blocks = slots // page
    pos = slots - blocks * page
    data_off = torch.arange(TOKEN_DATA_SIZE, device=cache.device, dtype=torch.int64)
    data = ((data_off.unsqueeze(0) + slots.unsqueeze(1) * 13) % 251).to(torch.uint8)
    data[:, 0] = (slots & 0xFF).to(torch.uint8)
    data[:, 1] = ((slots >> 8) & 0xFF).to(torch.uint8)
    backing[blocks.unsqueeze(1), pos.unsqueeze(1) * TOKEN_DATA_SIZE + data_off] = data
    scale_off = torch.arange(SCALE_BYTES, device=cache.device, dtype=torch.int64)
    scale = ((scale_off.unsqueeze(0) + slots.unsqueeze(1) * 7 + 3) % 251).to(torch.uint8)
    backing[
        blocks.unsqueeze(1),
        page * TOKEN_DATA_SIZE + pos.unsqueeze(1) * SCALE_BYTES + scale_off,
    ] = scale


class PackSlotsToPagedPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _assert_matches_reference(
        self,
        *,
        src_page: int,
        dst_page: int,
        num_blocks: int,
        slots: torch.Tensor,
    ) -> None:
        cache = _make_padded_cache(num_blocks, src_page, self.device)
        _fill_unique_footer(cache)
        paged, remapped = pack_slots_to_paged(cache, slots, dst_page)
        ref_paged, ref_remap = _pack_slots_to_paged_torch(cache, slots, dst_page)
        self.assertEqual(tuple(paged.shape), tuple(ref_paged.shape))
        self.assertTrue(
            torch.equal(paged, ref_paged),
            msg=(
                f"paged bytes differ: src_page={src_page} dst_page={dst_page} "
                f"slots={tuple(slots.shape)} max|diff|="
                f"{(paged.int() - ref_paged.int()).abs().max().item()}"
            ),
        )
        self.assertTrue(
            torch.equal(remapped, ref_remap),
            msg=f"remap differs: got={remapped.tolist()} ref={ref_remap.tolist()}",
        )

    def test_swa_tma_padded_128_to_64(self) -> None:
        slots = torch.tensor(
            [
                [0, 1, 2, 17, 127, 128, 200, -1],
                [3, -1, 64, 65, 255, 4, 5, 6],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self._assert_matches_reference(
            src_page=128,
            dst_page=64,
            num_blocks=3,
            slots=slots,
        )
        self.assertEqual(_tma_stride(128), 74880)

    def test_csa_page64_and_hca_page2(self) -> None:
        csa_slots = torch.arange(130, dtype=torch.int32, device=self.device)
        csa_slots[17] = -1
        self._assert_matches_reference(
            src_page=64,
            dst_page=64,
            num_blocks=4,
            slots=csa_slots,
        )
        self.assertEqual(_tma_stride(64), 37440)

        hca_slots = torch.tensor(
            [0, 1, 3, -1, 5], dtype=torch.int32, device=self.device
        )
        self._assert_matches_reference(
            src_page=2,
            dst_page=2,
            num_blocks=4,
            slots=hca_slots,
        )

    def test_random_slots_non_multiple_last_page(self) -> None:
        src_page, dst_page, num_blocks = 128, 64, 4
        n_slots = 200
        slots = torch.randint(
            -1, num_blocks * src_page, (n_slots,), dtype=torch.int32, device=self.device
        )
        self._assert_matches_reference(
            src_page=src_page,
            dst_page=dst_page,
            num_blocks=num_blocks,
            slots=slots,
        )

    def test_row_lens_skips_capture_tail(self) -> None:
        src_page, dst_page, num_blocks = 2, 2, 16
        rows, width, used = 4, 32, 5
        slots = torch.full((rows, width), -1, dtype=torch.int32, device=self.device)
        slots[:, :used] = torch.arange(
            rows * used, device=self.device, dtype=torch.int32
        ).view(rows, used)
        cache = _make_padded_cache(num_blocks, src_page, self.device)
        _fill_unique_footer(cache)
        lens = torch.full((rows,), used, dtype=torch.int32, device=self.device)
        paged, remapped = pack_slots_to_paged(
            cache, slots, dst_page, row_lens=lens
        )
        ref_paged, ref_remap = _pack_slots_to_paged_torch(cache, slots, dst_page)
        self.assertTrue(torch.equal(remapped[:, :used], ref_remap[:, :used]))
        self.assertTrue(
            torch.equal(remapped[:, used:], torch.zeros_like(remapped[:, used:]))
        )
        idx = (
            torch.arange(rows, device=self.device)[:, None] * width
            + torch.arange(used, device=self.device)
        ).reshape(-1)
        got = _as_block_bytes(paged)
        ref = _as_block_bytes(ref_paged)
        blocks = idx // dst_page
        pos = idx - blocks * dst_page
        data_off = torch.arange(TOKEN_DATA_SIZE, device=self.device)
        self.assertTrue(
            torch.equal(
                got[blocks.unsqueeze(1), pos.unsqueeze(1) * TOKEN_DATA_SIZE + data_off],
                ref[blocks.unsqueeze(1), pos.unsqueeze(1) * TOKEN_DATA_SIZE + data_off],
            )
        )


if __name__ == "__main__":
    unittest.main()
