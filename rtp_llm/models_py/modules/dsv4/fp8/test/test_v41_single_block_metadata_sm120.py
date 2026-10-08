"""Literal single-block ring slot oracles across raw page boundaries."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import compute_swa_slot_mapping
from rtp_llm.models_py.modules.dsv4.fp8.attention import _build_suffix_pool_slot_mapping


class SingleBlockMetadataTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    def check_writes(self, prefixes, lengths, entries):
        pages = [7, 3, 19, 2, 21, 11, 5, 17, 13][: len(lengths)]
        cu, expected, values = [0], [], []
        for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
            end = prefix + length
            for pos in range(prefix, end):
                expected.append(
                    pages[request] * entries + pos % entries
                    if pos + entries >= end
                    else -1
                )
                values.append(10000 * request + pos)
            cu.append(cu[-1] + length)
        table = torch.tensor(pages, device="cuda", dtype=torch.int32)[:, None]
        actual = compute_swa_slot_mapping(
            table,
            torch.tensor(cu, device="cuda", dtype=torch.int32),
            torch.tensor(
                [a + b for a, b in zip(prefixes, lengths)],
                device="cuda",
                dtype=torch.int32,
            ),
            cu[-1],
            pool_entries_per_block=entries,
            tokens_per_block_for_block_table=256,
            ring_entries=entries,
            single_block_per_request=True,
        )
        self.assertEqual(actual.tolist(), expected)
        valid = actual >= 0
        self.assertEqual(actual[valid].unique().numel(), valid.sum().item())
        pool = torch.full((24 * entries,), -99, device="cuda", dtype=torch.int64)
        payload = torch.tensor(values, device="cuda", dtype=torch.int64)
        pool.index_copy_(0, actual[valid], payload[valid])
        oracle = [-99] * (24 * entries)
        for slot, value in zip(expected, values):
            if slot >= 0:
                oracle[slot] = value
        self.assertEqual(pool.tolist(), oracle)

    def test_cold_final_tail_and_sentinels(self):
        lengths = [127, 128, 129, 255, 256, 257, 511, 512, 513]
        for entries in (128, 130):
            self.check_writes([0] * len(lengths), lengths, entries)

    def test_continuation_final_tail_and_sentinels(self):
        for entries in (128, 130):
            self.check_writes(
                [126, 127, 128, 254, 255, 256, 510, 511, 512],
                [3, 1, 129, 3, 257, 2, 513, 3, 1027],
                entries,
            )

    def test_suffix_reads_from_retained_block(self):
        ends = [0, 127, 128, 255, 256, 511, 512, 513, 1031]
        pages = [7, 3, 0, 2, 21, 11, -1, 17, 13]
        lengths = [min(end, 127) for end in ends]
        for entries in (128, 130):
            for device in ("cpu", "cuda"):
                slots = _build_suffix_pool_slot_mapping(
                    block_table=torch.tensor(pages, dtype=torch.int32, device=device)[
                        :, None
                    ],
                    seq_lens=torch.tensor(ends, dtype=torch.int32, device=device),
                    gather_lens=torch.tensor(lengths, dtype=torch.int32),
                    entries_per_block=entries,
                    tokens_per_block_for_block_table=256,
                    ring_entries=entries,
                    single_block_per_request=True,
                )
                expected = [
                    [
                        (
                            page * entries + (end - length + offset) % entries
                            if offset < length and page > 0
                            else -1
                        )
                        for offset in range(127)
                    ]
                    for page, end, length in zip(pages, ends, lengths)
                ]
                self.assertEqual(slots.tolist(), expected)

    def test_state_final_request_tail_and_snapshot(self):
        ends = [127, 128, 255, 256, 257, 511, 512, 513]
        pages = [7, 3, 19, 2, 21, 11, 5, 17]
        pos, req = [], []
        for request, end in enumerate(ends):
            pos.extend(range(-1, end + 1))
            req.extend([request] * (end + 2))
        pos += [5, 6]
        req += [-1, len(ends)]
        for dtype in (torch.int32, torch.int64):
            positions = torch.tensor(pos, device="cuda", dtype=dtype)
            requests = torch.tensor(req, device="cuda", dtype=dtype)
            table = torch.tensor(pages, device="cuda", dtype=dtype)[:, None]
            for entries in (2, 4):
                for publish_tail in (False, True):
                    slots = _v41_prefill_metadata.try_slot_mapping(
                        positions,
                        requests,
                        table,
                        entries,
                        256,
                        2,
                        state=True,
                        seq_ends=(
                            torch.tensor(ends, device="cuda", dtype=dtype)
                            if publish_tail
                            else None
                        ),
                        single_block_per_request=True,
                    )
                    expected = []
                    for p, r in zip(pos, req):
                        live = 0 <= r < len(ends) and p >= 0
                        if live and publish_tail:
                            live = ends[r] - entries <= p < ends[r]
                        expected.append(
                            pages[r] * entries + p % entries if live else -1
                        )
                    self.assertEqual(slots.tolist(), expected)
                    if publish_tail:
                        active = slots[slots >= 0]
                        self.assertEqual(active.numel(), active.unique().numel())
                        self.assertEqual(active.numel(), len(ends) * entries)

    def test_reject_non_single_block_layout_and_cp(self):
        table = torch.ones((1, 2), device="cuda", dtype=torch.int32)
        positions = torch.zeros(1, device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "one table column"):
            compute_swa_slot_mapping(
                table,
                torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                torch.ones(1, device="cuda", dtype=torch.int32),
                1,
                pool_entries_per_block=128,
                tokens_per_block_for_block_table=256,
                ring_entries=128,
                single_block_per_request=True,
            )
        for state, cp, cols in ((False, 1, 1), (True, 2, 1), (True, 1, 2)):
            self.assertIsNone(
                _v41_prefill_metadata.try_slot_mapping(
                    positions,
                    positions,
                    table[:, :cols],
                    4,
                    256,
                    2,
                    cp,
                    0,
                    state=state,
                    single_block_per_request=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
