"""Contracts for bounded CP source cohorts and zigzag row restoration."""

import bisect
import random
import types
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41 as STREAM


def _case(lengths):
    """Build CP4 packed rows independently of the source cohort planner."""
    chunks = tuple(2 * ((length + 7) // 8) for length in lengths)
    parts = [[] for _ in range(4)]
    base = 0
    for length, chunk in zip(lengths, chunks):
        padded = torch.full((chunk * 4,), -1, dtype=torch.long)
        padded[:length] = torch.arange(base, base + length)
        halves = padded.reshape(8, chunk // 2)
        for rank in range(4):
            parts[rank].extend((halves[rank], halves[7 - rank]))
        base += length
    packed = [torch.cat(part) for part in parts]
    restore = torch.empty(base, dtype=torch.long)
    all_packed = torch.cat(packed)
    valid = all_packed >= 0
    restore[all_packed[valid]] = torch.arange(len(all_packed))[valid]
    raw = torch.arange(base, dtype=torch.long)
    values = torch.stack([(raw // (128**column)) % 128 for column in range(4)], 1)
    expected = values.bfloat16()
    local = [
        expected[rows.clamp_min(0)].masked_fill((rows < 0)[:, None], -999)
        for rows in packed
    ]
    cp = types.SimpleNamespace(
        cp_size=4,
        chunk_lengths_per_req=chunks,
        input_lengths_global_host=tuple(lengths),
        chunk_length=sum(chunks),
        padded_seq_len=4 * sum(chunks),
        seq_len_full=base,
        unpad_restore=restore,
    )
    return cp, local, expected


class CPSourceCohortPlanCPU(unittest.TestCase):
    def test_other_cp_sizes_keep_global_order(self):
        lengths = (1, 129, 65537, 131071, 10000000)
        for cp_size in (2, 8):
            chunks = tuple(
                2 * ((length + 2 * cp_size - 1) // (2 * cp_size)) for length in lengths
            )
            cp = types.SimpleNamespace(
                cp_size=cp_size,
                input_lengths_global_host=lengths,
                chunk_lengths_per_req=chunks,
                seq_len_full=sum(lengths),
                chunk_length=sum(chunks),
            )
            plan = STREAM._prefill_x_cohort_plan(cp, projected_row_bytes=4096)
            self.assertEqual(plan[0][2], 0)
            self.assertEqual(plan[-1][3], sum(lengths))
            self.assertTrue(all(a[3] == b[2] for a, b in zip(plan, plan[1:])))
            self.assertTrue(
                all(
                    cp_size * (local_end - local_start) * 4096
                    <= STREAM._CP_SOURCE_RECEIVE_BYTES
                    for local_start, local_end, _, _ in plan
                )
            )

    def test_cohorts_cover_global_order_and_local_sources_with_bounded_memory(self):
        rng = random.Random(20261105)
        cases = [
            (1,),
            (129,) * 8,
            (3330, 3331, 3332, 3333, 3334, 3330, 3331, 3332, 3333),
            (1024,) * 32,
            (1024,) * 64,
            (8192,) * 8,
            (131071,),
            (131073,),
            (262144,),
            (1048576,),
            (10000000,),
            (1,) * 1024,
        ]
        cases.extend(
            tuple(rng.randrange(1, 131073) for _ in range(rng.randrange(1, 65)))
            for _ in range(50)
        )
        for lengths in cases:
            with self.subTest(batch=len(lengths), fresh=sum(lengths)):
                chunks = tuple(2 * ((n + 7) // 8) for n in lengths)
                cp = types.SimpleNamespace(
                    cp_size=4,
                    input_lengths_global_host=lengths,
                    chunk_lengths_per_req=chunks,
                    seq_len_full=sum(lengths),
                    chunk_length=sum(chunks),
                )
                global_ends = []
                local_starts = []
                global_cursor = local_cursor = 0
                for length, chunk in zip(lengths, chunks):
                    local_starts.append(local_cursor)
                    global_cursor += length
                    local_cursor += chunk
                    global_ends.append(global_cursor)
                plan = STREAM._prefill_x_cohort_plan(cp, projected_row_bytes=4096)
                self.assertTrue(plan)
                self.assertEqual(plan[0][2], 0)
                self.assertEqual(plan[-1][3], cp.seq_len_full)
                for index, (
                    local_start,
                    local_end,
                    global_start,
                    global_end,
                ) in enumerate(plan):
                    self.assertGreater(local_end, local_start)
                    self.assertGreater(global_end, global_start)
                    self.assertLessEqual(
                        cp.cp_size * (local_end - local_start) * 4096,
                        STREAM._CP_SOURCE_RECEIVE_BYTES,
                    )
                    if index:
                        self.assertEqual(global_start, plan[index - 1][3])
                    positions = [global_start, global_end - 1]
                    if global_end - global_start > 2:
                        positions.append((global_start + global_end) // 2)
                    for position in positions:
                        request = bisect.bisect_right(global_ends, position)
                        request_start = global_ends[request - 1] if request else 0
                        fresh = position - request_start
                        half = chunks[request] // 2
                        segment, offset = divmod(fresh, half)
                        local = local_starts[request] + offset
                        if segment >= cp.cp_size:
                            local += half
                        self.assertTrue(local_start <= local < local_end)

    def test_long_request_keeps_global_order_and_bounded_tiles(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_producer_metadata import (
            raw_tile_plan,
        )

        for length in (131073, 159357, 262144, 1048576, 10000000):
            chunk = 2 * ((length + 7) // 8)
            cp = types.SimpleNamespace(
                cp_size=4,
                input_lengths_global_host=(length,),
                chunk_lengths_per_req=(chunk,),
                seq_len_full=length,
                chunk_length=chunk,
            )
            for ratio, row_bytes in ((1, 1024), (2, 4096)):
                with self.subTest(length=length, ratio=ratio):
                    plan = STREAM._prefill_x_cohort_plan(
                        cp, projected_row_bytes=row_bytes
                    )
                    self.assertEqual(plan[0][2], 0)
                    self.assertEqual(plan[-1][3], length)
                    self.assertTrue(all(a[3] == b[2] for a, b in zip(plan, plan[1:])))
                    for local_begin, local_end, global_begin, global_end in plan:
                        self.assertLessEqual(
                            4 * (local_end - local_begin) * row_bytes,
                            STREAM._CP_SOURCE_RECEIVE_BYTES,
                        )
                    if length == 1048576:
                        self.assertEqual(len(plan), 16 if ratio == 1 else 56)
                    source_ranges = list(STREAM._prefill_source_tile_ranges(plan))
                    self.assertTrue(
                        all(0 < end - start <= 65536 for start, end in source_ranges)
                    )
                    self.assertEqual(source_ranges[0][0], 0)
                    self.assertEqual(source_ranges[-1][1], length)
                    self.assertTrue(
                        all(
                            a[1] == b[0]
                            for a, b in zip(source_ranges, source_ranges[1:])
                        )
                    )
                    for rank in range(4):
                        tiles = list(
                            raw_tile_plan(
                                (length,),
                                (31744,),
                                ratio,
                                4,
                                rank,
                                256,
                                ranges=iter(source_ranges),
                            )
                        )
                        self.assertEqual(
                            [(a, b) for a, b, _ in tiles],
                            source_ranges,
                        )

    def test_replay_layout_is_rejected_before_collective(self):
        cp, _, _ = _case((129, 1024))
        cp.swa_replay_start = 1
        with self.assertRaisesRegex(ValueError, "full zigzag"):
            STREAM._prefill_x_cohort_plan(cp, projected_row_bytes=4096)


if __name__ == "__main__":
    unittest.main()
