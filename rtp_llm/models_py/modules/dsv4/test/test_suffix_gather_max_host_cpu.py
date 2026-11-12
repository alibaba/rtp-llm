"""CPU tests for the ordinary suffix slot-mapping builders.

Extracts ONLY ``_build_suffix_pool_slot_mapping`` and
``_build_suffix_cp_sliced_slot_mapping`` from the production attention.py and
compares their outputs against independent Python coordinate oracles — the
expected slot for every (row, column) is derived in plain Python, never by a
second production builder.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import unittest
from runpy import run_path
from typing import Optional, Tuple

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
ATTN_PATH = DSV4 / "fp8" / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()


def _extract(names):
    tree = ast.parse(ATTN_SRC)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f"helper missing: {names}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    env = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(mod, str(ATTN_PATH), "exec"), env)
    return env


ENV = _extract(
    {
        "_build_suffix_pool_slot_mapping",
        "_build_suffix_cp_sliced_slot_mapping",
    }
)
build_suffix_pool_slot_mapping = ENV["_build_suffix_pool_slot_mapping"]
build_suffix_cp_sliced_slot_mapping = ENV["_build_suffix_cp_sliced_slot_mapping"]

# Fixture geometry: 4-column block tables, 64 entries per block, 256 tokens
# per block-table row (the logical/cache-key block), 64-entry SWA ring.
# Rows cover: a zero gather, two ordinary suffixes, one request whose gather
# overruns the 4-column table (seq=1280), one reclaimed table row with
# negative ids ([-1, -1, 10, 11]), and one short sequence whose gather
# reaches negative absolute positions (seq=2, gather=4).
SEQ_LENS = [20, 800, 1024, 1280, 700, 2]
GATHER_LENS = [0, 32, 64, 300, 250, 4]
BLOCK_TABLE = torch.tensor(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [-1, -1, 10, 11],
        [17, 18, 19, 20],
    ],
    dtype=torch.int32,
)
ENTRIES_PER_BLOCK = 64
TOKENS_PER_BLOCK = 256
RING_ENTRIES = 64
LOCAL_ENTRIES = 16
CP_SIZE = 4


def _oracle_pool(seq, gather, row):
    """Absolute-coordinate oracle for the pool suffix gather.

    Row ``b`` gathers absolute positions ``[seq - gather, seq)``; the slot is
    ``block_id * entries_per_block + pos % ring`` when the position is
    non-negative, inside the table's column capacity, and the block id is
    positive — else ``-1``.
    """
    out = []
    for j in range(gather):
        p = seq - gather + j
        block = p // TOKENS_PER_BLOCK
        if p >= 0 and 0 <= block < len(row) and row[block] > 0:
            out.append(row[block] * ENTRIES_PER_BLOCK + p % RING_ENTRIES)
        else:
            out.append(-1)
    return out


def _oracle_cp(seq, gather, row, cp_rank):
    """Owner-oracle for the CP-sliced local-block suffix gather.

    Same absolute positions as the pool variant, but a column only lands on
    this rank's slice when the ring owner matches (``(p % 64) // 16 ==
    cp_rank``) and the position still belongs to this block's tail write
    (``p + 64 >= min((block + 1) * 256, seq)``). The local slot is
    ``block_id * local_entries + (p % 64) % 16``.
    """
    out = []
    for j in range(gather):
        p = seq - gather + j
        block = p // TOKENS_PER_BLOCK
        in_capacity = p >= 0 and 0 <= block < len(row) and row[block] > 0
        owner = (p % (LOCAL_ENTRIES * CP_SIZE)) // LOCAL_ENTRIES
        tail = p + LOCAL_ENTRIES * CP_SIZE >= min((block + 1) * TOKENS_PER_BLOCK, seq)
        if in_capacity and owner == cp_rank and tail:
            out.append(
                row[block] * LOCAL_ENTRIES
                + (p % (LOCAL_ENTRIES * CP_SIZE)) % LOCAL_ENTRIES
            )
        else:
            out.append(-1)
    return out


def _padded(oracle_row, width):
    """Pad a per-row oracle to the output width with -1 in inactive columns.

    When ``width`` is an explicit capture bound smaller than a row's gather,
    the builder emits only the first ``width`` gather columns — truncate the
    oracle the same way.
    """
    return [row[:width] + [-1] * max(width - len(row), 0) for row in oracle_row]


class SuffixSlotMappingTest(unittest.TestCase):
    def test_pool_slots_match_absolute_coordinate_oracle(self):
        got = build_suffix_pool_slot_mapping(
            block_table=BLOCK_TABLE,
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
            gather_lens=torch.tensor(GATHER_LENS, dtype=torch.int32),
            entries_per_block=ENTRIES_PER_BLOCK,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            ring_entries=RING_ENTRIES,
        )
        width = max(GATHER_LENS)  # no override: the actual maximum
        self.assertEqual(tuple(got.shape), (len(SEQ_LENS), width))
        want = _padded(
            [
                _oracle_pool(seq, gather, BLOCK_TABLE[b].tolist())
                for b, (seq, gather) in enumerate(zip(SEQ_LENS, GATHER_LENS))
            ],
            width,
        )
        self.assertEqual(got.tolist(), want)

    def test_cp_sliced_slots_match_owner_oracle(self):
        for cp_rank in range(CP_SIZE):
            with self.subTest(cp_rank=cp_rank):
                got = build_suffix_cp_sliced_slot_mapping(
                    block_table=BLOCK_TABLE,
                    seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
                    gather_lens=torch.tensor(GATHER_LENS, dtype=torch.int32),
                    local_entries_per_block=LOCAL_ENTRIES,
                    tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
                    cp_rank=cp_rank,
                    cp_size=CP_SIZE,
                )
                width = max(GATHER_LENS)
                self.assertEqual(tuple(got.shape), (len(SEQ_LENS), width))
                want = _padded(
                    [
                        _oracle_cp(seq, gather, BLOCK_TABLE[b].tolist(), cp_rank)
                        for b, (seq, gather) in enumerate(zip(SEQ_LENS, GATHER_LENS))
                    ],
                    width,
                )
                self.assertEqual(got.tolist(), want)

    def test_capacity_and_reclaimed_columns_are_masked(self):
        # Row 3 (seq=1280): positions >= 4*256 index blocks past the
        # 4-column table -> -1. Row 4 (reclaimed [-1,-1,10,11]): positions in
        # blocks 0/1 read negative ids -> -1; blocks 2/3 stay live. Row 5
        # (seq=2, gather=4): p in {-2,-1} -> -1, p in {0,1} live.
        got = build_suffix_pool_slot_mapping(
            block_table=BLOCK_TABLE,
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
            gather_lens=torch.tensor(GATHER_LENS, dtype=torch.int32),
            entries_per_block=ENTRIES_PER_BLOCK,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            ring_entries=RING_ENTRIES,
        )
        # 1280-token row: only p in [980, 1024) sits in the table.
        self.assertTrue((got[3, 44:] == -1).all())
        self.assertTrue((got[3, :44] > 0).all())
        # Reclaimed row: block-1 columns masked, block-2 columns live.
        row4 = _oracle_pool(700, 250, [-1, -1, 10, 11])
        self.assertEqual(got[4, :250].tolist(), row4)
        self.assertTrue((got[4, :62] == -1).all())
        self.assertTrue((got[4, 62:250] > 0).all())
        # Short row: the two leading negative positions are masked.
        self.assertEqual(got[5, :2].tolist(), [-1, -1])
        self.assertEqual(got[5, 2:4].tolist(), [17 * 64 + 0, 17 * 64 + 1])

    def test_explicit_capture_width_preserves_padding(self):
        # Capture sizing passes an explicit max_gather (e.g. window-1=127);
        # the builder returns exactly that width and leaves columns past the
        # per-request gather as -1 padding.
        got = build_suffix_pool_slot_mapping(
            block_table=BLOCK_TABLE,
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
            gather_lens=torch.tensor(GATHER_LENS, dtype=torch.int32),
            entries_per_block=ENTRIES_PER_BLOCK,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            ring_entries=RING_ENTRIES,
            max_gather=127,
        )
        self.assertEqual(tuple(got.shape), (len(SEQ_LENS), 127))
        want = _padded(
            [
                _oracle_pool(seq, gather, BLOCK_TABLE[b].tolist())
                for b, (seq, gather) in enumerate(zip(SEQ_LENS, GATHER_LENS))
            ],
            127,
        )
        self.assertEqual(got.tolist(), want)
        # The CP-sliced builder takes no width override: it always uses the
        # actual maximum gather length.
        params = inspect.signature(build_suffix_cp_sliced_slot_mapping).parameters
        self.assertNotIn("max_gather", params)
        # Degenerate batches: empty input -> (0, 0); all-zero gathers ->
        # (B, 0). Both builders agree on the empty-width contract.
        empty = build_suffix_pool_slot_mapping(
            block_table=BLOCK_TABLE[:0],
            seq_lens=torch.zeros(0, dtype=torch.int32),
            gather_lens=torch.zeros(0, dtype=torch.int32),
            entries_per_block=ENTRIES_PER_BLOCK,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            ring_entries=RING_ENTRIES,
        )
        self.assertEqual(tuple(empty.shape), (0, 0))
        zero = build_suffix_pool_slot_mapping(
            block_table=BLOCK_TABLE,
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
            gather_lens=torch.zeros(len(SEQ_LENS), dtype=torch.int32),
            entries_per_block=ENTRIES_PER_BLOCK,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            ring_entries=RING_ENTRIES,
        )
        self.assertEqual(tuple(zero.shape), (len(SEQ_LENS), 0))
        zero_cp = build_suffix_cp_sliced_slot_mapping(
            block_table=BLOCK_TABLE,
            seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
            gather_lens=torch.zeros(len(SEQ_LENS), dtype=torch.int32),
            local_entries_per_block=LOCAL_ENTRIES,
            tokens_per_block_for_block_table=TOKENS_PER_BLOCK,
            cp_rank=0,
            cp_size=CP_SIZE,
        )
        self.assertEqual(tuple(zero_cp.shape), (len(SEQ_LENS), 0))


def setUpModule():
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
