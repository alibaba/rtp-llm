"""Prefix-reuse pool snapshots in one launch: byte-exactness.

The fused prefix-reuse writer used to open its transaction with two separate
snapshot launches (``_prefix_reuse_state_snapshot`` for the state ring and
``_prefix_reuse_kv_snapshot`` for the compressed KV pool).  Both snapshots
read the same frozen cu/prefix tables before any mutation, so a single launch
whose grid spans both index ranges now takes both snapshots: programs below
``PAGES`` run the state-ring body, programs below ``GROUPS`` run the
compressed-KV body.

This test locks the replacement contract against the retired kernel pair
(kept verbatim below as the reference):

* the fused launch's four outputs (state slots, state originals, KV slots,
  KV originals) match the retired two-launch outputs byte for byte across
  batch > 1, page-boundary, partial-page and no-group geometries;
* the writer's snapshot phase is one launch: the whole write enqueues exactly
  three kernels (snapshot + q prep + k compress).
"""

import json
import os
import tempfile
import unittest

import torch
import triton
import triton.language as tl

import indexer_prefix_reuse_triton_test as prefix_reuse_test
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton import (
    INDEXER_HEAD_DIM,
    STATE_RING_ROWS,
    write_prefix_reuse_prefill,
)


@triton.jit
def _state_snapshot_reference(
    cu,
    prefixes,
    table,
    pool,
    slots,
    original,
    WIDTH,
    PAGES,
):
    # Verbatim body of the retired _prefix_reuse_state_snapshot.
    request, page = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    prefix = tl.load(prefixes + request)
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * 128, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    valid = position >= page * 128
    # Prefixes are page aligned, so a local page index addresses the same
    # ring rows as the absolute one; only the table column shifts.
    column = prefix // 128 + page
    block = tl.load(
        table + request * WIDTH + column,
        mask=(page * 128 < length) & (column < WIDTH),
        other=0,
    )
    slot = block * 8 + r
    row = (request * PAGES + page) * 8 + r
    value = tl.load(
        pool + slot[:, None].to(tl.int64) * 128 + d[None, :],
        mask=valid[:, None],
        other=0,
    )
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row[:, None] * 128 + d[None, :], value)


@triton.jit
def _kv_snapshot_reference(
    cu,
    prefixes,
    table,
    pool,
    slots,
    original,
    WIDTH,
    GROUPS,
):
    # Verbatim body of the retired _prefix_reuse_kv_snapshot.
    request, group = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    prefix = tl.load(prefixes + request)
    valid = group < length // 4
    start = prefix + group * 4
    column = start // 128
    block = tl.load(
        table + request * WIDTH + column, mask=valid & (column < WIDTH), other=0
    )
    slot = block * 32 + (start // 4) % 32
    d = tl.arange(0, INDEXER_HEAD_DIM)
    row = request * GROUPS + group
    value = tl.load(pool + slot.to(tl.int64) * 128 + d, mask=valid, other=0)
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row * 128 + d, value)


def _buffers(batch, pages, groups, device):
    state_slots = torch.empty(batch * pages * 8, dtype=torch.int64, device=device)
    kv_slots = torch.empty(batch * groups, dtype=torch.int64, device=device)
    original_state = torch.empty(
        (state_slots.numel(), 128), dtype=torch.float32, device=device
    )
    original_kv = torch.empty(
        (kv_slots.numel(), 128), dtype=torch.bfloat16, device=device
    )
    return state_slots, original_state, kv_slots, original_kv


def _reference_snapshots(args, pages, groups):
    batch = len(args["lengths"])
    state_slots, original_state, kv_slots, original_kv = _buffers(
        batch, pages, groups, "cuda"
    )
    _state_snapshot_reference[(batch, pages)](
        args["cu"],
        args["prefixes_i32"],
        args["state_table"],
        args["state"],
        state_slots,
        original_state,
        WIDTH=args["state_table"].shape[1],
        PAGES=pages,
        num_warps=4,
    )
    if groups:
        _kv_snapshot_reference[(batch, groups)](
            args["cu"],
            args["prefixes_i32"],
            args["kv_table"],
            args["kv"],
            kv_slots,
            original_kv,
            WIDTH=args["kv_table"].shape[1],
            GROUPS=groups,
            num_warps=4,
        )
    return state_slots, original_state, kv_slots, original_kv


def _fused_snapshots(args, pages, groups):
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton import (
        _prefix_reuse_snapshots,
    )

    batch = len(args["lengths"])
    state_slots, original_state, kv_slots, original_kv = _buffers(
        batch, pages, groups, "cuda"
    )
    _prefix_reuse_snapshots[(batch, max(pages, groups))](
        args["cu"],
        args["prefixes_i32"],
        args["state_table"],
        args["state"],
        state_slots,
        original_state,
        args["kv_table"],
        args["kv"],
        kv_slots,
        original_kv,
        STATE_WIDTH=args["state_table"].shape[1],
        KV_WIDTH=args["kv_table"].shape[1],
        PAGES=pages,
        GROUPS=groups,
        num_warps=4,
    )
    return state_slots, original_state, kv_slots, original_kv


class PrefixReuseSnapshotFusionTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_snapshot_matches_retired_pair_bit_for_bit(self):
        harness = prefix_reuse_test.IndexerPrefixReuseTritonTest()
        cases = (
            ([0, 0], [128, 4]),  # page-boundary length; divisible + tiny tail
            ([128, 128], [129, 7]),  # prefix reuse crossing pages, non-divisible
            ([256, 0], [4, 132]),  # mixed hot/cold rows, multi-page maximum
            ([0], [3]),  # every row inside one incomplete group: no KV groups
            ([0, 0], [1, 1]),  # minimal single-row requests, no groups
            ([384, 128, 0], [260, 1, 131]),  # batch > 1, mixed prefixes/lengths
            ([0], [512]),  # four pages in one request
            ([128], [257]),  # three pages with a partial last page
        )
        for prefixes, lengths in cases:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                args = harness.inputs(prefixes, lengths)
                maximum = max(lengths)
                pages = triton.cdiv(maximum, 128)
                groups = maximum // 4
                reference = _reference_snapshots(args, pages, groups)
                fused = _fused_snapshots(args, pages, groups)
                for name, expected, got in zip(
                    (
                        "state_slots",
                        "original_state",
                        "kv_slots",
                        "original_kv",
                    ),
                    reference,
                    fused,
                ):
                    self.assertTrue(
                        torch.equal(expected, got),
                        f"{name} mismatch for prefixes={prefixes} "
                        f"lengths={lengths}",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_write_prefix_reuse_prefill_snapshots_in_one_launch(self):
        from torch.profiler import ProfilerActivity, profile

        harness = prefix_reuse_test.IndexerPrefixReuseTritonTest()
        args = harness.inputs([128, 0], [132, 4])
        # Warm the Triton compiles outside the profile window.
        result = harness.write(args, args["kv"].clone(), args["state"].clone())
        self.assertIsNotNone(result)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = harness.write(args, args["kv"].clone(), args["state"].clone())
            self.assertIsNotNone(result)
            torch.cuda.synchronize()
        path = os.path.join(
            tempfile.mkdtemp(prefix="qsa_prefix_reuse_snapshot_"), "t.json"
        )
        prof.export_chrome_trace(path)
        with open(path) as handle:
            events = json.load(handle)["traceEvents"]
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(
            len(kernels),
            3,
            "the write must enqueue exactly the fused snapshot, q prep and "
            f"k compress launches, got {len(kernels)}: {kernels}",
        )


if __name__ == "__main__":
    unittest.main()
