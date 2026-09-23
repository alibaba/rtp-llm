"""CPU-only rejecting contract tests for the default-off compact CP seam."""

import hashlib
import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

MODULE = pathlib.Path(__file__).parents[1] / "_compact_cp.py"
spec = importlib.util.spec_from_file_location("compact_cp_contract", MODULE)
compact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compact
spec.loader.exec_module(compact)


def zigzag(start, rank, local=1024):
    half, total = local // 2, local * 4
    return list(range(start + rank * half, start + (rank + 1) * half)) + list(
        range(start + total - (rank + 1) * half, start + total - rank * half)
    )


def geometry(chunk, length=4096):
    return {rank: zigzag(chunk * length, rank, length // 4) for rank in range(4)}


def context(chunk, rank, length=4096):
    # Deliberately CPContext-shaped: no invented current_chunk_start field.
    values = zigzag(chunk * length, rank, length // 4)
    return SimpleNamespace(
        cp_size=4,
        cp_rank=rank,
        global_positions=values,
        local_is_real=[True] * (length // 4),
        padded_seq_len=length,
        prefix_length=chunk * length,
        seq_len_full=length,
        kv_cache_sharded=False,
        batch_size=1,
    )


def pools(kind, rank, length=4096, chunk=0, capacity=4096):
    ratio = 4 if kind == "CSA" else 128
    tags = (
        (compact.CSA_INDEXER, compact.CSA_MAIN)
        if kind == "CSA"
        else (compact.HCA_MAIN,)
    )
    boundaries = [
        p for p in range(chunk * length, (chunk + 1) * length) if (p + 1) % ratio == 0
    ]
    table = {p: i + rank * 1024 for i, p in enumerate(boundaries)}
    return {
        tag: compact.CompactPoolSpec(
            tag,
            capacity,
            *((128, 4) if tag == compact.CSA_INDEXER else (576, 8)),
            table,
            f"rank{rank}-g1",
            f"rank{rank}-store",
        )
        for tag in tags
    }


def build(chunk=0, rank=0, kind="CSA", length=4096, stage_id=0, **kwargs):
    ps = pools(kind, rank, length, chunk)
    ratio = 4 if kind == "CSA" else 128
    local_rows = length // ratio // 4
    workspace = sum(
        (p.data_row_bytes + p.scale_row_bytes) * local_rows for p in ps.values()
    )
    group = tuple(range(0, 4) if stage_id == 0 else range(4, 8))
    history = geometry(chunk - 1, length) if chunk else None
    return compact.build_compact_cp_plan(
        context(chunk, rank, length),
        forward_id=f"f{chunk}",
        stage_id=stage_id,
        layer_id=0,
        attention_type=kind,
        pools=ps,
        cp_group=group,
        request_id=77,
        request_start=0,
        rank_global_positions=geometry(chunk, length),
        history_rank_global_positions=history,
        feature_requested=True,
        workspace_bytes=workspace,
        **kwargs,
    )


def digest(pools_by_tag):
    return {
        tag: hashlib.sha256(pool.data + pool.scales).digest()
        for tag, pool in pools_by_tag.items()
    }


class CompactCPContractTest(unittest.TestCase):
    def test_default_off_and_explicit_opt_in(self):
        kwargs = dict(
            forward_id="off",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=pools("CSA", 0),
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=geometry(0),
            workspace_bytes=400000,
        )
        off = compact.build_compact_cp_plan(context(0, 0), **kwargs)
        self.assertFalse(off.eligible)
        self.assertEqual("FEATURE_DEFAULT_OFF", off.fallback_reason)
        self.assertTrue(build().eligible)

    def test_all_eight_chunks_cp4_both_stage_rosters_and_history(self):
        for chunk in range(8):
            for stage in (0, 1):
                for rank in range(4):
                    plan = build(chunk, rank, stage_id=stage)
                    self.assertTrue(plan.eligible, plan.fallback_reason)
                    self.assertEqual(2, len(plan.intervals))
                    self.assertEqual(2, len(plan.published_tails))
                    self.assertEqual(
                        (0, 1, 2, 3) if stage == 0 else (4, 5, 6, 7), plan.stage_group
                    )
                    self.assertEqual(plan.stage_group, plan.cp_group)
                    self.assertEqual(
                        (compact.CSA_INDEXER, compact.CSA_MAIN),
                        tuple(x.pool_tag for x in plan.schedule),
                    )
                    self.assertEqual(bool(chunk), bool(plan.same_request_history))

    def test_foreign_request_ratio_and_nonowner_pack_are_rejected_without_output_change(
        self,
    ):
        plan = build()
        bytes_by_tag = {
            tag: compact.SplitPoolBytes.poison(pool) for tag, pool in plan.pools
        }
        before = digest(bytes_by_tag)
        foreign = compact.LogicalCompressedRow(
            999, plan.source_rows[0].absolute_boundary, compact.CSA_INDEXER, 4
        )
        adapter = compact.CompactCPTransportAdapter(plan, bytes_by_tag)
        with self.assertRaises(compact.CompactCPIneligible):
            adapter.pack((foreign,), {foreign: 1})
        self.assertEqual(before, digest(bytes_by_tag))
        self.assertEqual("released", adapter.lease.state)
        ratio_bad = compact.LogicalCompressedRow(
            77, plan.source_rows[0].absolute_boundary, compact.CSA_INDEXER, 128
        )
        self.assertRaises(compact.CompactCPIneligible, plan.receiver_slot, ratio_bad)

    def test_scatter_atomic_rejects_duplicate_incomplete_foreign_and_bad_width(self):
        plan = build(0, 1, "HCA")
        bytes_by_tag = {
            tag: compact.SplitPoolBytes.poison(pool) for tag, pool in plan.pools
        }
        good = tuple(
            compact.PackedRow(
                row, bytes([row.absolute_boundary % 251]) * 576, b"\x7f" * 8
            )
            for row in plan.receive_rows
        )
        foreign = compact.PackedRow(
            compact.LogicalCompressedRow(
                999, good[-1].identity.absolute_boundary, compact.HCA_MAIN, 128
            ),
            good[-1].data,
            good[-1].scales,
        )
        for bad in (
            good[:-1],
            good + (good[0],),
            tuple(list(good[:-1]) + [foreign]),
            tuple(list(good[:-1]) + [compact.PackedRow(good[-1].identity, b"x", b"y")]),
        ):
            adapter = compact.CompactCPTransportAdapter(plan, bytes_by_tag)
            before = digest(bytes_by_tag)
            adapter.accept_gathered()
            with self.assertRaises(compact.CompactCPIneligible):
                adapter.scatter(bad)
            self.assertEqual(before, digest(bytes_by_tag))
            self.assertEqual("released", adapter.lease.state)
        adapter = compact.CompactCPTransportAdapter(plan, bytes_by_tag)
        adapter.accept_gathered()
        adapter.scatter(good)
        adapter.release_after_consumer()
        self.assertEqual(
            (bytes([plan.receive_rows[0].absolute_boundary % 251]) * 576, b"\x7f" * 8),
            bytes_by_tag[compact.HCA_MAIN].read(
                plan.receiver_slot(plan.receive_rows[0])
            ),
        )

    def test_table_snapshot_valid_slot_zero_and_distinct_receiver_slots(self):
        original = pools("CSA", 0)
        plan = compact.build_compact_cp_plan(
            context(0, 0),
            forward_id="m",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=original,
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=geometry(0),
            feature_requested=True,
            workspace_bytes=400000,
        )
        row = plan.receive_rows[0]
        slot = plan.receiver_slot(row)
        self.assertEqual(0, slot)
        original[row.pool_tag].receiver_block_table[row.absolute_boundary] = 999
        self.assertEqual(slot, plan.receiver_slot(row))
        for tag in (compact.CSA_INDEXER, compact.CSA_MAIN):
            slots = [
                slot
                for identity, slot in plan.receiver_slots
                if identity.pool_tag == tag
            ]
            self.assertEqual(len(slots), len(set(slots)))

    def test_topology_local_geometry_and_history_are_rejecting(self):
        plan = compact.build_compact_cp_plan(
            context(0, 0),
            forward_id="bad",
            stage_id=1,
            layer_id=0,
            attention_type="CSA",
            pools=pools("CSA", 0),
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=geometry(0),
            feature_requested=True,
            workspace_bytes=400000,
        )
        self.assertEqual("UNSUPPORTED_CP_TOPOLOGY", plan.fallback_reason)
        peer = geometry(0)
        peer[0] = tuple(reversed(peer[0]))
        plan = compact.build_compact_cp_plan(
            context(0, 0),
            forward_id="badlocal",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=pools("CSA", 0),
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=peer,
            feature_requested=True,
            workspace_bytes=400000,
        )
        self.assertEqual("LOCAL_CONTEXT_PEER_GEOMETRY_MISMATCH", plan.fallback_reason)
        no_history = compact.build_compact_cp_plan(
            context(1, 0),
            forward_id="hist",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=pools("CSA", 0, chunk=1),
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=geometry(1),
            feature_requested=True,
            workspace_bytes=400000,
        )
        self.assertEqual(
            "MISSING_SAME_REQUEST_HISTORY_GEOMETRY", no_history.fallback_reason
        )

    def test_reconciliation_requires_all_peers_and_exposes_peer_bad_table(self):
        plans = [build(0, rank) for rank in range(4)]
        good = compact.reconcile_compact_cp_proposals(
            [compact.proposal_summary(x) for x in plans], cp_group=(0, 1, 2, 3)
        )
        self.assertTrue(good.agreed and good.eligible)
        bad_pool = pools("CSA", 3)
        bad_pool[compact.CSA_MAIN].receiver_block_table[3] = -1
        peer_bad = compact.build_compact_cp_plan(
            context(0, 3),
            forward_id="f0",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=bad_pool,
            cp_group=(0, 1, 2, 3),
            request_id=77,
            request_start=0,
            rank_global_positions=geometry(0),
            feature_requested=True,
            workspace_bytes=400000,
        )
        self.assertFalse(peer_bad.eligible)
        merged = compact.reconcile_compact_cp_proposals(
            [compact.proposal_summary(x) for x in plans[:3]]
            + [compact.proposal_summary(peer_bad)],
            cp_group=(0, 1, 2, 3),
        )
        self.assertTrue(merged.agreed)
        self.assertFalse(merged.eligible)
        self.assertEqual("PEER_INELIGIBLE", merged.reason)
        missing = compact.reconcile_compact_cp_proposals(
            [compact.proposal_summary(x) for x in plans[:3]], cp_group=(0, 1, 2, 3)
        )
        self.assertFalse(missing.agreed)
        self.assertEqual("MISSING_PEER_PROPOSAL", missing.reason)

    def test_peer_epoch_request_shape_and_wire_mismatch(self):
        good = [compact.proposal_summary(build(0, rank)) for rank in range(4)]
        for key, value in (
            ("forward_id", "other"),
            ("request_id", 99),
            ("layer_id", 1),
            ("chunk_start", 4096),
        ):
            changed = compact.proposal_summary(replace(build(0, 3), **{key: value}))
            result = compact.reconcile_compact_cp_proposals(
                good[:3] + [changed], cp_group=(0, 1, 2, 3)
            )
            self.assertFalse(result.agreed)
            self.assertEqual(result.reason, "FORWARD_IDENTITY_MISMATCH")
        for changed in (
            replace(good[3], wire_layouts=(("CSA_KV", 7, 4),)),
            replace(good[3], receive_rows=good[3].receive_rows[:-1]),
        ):
            result = compact.reconcile_compact_cp_proposals(
                good[:3] + [changed], cp_group=(0, 1, 2, 3)
            )
            self.assertFalse(result.agreed)
            self.assertEqual(result.reason, "WIRE_LAYOUT_OR_ROWS_MISMATCH")

    def test_typed_status_identity_and_resized_storage(self):
        plan = build()
        with self.assertRaises(compact.CompactCPIneligible):
            compact.LogicalCompressedRow(77, 3, compact.CSA_MAIN, 4.0)
        pools_by_tag = {
            tag: compact.SplitPoolBytes.poison(spec) for tag, spec in plan.pools
        }
        before = digest(pools_by_tag)
        with self.assertRaises(compact.CompactCPIneligible):
            compact.CompactCPTransportAdapter(plan, pools_by_tag).pack(
                plan.source_rows, {row: True for row in plan.source_rows}
            )
        self.assertEqual(before, digest(pools_by_tag))
        pools_by_tag[compact.CSA_MAIN].data.pop()
        before = digest(pools_by_tag)
        with self.assertRaises(compact.CompactCPIneligible):
            compact.CompactCPTransportAdapter(plan, pools_by_tag).pack(
                plan.source_rows, {row: 1 for row in plan.source_rows}
            )
        self.assertEqual(before, digest(pools_by_tag))

    def test_tail_cp1_and_guarded_fallback(self):
        self.assertTrue(build(0, 0, "CSA", 3616).eligible)
        self.assertEqual(
            "HCA_TAIL_OR_UNALIGNED_WINDOW", build(0, 0, "HCA", 3616).fallback_reason
        )
        ctx = context(0, 0)
        ctx.cp_size = 1
        ctx.cp_rank = 0
        cp1 = compact.build_compact_cp_plan(
            ctx,
            forward_id="cp1",
            stage_id=0,
            layer_id=0,
            attention_type="CSA",
            pools=pools("CSA", 0),
            cp_group=(0,),
            request_id=77,
            request_start=0,
        )
        self.assertEqual("CP1_BYPASS", cp1.fallback_reason)
        graph = build(0, 0, cuda_graph=True)
        self.assertFalse(graph.eligible)
        self.assertEqual((), graph.schedule)


if __name__ == "__main__":
    unittest.main(verbosity=2)
