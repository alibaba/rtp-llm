"""Generation leases, native shared-pool views and chunk-safe cache planning."""

import dataclasses
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings, dsv41_page_layouts
from rtp_llm.models.test import dsv41_cp_test as cp_fixture
from rtp_llm.models.test import dsv41_stage_pp_test as pp_fixture
from rtp_llm.models.test import dsv41_stage_test as fixture
from rtp_llm.models.test.dsv41_cache_transfer_test import Identity, Span, transfer

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_cache": fixture.codec,
        "rtp_llm.models_py.modules.dsv4.v41_cache_transfer": transfer,
        "rtp_llm.models_py.modules.dsv4.v41_compressor": fixture.compressor,
        "rtp_llm.models_py.modules.dsv4.v41_stage": fixture.stage,
        "rtp_llm.models_py.modules.dsv4.v41_stage_attention": fixture.execution,
    },
):
    planner = fixture.load_codec_source("_dsv41_cache_plan_test", "v41_cache_plan.py")


def native_fixture(config, role="prefill", pp_rank=0, device="cpu"):
    bindings = dsv41_cache_bindings(config.text, 2 if role == "prefill" else 1, pp_rank)
    layouts = dsv41_page_layouts(1)
    count = 40 if role == "decode" else 21 if pp_rank == 0 else 19
    # BlockPool uses layer-major narrow views of a shared backing allocation.
    raw = torch.full(
        (sum(8 * layouts[b.tag].stride_bytes for b in bindings),),
        165,
        dtype=torch.uint8,
        device=device,
    )
    groups, offset = [[] for _ in range(count)], 0
    for b in bindings:
        width = layouts[b.tag].stride_bytes
        view = raw[offset : offset + 8 * width].view(8, width)
        offset += view.numel()
        if b.tag == "csa_state":
            view = view.view(torch.float32)
        groups[b.storage_layer_id].append(
            SimpleNamespace(
                tag=b.tag,
                layer_id=b.storage_layer_id,
                seq_size_per_block=256,
                kv_scale_base=None,
                kv_cache_base=view,
                read_replica_source_layer_id=(
                    b.source_layer_id if b.is_read_replica else None
                ),
            )
        )
    return (
        SimpleNamespace(layer_count=count, get_layer_cache_groups=lambda i: groups[i]),
        raw,
    )


def make_owner(
    config=None, role="prefill", pp_rank=0, budget=64 * 1024**2, max_positions=512
):
    config = fixture.config_fixture() if config is None else config
    native, raw = native_fixture(config, role, pp_rank)
    owner = planner.V41StageCache.from_native(
        config,
        native,
        role=role,
        pp_rank=pp_rank,
        max_positions=max_positions,
        max_scratch_bytes=budget,
        backend="eager",
    )
    return owner, raw


def tables(owner, second=False, full=2):
    shift = 3 if second else 0
    values = {
        "swa_kv": (1 + shift,),
        "csa_state": (2 + shift,),
        "csa_kv": (1 + shift, 3 + shift),
        "hca_kv": (2 + shift, 1 + shift),
        "indexer_kv": (3 + shift, 2 + shift),
    }
    return {
        tag: value if tag in ("swa_kv", "csa_state") else value[:full]
        for tag, value in values.items()
        if tag in owner._capacities
    }


def rows_for(spans, *, reverse=False, padded=True):
    records = [
        (s.identity.request_id, p, True)
        for s in spans
        for p in range(s.start, s.start + s.count)
    ]
    if reverse:
        records.reverse()
    if padded:
        records.append((-999, -999, False))
    return fixture.execution.V41ForwardRows(
        "chunk-test",
        torch.tensor([r[0] for r in records], dtype=torch.int64),
        torch.tensor([r[1] for r in records], dtype=torch.int64),
        torch.tensor([r[2] for r in records], dtype=torch.bool),
    )


def publish(batch, *, check_windows=False):
    rows = batch.rows
    for layer, plan in batch.plans.items():
        values = (
            ((rows.positions % 17).float() / 4 + layer / 8)[:, None]
            .expand(-1, 512)
            .contiguous()
            .bfloat16()
        )
        kwargs, pair = {}, None
        complete = rows.valid_rows
        if plan.global_writes is not None:
            ratio = 2 if plan.global_cache.tag == "csa_kv" else 1
            complete = rows.valid_rows & ((rows.positions + 1) % ratio == 0)
            kwargs = dict(
                global_cache=plan.global_cache,
                global_slots=plan.global_writes,
                index_cache=plan.index_cache,
                index_slots=plan.index_writes,
            )
            if ratio == 2:
                pair = fixture.compressor.prepare_ratio2(
                    values.float(),
                    torch.zeros_like(values, dtype=torch.float32),
                    torch.ones(512, dtype=torch.bfloat16),
                    plan.pair_state,
                    plan.pair_metadata,
                    backend="eager",
                )
        frame = fixture.projection.V41KVFrame(
            values,
            values if kwargs else None,
            values[:, :128].contiguous() if kwargs else None,
            rows.valid_rows,
            complete,
            plan.global_cache.tag if kwargs else None,
            pair,
            "eager",
            None,
        )
        frame.publish(plan.swa, plan.swa_writes, **kwargs)
        if check_windows and layer in (0, 20):
            for row, (pos, valid) in enumerate(
                zip(rows.positions.tolist(), rows.valid_rows.tolist())
            ):
                if not valid:
                    continue
                positions = torch.arange(max(0, pos - 127), pos + 1)
                expected_values = (
                    ((positions % 17).float() / 4 + layer / 8)[:, None]
                    .expand(-1, 512)
                    .contiguous()
                    .bfloat16()
                )
                expected = fixture.codec.quantize_rows_eager(expected_values, "swa_kv")
                actual = plan.swa.gather(
                    plan.swa_reads[row, : len(positions)].contiguous(), backend="eager"
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return "published"


class V41CachePlanTest(unittest.TestCase):
    def cp_fixture(self, *, mismatch=False):
        group = cp_fixture.SimulatedGroup()
        sessions = [cp_fixture.session(group, rank) for rank in range(4)]
        owners, identities = [], []
        for rank in range(4):
            owner, _ = make_owner(max_positions=256)
            identity = Identity(19, 2 if mismatch and rank == 2 else 1, 0)
            owner.begin_request(identity, tables(owner, second=rank % 2, full=1))
            owners.append(owner)
            identities.append(identity)
        return group, sessions, owners, identities

    def test_cp_generation_or_local_planning_failure_rejects_before_native_writes(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                group, sessions, owners, identities = self.cp_fixture(mismatch=mismatch)
                forwards = group.run(
                    lambda rank: sessions[rank].begin(cp_fixture.peer_rows())
                )
                before = [owner.caches[0, "swa_kv"].raw.clone() for owner in owners]

                def run(rank):
                    span = Span(
                        identities[rank], 0, 0, 4 if not mismatch and rank == 2 else 5
                    )
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        owners[rank].prepare_cp(forwards[rank], (span,))

                group.run(run)
                for rank, owner in enumerate(owners):
                    self.assertIsNone(owner._active)
                    self.assertTrue(owner._drained)
                    self.assertEqual(owner._state(identities[rank]).visible, 0)
                    torch.testing.assert_close(
                        owner.caches[0, "swa_kv"].raw, before[rank], rtol=0, atol=0
                    )

    def test_cp_stale_batch_preflight_rejects_every_rank(self):
        group, sessions, owners, identities = self.cp_fixture()
        forwards = group.run(lambda rank: sessions[rank].begin(cp_fixture.peer_rows()))
        batches = group.run(
            lambda rank: owners[rank].prepare_cp(
                forwards[rank], (Span(identities[rank], 0, 0, 5),)
            )
        )
        support = fixture.V41StageTest()
        support.setUp()
        self.addCleanup(support.doCleanups)
        stage = support.driver("prefill", 0)
        batches[2].discard()

        def run(rank):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                batches[rank].run(stage, None, cp_forward=forwards[rank])

        group.run(run)
        for rank, batch in enumerate(batches):
            self.assertEqual(owners[rank]._state(identities[rank]).visible, 0)
            if rank != 2:
                batch.discard()

    def test_cp_retention_failure_prevents_all_peers_advancing(self):
        group, sessions, owners, identities = self.cp_fixture()
        for owner, identity in zip(owners, identities):
            spans = (Span(identity, 0, 0, 128),)
            batch = owner.prepare(rows_for(spans, padded=False), spans)
            batch._execute(lambda _: publish(batch))
        peers = tuple(
            dataclasses.replace(rows, positions=rows.positions + 128)
            for rows in cp_fixture.peer_rows()
        )
        forwards = group.run(lambda rank: sessions[rank].begin(peers))
        batches = group.run(
            lambda rank: owners[rank].prepare_cp(
                forwards[rank], (Span(identities[rank], 1, 128, 5),)
            )
        )
        self.assertTrue(all(batch.scratch_bytes for batch in batches))

        def run(rank):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                batches[rank]._execute(lambda _: publish(batches[rank]))

        original = fixture.codec.V41PagedCache.scatter

        def scatter(cache, *args, **kwargs):
            if cache is owners[2].caches[0, "swa_kv"]:
                raise RuntimeError("injected retention failure")
            return original(cache, *args, **kwargs)

        with patch.object(fixture.codec.V41PagedCache, "scatter", scatter):
            group.run(run)
        for owner, identity in zip(owners, identities):
            self.assertTrue(owner._failed)
            self.assertEqual(owner._state(identity).visible, 128)
            self.assertEqual(owner._state(identity).chunk, 1)
            owner.release(identity)

    def test_real_cp_stage_matches_unsplit_with_relocated_pools_and_empty_rank(self):
        support = fixture.V41StageTest()
        support.setUp()
        self.addCleanup(support.doCleanups)
        stage = support.driver("prefill", 0)
        group, sessions, owners, identities = self.cp_fixture()
        forwards = group.run(lambda rank: sessions[rank].begin(cp_fixture.peer_rows()))
        batches = group.run(
            lambda rank: owners[rank].prepare_cp(
                forwards[rank], (Span(identities[rank], 0, 0, 5),)
            )
        )
        rows = forwards[0].rows
        state = fixture.block.start_v41_residual(
            torch.randn(rows.positions.numel(), 32).bfloat16(), rows.valid_rows
        )
        windows = torch.full((rows.positions.numel(), 4), -1, dtype=torch.int64)
        for row in range(rows.positions.numel()):
            if rows.valid_rows[row]:
                position = int(rows.positions[row])
                windows[row, : min(4, position + 1)] = torch.tensor(
                    [(position - i) % 5 for i in range(min(4, position + 1))]
                )
        reference, _ = make_owner(support.config, max_positions=256)
        reference.begin_request(identities[0], tables(reference, full=1))
        expected = reference.prepare(rows, batches[0].spans).run(
            stage, state, windows=windows
        )

        def run(rank):
            part = forwards[rank].local_slice
            local = fixture.block.V41ResidualState(
                state.residual[part].contiguous(),
                state.previous_pre_mix[part].contiguous(),
                0,
            )
            return batches[rank].run(
                stage,
                local,
                windows=windows[part].contiguous(),
                cp_forward=forwards[rank],
            )

        results = group.run(run)
        for rank, result in enumerate(results):
            part = forwards[rank].local_slice
            torch.testing.assert_close(
                result.state.residual, expected.state.residual[part], rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.selections.result.topk,
                expected.selections.result.topk[part],
                rtol=0,
                atol=0,
            )
            self.assertEqual(owners[rank]._state(identities[rank]).visible, 5)
            self.assertEqual(sessions[rank]._round, 1)

    def test_native_views_bind_all_stage_ranges_and_reject_wrong_replicas_or_overlap(
        self,
    ):
        config = fixture.config_fixture()
        for role, rank, count in (
            ("prefill", 0, 21),
            ("prefill", 1, 19),
            ("decode", 0, 40),
        ):
            owner, _ = make_owner(config, role, rank)
            self.assertEqual(len(owner.sources), count)
            ptrs = {v.raw.untyped_storage().data_ptr() for v in owner.caches.values()}
            self.assertEqual(len(ptrs), 1)
            if rank == 1:
                self.assertIn((20, "hca_kv"), owner.caches)
                self.assertNotIn((20, "swa_kv"), owner.caches)
        native, _ = native_fixture(config, pp_rank=1)
        native.get_layer_cache_groups(0)[-1].read_replica_source_layer_id = None
        with self.assertRaisesRegex(ValueError, "source binding"):
            planner.V41StageCache.from_native(
                config,
                native,
                role="prefill",
                pp_rank=1,
                max_scratch_bytes=0,
                backend="eager",
            )
        owner, _ = make_owner(config)
        caches = dict(owner.caches)
        caches[1, "swa_kv"] = caches[0, "swa_kv"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            planner.V41StageCache(
                config,
                caches,
                role="prefill",
                pp_rank=0,
                max_scratch_bytes=0,
                backend="eager",
            )

    def test_chunks_preserve_every_swa_window_pair_carry_and_ratio2_page_stride(self):
        owner, _ = make_owner()
        identity = Identity(4, 1, 9)
        owner.begin_request(identity, tables(owner, full=1))
        for chunk, start, count in (
            (0, 0, 128),
            (1, 128, 127),
            (2, 255, 2),
            (3, 257, 1),
        ):
            if chunk == 2:
                owner.extend_request(identity, tables(owner))
            spans = (Span(identity, chunk, start, count),)
            rows = rows_for(spans, reverse=True, padded=count < 128)
            batch = owner.prepare(rows, spans)
            self.assertEqual(batch.scratch_bytes > 0, chunk == 1)
            for sources in owner.sources:
                batch.plans[sources.layer_id].validate(sources, batch.rows, 512)
            if chunk in (2, 3):
                meta = batch.plans[2].pair_metadata
                row = int((batch.rows.positions == start).nonzero()[0])
                self.assertEqual(int(meta.previous_rows[row]), -1)
                self.assertEqual(int(meta.previous_slots[row]), 2 * 4 + (start - 1) % 4)
            if chunk == 3:
                self.assertEqual(int(batch.plans[2].global_writes[0]), 3 * 128)
                self.assertEqual(int(batch.plans[2].index_writes[0]), 2 * 256)
            self.assertEqual(
                batch._execute(lambda _: publish(batch, check_windows=True)),
                "published",
            )
            self.assertEqual(owner._state(identity).visible, start + count)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                batch.discard()

    def test_generation_replay_cancel_growth_and_page_conflicts(self):
        owner, _ = make_owner()
        first, second = Identity(4, 1, 9), Identity(9, 2, 9)
        owner.begin_request(first, tables(owner, full=1))
        with self.assertRaisesRegex(ValueError, "overlap"):
            owner.begin_request(second, tables(owner, full=1))
        owner.begin_request(second, tables(owner, second=True))
        spans = (Span(first, 0, 0, 1), Span(second, 0, 0, 2))
        batch = owner.prepare(rows_for(spans, reverse=True), spans)
        with self.assertRaisesRegex(RuntimeError, "in-flight"):
            owner.release(first)
        batch._execute(lambda _: publish(batch))
        with self.assertRaisesRegex(ValueError, "gaps/replay"):
            owner.prepare(rows_for(spans), spans)
        with self.assertRaisesRegex(ValueError, "append-only"):
            owner.extend_request(first, tables(owner, second=True))
        owner.cancel(first)
        with self.assertRaises(ValueError):
            owner.prepare(rows_for((Span(first, 1, 1, 1),)), (Span(first, 1, 1, 1),))
        owner.release(first)
        with self.assertRaisesRegex(ValueError, "generation"):
            owner.begin_request(first, tables(owner))
        recycled = Identity(4, 3, 10)
        owner.begin_request(recycled, tables(owner))
        with self.assertRaisesRegex(ValueError, "stale"):
            owner.block_tables(first)
        spans = (Span(recycled, 0, 0, 1),)
        batch = owner.prepare(rows_for(spans), spans)
        self.assertTrue(bool((batch.plans[2].pair_metadata.previous_slots == -1).all()))
        batch.discard()

    def test_invalid_rows_budget_and_allocations_fail_without_native_mutation(self):
        owner, raw = make_owner(budget=0)
        identity = Identity(4, 1, 0)
        supplied = tables(owner)
        owner.begin_request(identity, supplied)
        supplied["swa_kv"] = (7,)
        self.assertEqual(owner.block_tables(identity)["swa_kv"], (1,))
        span = Span(identity, 0, 0, 128)
        batch = owner.prepare(rows_for((span,), padded=False), (span,))
        batch._execute(lambda _: publish(batch))
        before = raw.clone()
        span = Span(identity, 1, 128, 4)
        with self.assertRaisesRegex(ValueError, "budget"):
            owner.prepare(rows_for((span,)), (span,))
        with self.assertRaisesRegex(ValueError, "exactly"):
            owner.prepare(rows_for((dataclasses.replace(span, count=3),)), (span,))
        torch.testing.assert_close(raw, before, rtol=0, atol=0)
        span = dataclasses.replace(span, count=3)
        batch = owner.prepare(rows_for((span,)), (span,))
        self.assertEqual(batch.scratch_bytes, 0)
        batch.discard()
        self.assertEqual(owner._state(identity).visible, 128)

    def test_execution_failure_does_not_advance_and_poisoned_owner_can_release_after_drain(
        self,
    ):
        owner, _ = make_owner()
        identity = Identity(4, 1, 0)
        owner.begin_request(identity, tables(owner))
        span = Span(identity, 0, 0, 1)
        batch = owner.prepare(rows_for((span,)), (span,))

        def fail(plans):
            plans[0].swa.raw[1].zero_()
            raise RuntimeError("injected store failure")

        with self.assertRaisesRegex(RuntimeError, "injected"):
            batch._execute(fail)
        self.assertEqual(owner._state(identity).visible, 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            owner.prepare(rows_for((span,)), (span,))
        owner.release(identity)

    def test_real_stage_matches_manual_plans_using_shared_native_pool_views(self):
        support = fixture.V41StageTest()
        support.setUp()
        self.addCleanup(support.doCleanups)
        stage = support.driver("prefill", 0)
        owner, _ = make_owner(support.config)
        # Match the stage's bounded context while keeping the same real layouts.
        owner.max_positions = stage.max_positions
        identity = Identity(4, 1, 0)
        owner.begin_request(identity, tables(owner, full=1))
        span = Span(identity, 0, 0, 2)
        batch = owner.prepare(support.batch, (span,))
        manual, _ = fixture.plans_fixture(stage.blocks, support.batch)
        expected = stage.run(
            support.initial, support.batch, manual, windows=support.windows
        )
        actual = batch.run(stage, support.initial, windows=support.windows)
        torch.testing.assert_close(
            actual.state.residual, expected.state.residual, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.state.previous_pre_mix,
            expected.state.previous_pre_mix,
            rtol=0,
            atol=0,
        )

    def test_decode_single_token_chunks_and_empty_batches(self):
        owner, _ = make_owner(role="decode")
        identities = (Identity(4, 1, 0), Identity(9, 2, 0))
        for rank, identity in enumerate(identities):
            owner.begin_request(identity, tables(owner, second=bool(rank)))
        for chunk in range(3):
            spans = tuple(Span(identity, chunk, chunk, 1) for identity in identities)
            batch = owner.prepare(rows_for(spans, reverse=True), spans)
            self.assertEqual(batch.scratch_bytes, 0)
            for sources in owner.sources:
                batch.plans[sources.layer_id].validate(sources, batch.rows, 512)
            batch._execute(lambda _: publish(batch, check_windows=True))
        spans = (Span(identities[0], 3, 3, 2),)
        with self.assertRaisesRegex(ValueError, "single-token"):
            owner.prepare(rows_for(spans), spans)
        batch = owner.prepare(rows_for((), padded=False), ())
        batch._execute(lambda _: publish(batch))
        for identity in identities:
            self.assertEqual(owner._state(identity).visible, 3)

    def test_failed_discard_drain_retains_batch_and_blocks_release(self):
        owner, _ = make_owner()
        identity = Identity(4, 1, 0)
        owner.begin_request(identity, tables(owner))
        spans = (Span(identity, 0, 0, 1),)
        batch = owner.prepare(rows_for(spans), spans)
        with patch.object(
            owner, "drain", side_effect=RuntimeError("injected drain failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                batch.discard()
        self.assertIs(owner._undrained, batch)
        with self.assertRaisesRegex(RuntimeError, "not drained"):
            owner.release(identity)
        owner.drain()
        self.assertIsNone(owner._undrained)
        owner.release(identity)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            owner.begin_request(Identity(4, 2, 0), tables(owner))

    def test_stage_one_plans_bind_actual_pp_replica_imports(self):
        support = pp_fixture.V41StagePPTest()
        support.setUp()
        self.addCleanup(support.doCleanups)
        spans = (
            Span(support.identities[0], 0, 0, 2),
            Span(support.identities[1], 0, 0, 1),
        )
        dispatch = dataclasses.replace(
            support.dispatch,
            spans=spans,
            row_map=torch.tensor([[0, 1, 1], [1, 0, 1], [-1, -1, 0], [0, 0, 1]]),
        )
        support.current_dispatch = dispatch
        owner, _ = make_owner(pp_rank=1)
        replicas = {tag: owner.caches[20, tag] for tag in ("hca_kv", "indexer_kv")}
        receiver = pp_fixture.Receiver(
            support.contract, dispatch.cp_rank, replicas, backend="eager"
        )
        support.receiver = receiver
        for rank, identity in enumerate(support.identities):
            blocks = tables(owner, second=bool(rank), full=1)
            owner.begin_request(identity, blocks)
            receiver.begin_request(identity, {tag: blocks[tag] for tag in replicas})
        batch = owner.prepare(dispatch.forward_rows("cpu"), spans)
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.dsv4.v41_stage_pp": pp_fixture.adapter},
        ):
            batch.run_pp(support.driver, support.wire(dispatch), receiver, dispatch)
        support.execution.assert_called_once()
        for span in spans:
            self.assertEqual(owner._state(span.identity).visible, span.count)
            receiver.require_ready(span.identity, span.count, span.chunk_id)
            for tag, cache in replicas.items():
                actual = cache.gather(
                    pp_fixture.slots(
                        owner.block_tables(span.identity)[tag], 0, span.count
                    ),
                    backend="eager",
                )
                expected = support.source[tag].gather(
                    pp_fixture.slots(
                        support.source_tables[span.identity][tag], 0, span.count
                    ),
                    backend="eager",
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
