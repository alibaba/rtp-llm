"""PP adapter admission and real byte imports; GPU tests cover block execution."""

import dataclasses
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.test import dsv41_stage_test as fixture
from rtp_llm.models.test.dsv41_cache_transfer_test import (
    Contract,
    Identity,
    Receiver,
    Span,
    caches,
    slots,
    transfer,
)
from rtp_llm.models.test.dsv41_pp_test import frame_for, pp

stage, execution = fixture.stage, fixture.execution
with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_block": fixture.block,
        "rtp_llm.models_py.modules.dsv4.v41_cache_transfer": transfer,
        "rtp_llm.models_py.modules.dsv4.v41_indexer": SimpleNamespace(
            V41IndexResult=execution.V41IndexResult
        ),
        "rtp_llm.models_py.modules.dsv4.v41_pp": pp,
        "rtp_llm.models_py.modules.dsv4.v41_stage": stage,
        "rtp_llm.models_py.modules.dsv4.v41_stage_attention": execution,
    },
):
    adapter = fixture.load_codec_source("_dsv41_stage_pp_test", "v41_stage_pp.py")


class V41StagePPTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract("a" * 64, "b" * 64, "c" * 64, 512)
        self.identities = (Identity(901, 1, 9), Identity(52, 2, 17))
        self.spans = (
            Span(self.identities[0], 0, 0, 257),
            Span(self.identities[1], 0, 0, 3),
        )
        self.dispatch = adapter.V41PPDispatch(
            "stage-pp-0",
            2,
            0,
            self.spans,
            torch.tensor([[1, 2, 1], [0, 256, 1], [-1, -1, 0], [0, 0, 1]]),
        )
        self.source, self.target = caches(), caches()
        self.source_tables = dict(
            zip(
                self.identities,
                (
                    {"hca_kv": (2, 0), "indexer_kv": (3, 1)},
                    {"hca_kv": (4,), "indexer_kv": (5,)},
                ),
            )
        )
        self.target_tables = dict(
            zip(
                self.identities,
                (
                    {"hca_kv": (1, 3), "indexer_kv": (4, 0)},
                    {"hca_kv": (5,), "indexer_kv": (2,)},
                ),
            )
        )
        self.receiver = Receiver(self.contract, 2, self.target, backend="eager")
        for identity in self.identities:
            self.receiver.begin_request(identity, self.target_tables[identity])
        for span in self.spans:
            for tag, cache in self.source.items():
                width = (
                    cache.layout.payload_bytes_per_entry
                    + cache.layout.scale_bytes_per_entry
                )
                packed = (
                    (torch.arange(span.count * width) % 251)
                    .byte()
                    .view(span.count, width)
                )
                cache.scatter(
                    packed,
                    slots(self.source_tables[span.identity][tag], 0, span.count),
                    backend="eager",
                )

        # Use the real stage preflight with lightweight layer descriptors. This
        # isolates transport tests from full-width FFN numerics tested on SM120.
        self.driver = object.__new__(stage.V41Stage)
        self.driver.begin, self.driver.end = 21, 40
        self.driver.role, self.driver.pp_rank = "prefill", 1
        self.driver.dim, self.driver.device = 5120, torch.device("cpu")
        self.driver.max_rows, self.driver.max_positions = 128, 512
        self.driver.blocks = tuple(
            SimpleNamespace(layer_id=s.layer_id, attention=SimpleNamespace(sources=s))
            for s in fixture.config_fixture().layer_sources()[21:]
        )
        self.driver.engram = fixture.stage_module.V41EngramStage(
            {},
            None,
            device="cpu",
            max_host_bytes=0,
            max_weight_bytes=0,
            backend="eager",
        ).initialize()
        self.addCleanup(self.driver.engram.close)
        execution_patch = patch.object(
            self.driver, "_run_prepared", side_effect=self.execute
        )
        self.execution = execution_patch.start()
        self.addCleanup(execution_patch.stop)

    def execute(self, state, rows, plans, hashes, selections):
        for span in self.current_dispatch.spans:
            self.receiver.require_ready(
                span.identity, span.start + span.count, span.chunk_id
            )
        self.assertIsNone(hashes)
        return stage.V41StageResult(state, rows, selections, None)

    def plan(self, layer, dispatch, memory, tables):
        batch = dispatch.forward_rows("cpu")
        n = batch.positions.numel()
        pages = {}
        for tag in memory:
            pages[tag] = torch.full((n, 2), -1, dtype=torch.int32)
            for row, (index, _, valid) in enumerate(dispatch.row_map.tolist()):
                if valid:
                    admitted = tables[dispatch.spans[index].identity][tag]
                    pages[tag][row, : len(admitted)] = torch.tensor(admitted)
        kwargs = dict(
            swa=fixture.make_cache("swa_kv")[0],
            swa_writes=torch.arange(n),
            swa_reads=torch.arange(n).view(n, 1),
            kv_source_layer_id=20,
            global_cache=memory["hca_kv"],
            index_cache=memory["indexer_kv"],
            global_pages=pages["hca_kv"],
            index_pages=pages["indexer_kv"],
        )
        if layer == 20:
            for tag, key in (
                ("hca_kv", "global_writes"),
                ("indexer_kv", "index_writes"),
            ):
                kwargs[key] = torch.tensor(
                    [
                        (
                            tables[dispatch.spans[index].identity][tag][position // 256]
                            * 256
                            + position % 256
                            if valid
                            else -1
                        )
                        for index, position, valid in dispatch.row_map.tolist()
                    ],
                    dtype=torch.int64,
                )
        return execution.V41AttentionPlan(**kwargs)

    def result(self, dispatch):
        frame = frame_for(dispatch.row_map)
        batch = dispatch.forward_rows("cpu")
        selected = execution.V41Selections(
            batch,
            20,
            20,
            1,
            execution.V41IndexResult(
                frame.topk, frame.candidate_blocks, frame.candidate_counts
            ),
        )
        return stage.V41StageResult(
            fixture.block.V41ResidualState(frame.hidden_states, frame.prev_pre_mix, 21),
            batch,
            selected,
            None,
            self.plan(20, dispatch, self.source, self.source_tables),
        )

    def wire(self, dispatch=None):
        dispatch = dispatch or self.dispatch
        return adapter.export_v41_stage_result(
            self.result(dispatch),
            dispatch,
            self.contract,
            self.source_tables,
            backend="eager",
        )

    def plans(self, dispatch=None):
        dispatch = dispatch or self.dispatch
        return {
            layer: self.plan(layer, dispatch, self.target, self.target_tables)
            for layer in range(21, 40)
        }

    def run_stage(self, wire=None, dispatch=None, plans=None):
        dispatch = dispatch or self.dispatch
        self.current_dispatch = dispatch
        return adapter.run_v41_pp_stage(
            self.driver,
            wire if wire is not None else self.wire(dispatch),
            self.receiver,
            dispatch,
            plans if plans is not None else self.plans(dispatch),
        )

    def assert_pristine(self):
        for cache in self.target.values():
            self.assertTrue(bool((cache.raw == 173).all()))
        for span in self.spans:
            with self.assertRaises(ValueError):
                self.receiver.require_ready(
                    span.identity, span.start + span.count, span.chunk_id
                )
        self.execution.assert_not_called()

    def test_relocated_multi_request_import_binds_rows_and_preserves_payload(self):
        wire = self.wire()
        result = self.run_stage(wire)
        self.assertEqual(result.rows.request_ids.tolist(), [52, 901, -1, 901])
        self.assertEqual(result.rows.invocation_id, self.dispatch.invocation_id)
        self.assertIs(result.selections.result.topk, wire["v41_l20_topk"])
        self.assertIs(
            result.selections.result.candidate_blocks, wire["v41_candidate_blocks"]
        )
        valid = result.rows.valid_rows
        torch.testing.assert_close(
            result.state.residual[valid], wire["hidden_states"][valid], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.state.previous_pre_mix[valid],
            wire["v41_prev_pre_mix"][valid],
            rtol=0,
            atol=0,
        )
        self.assertTrue(bool((result.state.residual[~valid] == 0).all()))
        for span in self.spans:
            for tag in self.target:
                expected = self.source[tag].gather(
                    slots(self.source_tables[span.identity][tag], 0, span.count),
                    backend="eager",
                )
                actual = self.target[tag].gather(
                    slots(self.target_tables[span.identity][tag], 0, span.count),
                    backend="eager",
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "round"):
            self.run_stage(wire)
        self.assertEqual(self.execution.call_count, 1)

    def test_invalid_late_layer_and_replica_bindings_fail_before_import(self):
        wire = self.wire()
        for failure in (
            "missing",
            "late_swa",
            "hca_pages",
            "index_pages",
            "foreign_pool",
            "alias",
        ):
            plans = self.plans()
            if failure == "missing":
                del plans[39]
            elif failure == "late_swa":
                plans[39] = dataclasses.replace(
                    plans[39], swa_writes=torch.full((4,), 999999)
                )
            elif failure == "alias":
                plans[39] = dataclasses.replace(plans[39], swa=plans[21].swa)
            else:
                foreign = caches()
                for layer, plan in plans.items():
                    changes = {}
                    if failure == "foreign_pool":
                        changes = dict(
                            global_cache=foreign["hca_kv"],
                            index_cache=foreign["indexer_kv"],
                        )
                    else:
                        name = (
                            "global_pages" if failure == "hca_pages" else "index_pages"
                        )
                        changed = getattr(plan, name).clone()
                        changed[0, 0] = 0
                        changes[name] = changed
                    plans[layer] = dataclasses.replace(plan, **changes)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.run_stage(wire, plans=plans)
            self.assert_pristine()
        self.run_stage(wire)

    def test_closed_engram_owner_rejects_before_import(self):
        self.driver.engram.close()
        with self.assertRaisesRegex(RuntimeError, "ready|closed"):
            self.run_stage()
        self.assert_pristine()

    def test_dispatch_invocation_round_generation_and_cp_are_bound(self):
        wire = self.wire()
        for changes in (
            dict(invocation_id="another-forward"),
            dict(round_id=1),
            dict(cp_rank=1),
            dict(row_map=self.dispatch.row_map[[1, 0, 2, 3]]),
            dict(
                spans=(
                    dataclasses.replace(self.spans[0], identity=Identity(901, 3, 9)),
                    self.spans[1],
                )
            ),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "dispatch|partner"
            ):
                self.run_stage(
                    wire, dataclasses.replace(self.dispatch, **changes), self.plans()
                )
            self.assert_pristine()
        with self.assertRaisesRegex(ValueError, "dispatch"):
            self.run_stage(
                {
                    key: value
                    for key, value in wire.items()
                    if key != "v41_stage_dispatch"
                }
            )
        self.assert_pristine()

    def test_current_dispatch_cannot_bypass_delta_admission(self):
        wire = self.wire()
        for key in ("v41_delta_header", "v41_delta_spans", "v41_cache_fingerprint"):
            changed = wire[key].clone()
            changed.flatten()[-1] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_stage({**wire, key: changed})
            self.assert_pristine()
        self.receiver.cancel(self.identities[1])
        with self.assertRaisesRegex(ValueError, "cancelled"):
            self.run_stage(wire)
        self.assert_pristine()

    def test_zero_local_rows_import_full_cp_deltas(self):
        dispatch = dataclasses.replace(
            self.dispatch, row_map=torch.empty(0, 3, dtype=torch.int64)
        )
        wire = self.wire(dispatch)
        self.assertNotIn("hidden_states", wire)
        self.assertTrue(all(t.numel() for t in wire.values()))
        result = self.run_stage(wire, dispatch)
        self.assertEqual(result.state.residual.shape, (0, 4, 5120))
        for span in self.spans:
            self.receiver.require_ready(span.identity, span.count, 0)

    def test_empty_round_followed_by_nonempty_round(self):
        empty = dataclasses.replace(
            self.dispatch, spans=(), row_map=torch.empty(0, 3, dtype=torch.int64)
        )
        self.run_stage(dispatch=empty)
        self.run_stage(
            dispatch=dataclasses.replace(
                self.dispatch, round_id=1, invocation_id="next"
            )
        )
        self.assertEqual(self.execution.call_count, 2)

    def test_next_chunk_keeps_absolute_rows_and_exact_readiness(self):
        first = self.run_stage()
        saved = first.state.previous_pre_mix.clone()
        span = Span(self.identities[0], 1, 257, 2)
        dispatch = dataclasses.replace(
            self.dispatch,
            invocation_id="stage-pp-1",
            round_id=1,
            spans=(span,),
            row_map=torch.tensor([[0, 258, 1]]),
        )
        second = self.run_stage(dispatch=dispatch)
        self.assertEqual(second.rows.request_ids.tolist(), [901])
        self.assertEqual(second.rows.positions.tolist(), [258])
        self.receiver.require_ready(self.identities[0], 259, 1)
        with self.assertRaisesRegex(ValueError, "not current"):
            self.receiver.require_ready(self.identities[0], 257, 0)
        torch.testing.assert_close(first.state.previous_pre_mix, saved, rtol=0, atol=0)

    def test_malformed_trusted_rows_are_rejected_before_translation(self):
        for row_map in (
            torch.tensor([[-1, 0, 1]]),
            torch.tensor([[0, 257, 1]]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[0, 0, 2]]),
            torch.tensor([[0, 0, 1], [0, 0, 1]]),
            self.dispatch.row_map.int(),
            torch.zeros(129, 3, dtype=torch.int64),
        ):
            with self.subTest(row_map=row_map), self.assertRaises(ValueError):
                dataclasses.replace(self.dispatch, row_map=row_map).snapshot(
                    self.contract
                )

    def test_export_rejects_stale_results_or_wrong_source_page_maps(self):
        result = self.result(self.dispatch)
        wrong_plan = dataclasses.replace(
            result.l20_plan, index_pages=result.l20_plan.global_pages
        )
        wrong_rows = dataclasses.replace(result.rows, invocation_id="stale")
        for changed in (
            dataclasses.replace(result, rows=wrong_rows),
            dataclasses.replace(result, l20_plan=None),
            dataclasses.replace(result, l20_plan=wrong_plan),
            dataclasses.replace(
                result, state=dataclasses.replace(result.state, next_layer_id=40)
            ),
            dataclasses.replace(
                result,
                selections=dataclasses.replace(result.selections, source_layer_id=24),
            ),
        ):
            with self.assertRaises(ValueError):
                adapter.export_v41_stage_result(
                    changed,
                    self.dispatch,
                    self.contract,
                    self.source_tables,
                    backend="eager",
                )

    def test_receiver_table_snapshot_cannot_change_admission(self):
        tables = self.receiver.block_tables(self.identities[0])
        tables["hca_kv"] = (0, 2)
        self.assertEqual(
            self.receiver.block_tables(self.identities[0]),
            self.target_tables[self.identities[0]],
        )

    def test_cp_peers_preflight_full_delta_before_any_replica_import(self):
        from rtp_llm.models.test.dsv41_cp_test import SimulatedGroup, cp

        for invalid in (True, False):
            group = SimulatedGroup()
            cases, sessions, dispatches, wires = [], [], [], []
            for rank, records in enumerate(
                ([[0, 1, 1]], [[1, 0, 1]], [[0, 0, 1], [1, 1, 1]], [])
            ):
                case = V41StagePPTest()
                case.setUp()
                self.addCleanup(case.doCleanups)
                case.spans = tuple(
                    Span(identity, 0, 0, 2) for identity in case.identities
                )
                dispatch = dataclasses.replace(
                    case.dispatch,
                    cp_rank=rank,
                    spans=case.spans,
                    row_map=torch.tensor(records, dtype=torch.int64).reshape(-1, 3),
                )
                case.current_dispatch = dispatch
                case.receiver.cp_rank = rank
                for op in case.driver.blocks:
                    op.kv = SimpleNamespace(max_rows=128)
                op = cp.V41CPExchange(
                    cp.V41CPPlan((0, 1, 2, 3), 1, "pp-cp", 2, 512), group, "cpu"
                )
                op.rank, op._identity = rank, op._runtime_identity()
                cases.append(case)
                sessions.append(op)
                dispatches.append(dispatch)
                wires.append(case.wire(dispatch))
            peers = tuple(dispatch.forward_rows("cpu") for dispatch in dispatches)
            forwards = group.run(lambda rank: sessions[rank].begin(peers))
            full_map = torch.tensor(
                [
                    [0, 1, 1],
                    [-1, -1, 0],
                    [1, 0, 1],
                    [-1, -1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                    [-1, -1, 0],
                    [-1, -1, 0],
                ]
            )
            if invalid:
                wires[1]["v41_delta_header"][3] += 1

            def run(rank):
                case, forward = cases[rank], forwards[rank]
                full_dispatch = dataclasses.replace(dispatches[rank], row_map=full_map)
                plans = case.plans(full_dispatch)

                # This fixture covers admission/import; full CP stage numerics
                # are exercised independently by dsv41_cp_test and SM120.
                def execute(driver, prepared, plans, selected):
                    return driver._run_prepared(
                        prepared[0], prepared[1], plans, prepared[2], selected
                    )

                with patch.object(forward, "run_prepared", side_effect=execute):
                    if invalid:
                        with self.assertRaisesRegex(RuntimeError, "mismatch"):
                            adapter.run_v41_pp_stage(
                                case.driver,
                                wires[rank],
                                case.receiver,
                                dispatches[rank],
                                plans,
                                cp_forward=forward,
                            )
                    else:
                        result = adapter.run_v41_pp_stage(
                            case.driver,
                            wires[rank],
                            case.receiver,
                            dispatches[rank],
                            plans,
                            cp_forward=forward,
                        )
                        self.assertEqual(
                            result.rows.positions.numel(), peers[rank].positions.numel()
                        )

            group.run(run)
            for case in cases:
                if invalid:
                    case.assert_pristine()
                else:
                    case.execution.assert_called_once()


if __name__ == "__main__":
    unittest.main()
