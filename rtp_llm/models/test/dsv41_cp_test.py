"""Real stage numerics with a four-thread CPU simulation of NCCL row exchange."""

import dataclasses
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

from rtp_llm.models.test import dsv41_stage_test as fixture

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_projection": fixture.projection,
        "rtp_llm.models_py.modules.dsv4.v41_stage": fixture.stage,
        "rtp_llm.models_py.modules.dsv4.v41_stage_attention": fixture.execution,
    },
):
    cp = fixture.load_codec_source("_dsv41_cp_test", "v41_cp.py")


def peer_rows(invocation="cp-test"):
    return tuple(
        fixture.execution.V41ForwardRows(
            invocation,
            torch.full((len(positions),), 19, dtype=torch.int64),
            torch.tensor(positions, dtype=torch.int64),
            torch.ones(len(positions), dtype=torch.bool),
        )
        for positions in ([1, 4], [3], [0, 2], [])
    )


def plans_for(blocks, rows, *, memory=None, page=2):
    memory = {} if memory is None else memory
    plans = {}
    n = rows.positions.numel()
    for op in blocks:
        sources = op.attention.sources
        reads = torch.full((n, 8), -1, dtype=torch.int64)
        for row in range(n):
            if rows.valid_rows[row]:
                length = int(rows.positions[row]) + 1
                reads[row, :length] = torch.arange(length)
        kwargs = dict(
            swa=fixture.make_cache("swa_kv")[0],
            swa_writes=rows.positions.clone(),
            swa_reads=reads,
        )
        if sources.compress_ratio:
            source = sources.kv_source_layer_id
            if source not in memory:
                memory[source] = (
                    fixture.make_cache(
                        "csa_kv" if sources.compress_ratio == 2 else "hca_kv"
                    )[0],
                    fixture.make_cache("indexer_kv")[0],
                )
            global_cache, index_cache = memory[source]
            kwargs.update(
                kv_source_layer_id=source,
                global_cache=global_cache,
                index_cache=index_cache,
                global_pages=torch.full((n, 1), page, dtype=torch.int32),
                index_pages=torch.full((n, 1), (page + 1) % 3, dtype=torch.int32),
            )
            if sources.owns_kv:
                ratio = sources.compress_ratio
                kwargs.update(
                    global_writes=rows.positions // ratio
                    + page * global_cache.layout.entries,
                    index_writes=rows.positions // ratio
                    + ((page + 1) % 3) * index_cache.layout.entries,
                )
                if ratio == 2:
                    previous = torch.full((n,), -1)
                    for row in range(n):
                        if rows.valid_rows[row] and rows.positions[row] % 2:
                            previous[row] = (
                                (rows.positions == rows.positions[row] - 1)
                                & rows.valid_rows
                            ).nonzero()[0, 0]
                    kwargs.update(
                        pair_state=fixture.compressor.V41PairState(
                            torch.zeros(2, 4096)
                        ),
                        pair_metadata=fixture.compressor.V41PairMetadata(
                            rows.request_ids,
                            rows.positions,
                            rows.valid_rows,
                            previous,
                            torch.full((n,), -1),
                            torch.where(
                                rows.valid_rows & (rows.positions > 0),
                                rows.positions % 4,
                                -1,
                            ),
                        ),
                    )
        plans[op.layer_id] = fixture.execution.V41AttentionPlan(**kwargs)
    return plans, memory


class SimulatedGroup:
    def __init__(self):
        self.barrier = threading.Barrier(4, timeout=30)
        self.values = [None] * 4
        self.local = threading.local()

    def gather(self, out, value, *, group):
        if group is not self:
            raise AssertionError("CP transport escaped its explicit stage group")
        self.values[self.local.rank] = value.clone()
        self.barrier.wait()
        out.copy_(torch.cat(self.values))
        self.barrier.wait()

    def run(self, callback):
        def worker(rank):
            self.local.rank = rank
            return callback(rank)

        with patch.object(
            torch.distributed, "get_world_size", return_value=4
        ), patch.object(
            torch.distributed, "all_gather_into_tensor", side_effect=self.gather
        ), ThreadPoolExecutor(
            max_workers=4
        ) as executor:
            futures = [executor.submit(worker, rank) for rank in range(4)]
            return [future.result(timeout=60) for future in futures]


def session(group, rank, pp_rank=0):
    op = cp.V41CPExchange(
        cp.V41CPPlan(tuple(range(4)), pp_rank, "generation-1", 2, 256), group, "cpu"
    )
    # Only the communication simulator bypasses CUDA admission.
    op.rank, op._identity = rank, op._runtime_identity()
    return op


class V41CPTest(unittest.TestCase):
    def test_plan_bounds_membership_and_fingerprint(self):
        plan = cp.V41CPPlan((0, 1, 2, 3), 0, "worker-1", 2, 256)
        for field, value in (
            ("pp_rank", 2),
            ("world_ranks", (0, 0, 1, 2)),
            ("world_ranks", [0, 1, 2, 3]),
            ("generation", ""),
            ("rows_per_rank", 33),
            ("rows_per_rank", 0),
            ("max_positions", 32769),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                dataclasses.replace(plan, **{field: value})
        for field, value in (
            ("pp_rank", 1),
            ("generation", "worker-2"),
            ("rows_per_rank", 3),
            ("max_positions", 512),
            ("world_ranks", (4, 5, 6, 7)),
        ):
            self.assertNotEqual(
                plan.fingerprint,
                dataclasses.replace(plan, **{field: value}).fingerprint,
            )
        op = cp.V41CPExchange(plan, object(), "cpu")
        with self.assertRaisesRegex(RuntimeError, "NCCL"):
            op.admit()
        with self.assertRaisesRegex(RuntimeError, "admitted"):
            op.begin(peer_rows())

    def test_dispatch_rejects_missing_duplicate_and_inconsistent_rows(self):
        plan = cp.V41CPPlan((0, 1, 2, 3), 0, "worker-1", 2, 256)
        peers = peer_rows()
        for rank, changed in (
            (0, dataclasses.replace(peers[0], invocation_id="other")),
            (2, dataclasses.replace(peers[2], positions=torch.tensor([1, 2]))),
            (0, dataclasses.replace(peers[0], positions=torch.tensor([1, 7]))),
        ):
            wrong = list(peers)
            wrong[rank] = changed
            with self.assertRaises(ValueError):
                cp._dispatch_rows(wrong, plan, "cpu")

    def test_peer_dispatch_mismatch_rejects_all_ranks_and_poisons_session(self):
        group = SimulatedGroup()
        sessions = [session(group, rank) for rank in range(4)]

        def run(rank):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                sessions[rank].begin(peer_rows("wrong" if rank == 2 else "right"))
            with self.assertRaisesRegex(RuntimeError, "healthy"):
                sessions[rank].begin(peer_rows())

        group.run(run)

    def test_full_21_19_stages_match_unsplit_rows_with_remote_pairs_and_empty_rank(
        self,
    ):
        environment = fixture.V41StageTest(
            methodName="test_empty_forward_and_fresh_invocation_after_failure"
        )
        environment.setUp()
        self.addCleanup(environment.doCleanups)
        peers = peer_rows()
        group = SimulatedGroup()
        sessions = [session(group, rank) for rank in range(4)]
        forwards = group.run(lambda rank: sessions[rank].begin(peers))
        full = forwards[0].rows
        embeddings = torch.randn(full.positions.numel(), 32).bfloat16()
        initial = fixture.block.start_v41_residual(embeddings, full.valid_rows)
        windows = torch.full((full.positions.numel(), 4), -1, dtype=torch.int64)
        for row in range(full.positions.numel()):
            if full.valid_rows[row]:
                position = int(full.positions[row])
                windows[row, : min(4, position + 1)] = torch.tensor(
                    [(position - i) % 5 for i in range(min(4, position + 1))]
                )
        reference_plans, reference_memory = plans_for(environment.blocks[:21], full)
        reference = environment.driver("prefill", 0).run(
            initial, full, reference_plans, windows=windows
        )
        local_plans, memories = zip(
            *(
                plans_for(environment.blocks[:21], full, page=rank % 3)
                for rank in range(4)
            )
        )

        def first(rank):
            forward = forwards[rank]
            part = forward.local_slice
            state = fixture.block.V41ResidualState(
                initial.residual[part].contiguous(),
                initial.previous_pre_mix[part].contiguous(),
                0,
            )
            return forward.run(
                environment.driver("prefill", 0),
                state,
                peers[rank],
                local_plans[rank],
                windows=windows[part].contiguous(),
            )

        results = group.run(first)
        for rank, result in enumerate(results):
            part = forwards[rank].local_slice
            torch.testing.assert_close(
                result.state.residual, reference.state.residual[part], rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.selections.result.topk,
                reference.selections.result.topk[part],
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                result.l20_plan.swa_writes.shape, peers[rank].positions.shape
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                forwards[rank].exchange(None)
        reference_plans, _ = plans_for(
            environment.blocks[21:], full, memory=reference_memory
        )
        expected = environment.driver("prefill", 1).run(
            reference.state, full, reference_plans, selections=reference.selections
        )
        sessions = [session(group, rank, 1) for rank in range(4)]
        forwards = group.run(lambda rank: sessions[rank].begin(peers))

        def second(rank):
            plans, _ = plans_for(
                environment.blocks[21:], full, memory=memories[rank], page=rank % 3
            )
            return forwards[rank].run(
                environment.driver("prefill", 1),
                results[rank].state,
                peers[rank],
                plans,
                selections=results[rank].selections,
            )

        actual = group.run(second)
        for rank, result in enumerate(actual):
            torch.testing.assert_close(
                result.hidden,
                expected.hidden[forwards[rank].local_slice],
                rtol=0,
                atol=0,
            )
            self.assertEqual(sessions[rank]._round, 1)

    def test_one_bad_stage_plan_rejects_all_peers_before_any_cache_write(self):
        environment = fixture.V41StageTest(
            methodName="test_empty_forward_and_fresh_invocation_after_failure"
        )
        environment.setUp()
        self.addCleanup(environment.doCleanups)
        group = SimulatedGroup()
        sessions = [session(group, rank) for rank in range(4)]
        peers = peer_rows()
        forwards = group.run(lambda rank: sessions[rank].begin(peers))
        plans = [
            plans_for(environment.blocks[:21], forwards[rank].rows)[0]
            for rank in range(4)
        ]
        plans[2][20] = dataclasses.replace(
            plans[2][20], swa_writes=torch.full((8,), 999999)
        )

        def run(rank):
            count = peers[rank].positions.numel()
            initial = fixture.block.start_v41_residual(
                torch.randn(count, 32).bfloat16(), peers[rank].valid_rows
            )
            with self.assertRaisesRegex(RuntimeError, "preflight mismatch"):
                forwards[rank].run(
                    environment.driver("prefill", 0),
                    initial,
                    peers[rank],
                    plans[rank],
                    windows=torch.zeros(count, 4, dtype=torch.int64),
                )

        group.run(run)
        for rank in range(4):
            for plan in plans[rank].values():
                self.assertTrue(bool((plan.swa.raw == 173).all()))
            self.assertTrue(sessions[rank]._failed)


if __name__ == "__main__":
    unittest.main()
