import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

compute_ops = types.ModuleType("rtp_llm.ops.compute_ops")
compute_ops.PyAttentionInputs = type("PyAttentionInputs", (), {})
compute_ops.PyCacheStorePublishPlan = type("PyCacheStorePublishPlan", (), {})
compute_ops.PyEmbeddingInputs = type("PyEmbeddingInputs", (), {})
compute_ops.PyModelInputs = type("PyModelInputs", (), {})
compute_ops.PyMultimodalInputs = type("PyMultimodalInputs", (), {})
rtp_llm_package = types.ModuleType("rtp_llm")
rtp_llm_package.__path__ = [str(Path(__file__).resolve().parents[4])]
ops_package = types.ModuleType("rtp_llm.ops")
ops_package.__path__ = []
sys.modules.setdefault("rtp_llm", rtp_llm_package)
sys.modules.setdefault("rtp_llm.ops", ops_package)
sys.modules.setdefault("rtp_llm.ops.compute_ops", compute_ops)

module_path = Path(__file__).resolve().parents[1] / "chunk_prefill.py"
spec = importlib.util.spec_from_file_location(
    "kimi_k3_chunk_prefill_tested", module_path
)
assert spec is not None and spec.loader is not None
chunk_prefill = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chunk_prefill
spec.loader.exec_module(chunk_prefill)

KimiK3ChunkRdmaPublisher = chunk_prefill.KimiK3ChunkRdmaPublisher
KimiK3ChunkCachePublisher = chunk_prefill.KimiK3ChunkCachePublisher
chunkwise_rdma_enabled = chunk_prefill.chunkwise_rdma_enabled
plan_kimi_k3_chunk_rounds = chunk_prefill.plan_kimi_k3_chunk_rounds


def _publish_all(
    input_lengths: list[int],
    prefix_lengths: list[int],
    *,
    budget: int,
    alignment_tokens: int,
    transfer_page_tokens: int,
) -> tuple[
    KimiK3ChunkRdmaPublisher,
    list[tuple[tuple[int, ...], tuple[int, ...]]],
]:
    kda_layers = (0, 3)
    publisher = KimiK3ChunkRdmaPublisher(
        input_lengths,
        prefix_lengths,
        transfer_page_tokens=transfer_page_tokens,
        kda_layer_indices=kda_layers,
    )
    ranges = []
    prefix = publisher.prefix_step()
    ranges.append((prefix.begin_blocks, prefix.end_blocks))
    publisher.commit(prefix)
    rounds = plan_kimi_k3_chunk_rounds(
        input_lengths,
        prefix_lengths,
        chunk_budget=budget,
        alignment_tokens=alignment_tokens,
    )
    for round_plan in rounds:
        step = publisher.round_step(round_plan)
        ranges.append((step.begin_blocks, step.end_blocks))
        for layer_idx in kda_layers:
            if step.terminal_indices:
                publisher.record_kda_layer(layer_idx, step)
        publisher.commit(step)
    publisher.validate_complete()
    return publisher, ranges


class KimiK3ChunkRdmaPublisherTest(unittest.TestCase):
    def test_64k_is_an_upper_bound_and_nonterminal_ends_align_to_v(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [65537],
            [0],
            chunk_budget=65536,
            alignment_tokens=1024,
        )

        self.assertEqual([round_plan.token_count for round_plan in rounds], [65536, 1])
        self.assertFalse(rounds[0].slices[0].terminal)
        self.assertEqual(rounds[0].slices[0].absolute_end, 65536)
        self.assertTrue(rounds[1].slices[0].terminal)
        self.assertEqual(rounds[1].slices[0].absolute_end, 65537)

    def test_multi_request_rounds_may_leave_budget_holes(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [2048, 2048],
            [0, 0],
            chunk_budget=1536,
            alignment_tokens=1024,
        )

        self.assertEqual([round_plan.token_count for round_plan in rounds], [1024] * 4)
        self.assertTrue(all(round_plan.token_count < 1536 for round_plan in rounds))
        for round_plan in rounds:
            for item in round_plan.slices:
                if not item.terminal:
                    self.assertEqual(item.absolute_end % 1024, 0)

    def test_planner_fails_when_budget_cannot_reach_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            plan_kimi_k3_chunk_rounds(
                [2048],
                [0],
                chunk_budget=512,
                alignment_tokens=1024,
            )

    def test_cache_publisher_owns_prefix_round_and_layer_publication(self) -> None:
        class FakeLayer:
            def __init__(self, is_kda: bool) -> None:
                self.is_kda = is_kda
                self.prepared = 0

            def prepare_kda_cache_store(self, layer_cache: object) -> None:
                self.prepared += 1

        class FakeCache:
            def get_layer_cache(self, layer_idx: int) -> int:
                return layer_idx

        layers = [FakeLayer(False), FakeLayer(True)]
        writes = []

        def writer(layer_cache: int, plan: object) -> None:
            writes.append(
                (
                    layer_cache,
                    tuple(plan.begin_block_host.tolist()),
                    tuple(plan.end_block_host.tolist()),
                    tuple(plan.terminal_host.tolist()),
                )
            )

        publisher = KimiK3ChunkRdmaPublisher(
            [700],
            [1024],
            transfer_page_tokens=128,
            kda_layer_indices=[1],
        )
        cache_publisher = KimiK3ChunkCachePublisher(
            writer=writer,
            publisher=publisher,
            layers=layers,
            kv_cache=FakeCache(),
        )
        cache_publisher.publish_prefix()
        for round_plan in plan_kimi_k3_chunk_rounds(
            [700], [1024], chunk_budget=1024, alignment_tokens=1024
        ):
            context = cache_publisher.begin_round(round_plan)
            self.assertIsNotNone(context)
            for layer_idx, layer in enumerate(layers):
                context.publish_layer(layer_idx, layer, layer_idx)
            cache_publisher.commit_round(context)
        cache_publisher.validate_complete()

        self.assertEqual([write[0] for write in writes], [0, 0, 1])
        self.assertEqual(writes[-1][1:], ((8,), (14,), (True,)))
        self.assertEqual(layers[1].prepared, 1)

    def test_multi_batch_frontiers_cover_each_page_and_tail_once(self) -> None:
        publisher, ranges = _publish_all(
            [1369, 1209, 1813],
            [0, 0, 0],
            budget=1024,
            alignment_tokens=1024,
            transfer_page_tokens=128,
        )

        self.assertEqual(publisher.frontier, (11, 10, 15))
        covered = [[], [], []]
        for begins, ends in ranges:
            for request_idx, (begin, end) in enumerate(zip(begins, ends)):
                covered[request_idx].extend(range(begin, end))
        self.assertEqual(covered, [list(range(11)), list(range(10)), list(range(15))])

    def test_prefix_hit_and_inactive_rows_keep_monotonic_frontiers(self) -> None:
        publisher, ranges = _publish_all(
            [600, 900],
            [1024, 0],
            budget=1024,
            alignment_tokens=1024,
            transfer_page_tokens=128,
        )

        self.assertEqual(ranges[0], ((0, 0), (8, 0)))
        self.assertEqual(publisher.frontier, (13, 8))
        self.assertTrue(any(begin[1] == end[1] for begin, end in ranges[1:]))

    def test_terminal_tail_is_not_exposed_by_nonterminal_step(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [1153], [0], chunk_budget=1024, alignment_tokens=1024
        )
        publisher = KimiK3ChunkRdmaPublisher(
            [1153],
            [0],
            transfer_page_tokens=128,
            kda_layer_indices=[1],
        )
        publisher.commit(publisher.prefix_step())

        first = publisher.round_step(rounds[0])
        self.assertEqual(first.begin_blocks, (0,))
        self.assertEqual(first.end_blocks, (8,))
        self.assertEqual(first.terminal, (False,))
        publisher.commit(first)

        tail = publisher.round_step(rounds[1])
        self.assertEqual(tail.begin_blocks, (8,))
        self.assertEqual(tail.end_blocks, (10,))
        self.assertEqual(tail.terminal, (True,))

    def test_stale_commit_and_duplicate_kda_are_rejected(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [1153], [0], chunk_budget=1024, alignment_tokens=1024
        )
        publisher = KimiK3ChunkRdmaPublisher(
            [1153],
            [0],
            transfer_page_tokens=128,
            kda_layer_indices=[1],
        )
        publisher.commit(publisher.prefix_step())
        first = publisher.round_step(rounds[0])
        publisher.commit(first)
        with self.assertRaisesRegex(RuntimeError, "stale frontier"):
            publisher.commit(first)
        tail = publisher.round_step(rounds[1])
        publisher.record_kda_layer(1, tail)
        with self.assertRaisesRegex(RuntimeError, "published twice"):
            publisher.record_kda_layer(1, tail)

    def test_chunkwise_switch(self) -> None:
        cases = [
            (None, False),
            ("0", False),
            ("1", True),
        ]
        for setting, expected in cases:
            with self.subTest(setting=setting):
                environment = {"CACHE_STORE_RDMA_MODE": "invalid"}
                if setting is not None:
                    environment["KIMI_K3_CHUNKWISE_RDMA"] = setting
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(chunkwise_rdma_enabled(), expected)

    def test_invalid_chunkwise_switch_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"KIMI_K3_CHUNKWISE_RDMA": "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "KIMI_K3_CHUNKWISE_RDMA must be 0 or 1"
            ):
                chunkwise_rdma_enabled()


if __name__ == "__main__":
    unittest.main()
