import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

compute_ops = types.ModuleType("rtp_llm.ops.compute_ops")
compute_ops.PyAttentionInputs = type("PyAttentionInputs", (), {})
compute_ops.PyCacheStorePublishPlan = type("PyCacheStorePublishPlan", (), {})
compute_ops.PyModelInputs = type("PyModelInputs", (), {})
rtp_llm_package = types.ModuleType("rtp_llm")
rtp_llm_package.__path__ = [str(Path(__file__).resolve().parents[4])]
ops_package = types.ModuleType("rtp_llm.ops")
ops_package.__path__ = []
ops_package.CPRotateMethod = type("CPRotateMethod", (), {})
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
build_chunk_attention_inputs = chunk_prefill.build_chunk_attention_inputs
logical_chunk_round = chunk_prefill.logical_chunk_round
plan_kimi_k3_chunk_rounds = chunk_prefill.plan_kimi_k3_chunk_rounds


def _publish_all(
    input_lengths: list[int],
    prefix_lengths: list[int],
    *,
    budget: int,
    page_size: int,
) -> tuple[
    KimiK3ChunkRdmaPublisher,
    list[tuple[tuple[int, ...], tuple[int, ...]]],
]:
    kda_layers = (0, 3)
    publisher = KimiK3ChunkRdmaPublisher(
        input_lengths,
        prefix_lengths,
        page_size=page_size,
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
        page_size=page_size,
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
    def test_padded_prefill_round_rebuilds_logical_layout(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [4991, 1], [0, 0], chunk_budget=4096, page_size=4096
        )
        self.assertEqual([round_plan.token_count for round_plan in rounds], [4096, 896])

        attention_inputs = types.SimpleNamespace(
            input_lengths_host=torch.tensor([4991, 1], dtype=torch.int32),
            logical_request_count=1,
            kv_cache_block_id_host=None,
            kv_cache_kernel_block_id_host=None,
            kv_cache_kernel_block_id_device=None,
            kv_cache_block_id_host_by_group=[],
            kv_cache_kernel_block_id_host_by_group=[],
            kv_cache_kernel_block_id_device_by_group=[],
        )
        first = build_chunk_attention_inputs(
            attention_inputs, round_plan=rounds[0], device=torch.device("cpu")
        )
        self.assertEqual(first.logical_request_count, 1)
        self.assertEqual(first.physical_request_count, 1)
        self.assertEqual(first.logical_token_count, 4096)
        self.assertEqual(first.physical_token_count, 4096)
        self.assertFalse(first.is_s_padded)

        final = build_chunk_attention_inputs(
            attention_inputs, round_plan=rounds[1], device=torch.device("cpu")
        )
        self.assertEqual(final.logical_request_count, 1)
        self.assertEqual(final.physical_request_count, 2)
        self.assertEqual(final.logical_token_count, 895)
        self.assertEqual(final.physical_token_count, 896)
        self.assertTrue(final.is_s_padded)
        logical_final = logical_chunk_round(rounds[1], 1)
        self.assertEqual(logical_final.token_count, 895)
        self.assertEqual(
            [item.original_batch_idx for item in logical_final.slices], [0]
        )

        publisher = KimiK3ChunkRdmaPublisher(
            [4991], [0], page_size=4096, kda_layer_indices=[]
        )
        publisher.commit(publisher.prefix_step())
        for round_plan in rounds:
            step = publisher.round_step(logical_chunk_round(round_plan, 1))
            publisher.commit(step)
        publisher.validate_complete()
        self.assertEqual(publisher.frontier, (2,))

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
            [700], [512], page_size=512, kda_layer_indices=[1]
        )
        cache_publisher = KimiK3ChunkCachePublisher(
            writer=writer,
            publisher=publisher,
            layers=layers,
            kv_cache=FakeCache(),
        )
        cache_publisher.publish_prefix()
        for round_plan in plan_kimi_k3_chunk_rounds(
            [700], [512], chunk_budget=512, page_size=512
        ):
            context = cache_publisher.begin_round(round_plan)
            self.assertIsNotNone(context)
            for layer_idx, layer in enumerate(layers):
                context.publish_layer(layer_idx, layer, layer_idx)
            cache_publisher.commit_round(context)
        cache_publisher.validate_complete()

        self.assertEqual([write[0] for write in writes], [0, 0, 0, 1])
        self.assertEqual(writes[-1][1:], ((2,), (3,), (True,)))
        self.assertEqual(layers[1].prepared, 1)

    def test_multi_batch_frontiers_cover_each_page_and_tail_once(self) -> None:
        publisher, ranges = _publish_all(
            [1369, 1209, 1813],
            [0, 0, 0],
            budget=1024,
            page_size=512,
        )

        self.assertEqual(publisher.frontier, (3, 3, 4))
        covered = [[], [], []]
        for begins, ends in ranges:
            for request_idx, (begin, end) in enumerate(zip(begins, ends)):
                covered[request_idx].extend(range(begin, end))
        self.assertEqual(covered, [list(range(3)), list(range(3)), list(range(4))])

    def test_prefix_hit_and_inactive_rows_keep_monotonic_frontiers(self) -> None:
        publisher, ranges = _publish_all(
            [600, 900],
            [1024, 0],
            budget=512,
            page_size=512,
        )

        self.assertEqual(ranges[0], ((0, 0), (2, 0)))
        self.assertEqual(publisher.frontier, (4, 2))
        self.assertTrue(any(begin[1] == end[1] for begin, end in ranges[1:]))

    def test_terminal_tail_is_not_exposed_by_nonterminal_step(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds([700], [0], chunk_budget=512, page_size=512)
        publisher = KimiK3ChunkRdmaPublisher(
            [700], [0], page_size=512, kda_layer_indices=[1]
        )
        publisher.commit(publisher.prefix_step())

        first = publisher.round_step(rounds[0])
        self.assertEqual(first.begin_blocks, (0,))
        self.assertEqual(first.end_blocks, (1,))
        self.assertEqual(first.terminal, (False,))
        publisher.commit(first)

        tail = publisher.round_step(rounds[1])
        self.assertEqual(tail.begin_blocks, (1,))
        self.assertEqual(tail.end_blocks, (2,))
        self.assertEqual(tail.terminal, (True,))

    def test_stale_commit_and_duplicate_kda_are_rejected(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds([700], [0], chunk_budget=512, page_size=512)
        publisher = KimiK3ChunkRdmaPublisher(
            [700], [0], page_size=512, kda_layer_indices=[1]
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

if __name__ == "__main__":
    unittest.main()
