"""Bounded workspace, output ownership, and real four-rank chunk scheduling."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.chunking import (
    MegaMoeChunker,
    mega_moe_chunk_plan,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig, RoleType

_MODULE = "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.chunking"


def _distributed_worker(rank, rendezvous):
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        chunker = MegaMoeChunker(8, 8, dist.group.WORLD)
        layers = [SimpleNamespace(mlp=SimpleNamespace(_mega_moe_chunker=chunker))] * 2
        # Both imbalance directions, empty ranks, exact boundaries, and all-idle.
        for counts in ([20, 9, 1, 0], [0, 1, 9, 20], [8, 8, 8, 8], [0, 0, 0, 0]):
            x = torch.arange(counts[rank] * 2, dtype=torch.float32).reshape(-1, 2)
            scratch = torch.empty(8, 2)
            rounds = []
            preparations = []

            def kernel(part):
                n = part.shape[0]
                sizes = [None] * 4
                dist.all_gather_object(sizes, n)
                rounds.append(sizes)
                # All peers must execute this round even with no local rows.
                scratch[:n].copy_(part * 2)
                return scratch[:n]

            def prepare(full):
                preparations.append(full.shape[0])
                return lambda begin, end: kernel(full[begin:end])

            with mega_moe_chunk_plan(layers, x):
                for _ in layers:
                    y = chunker.forward(x, kernel, prepare)
                    torch.testing.assert_close(y, x * 2)
            assert preparations == ([counts[rank]] * 2 if max(counts) > 8 else [])
            expected = max(1, (max(counts) + 7) // 8)
            assert len(rounds) == 2 * expected
            for i, sizes in enumerate(rounds):
                assert sizes == [min(8, max(t - (i % expected) * 8, 0)) for t in counts]
    finally:
        dist.destroy_process_group()


class MegaMoeChunkingTest(unittest.TestCase):
    def test_reused_workspace_is_copied_before_next_chunk(self):
        chunker = MegaMoeChunker(4, 4, object())
        for tokens in (0, 1, 4, 5, 13):
            with self.subTest(tokens=tokens):
                x = torch.arange(tokens * 3, dtype=torch.float32).reshape(-1, 3)
                scratch = torch.empty(4, 3)
                calls = []

                def kernel(part):
                    n = part.shape[0]
                    calls.append(n)
                    scratch[:n].copy_(part + 7)
                    return scratch[:n]

                with patch.object(chunker, "max_tokens", return_value=13):
                    y = chunker.forward(x, kernel)
                torch.testing.assert_close(y, x + 7)
                self.assertEqual(len(calls), 4)
                self.assertLessEqual(max(calls), 4)

    def test_stack_synchronizes_once_and_resets_after_exception(self):
        chunker = MegaMoeChunker(4, 4, object())
        layers = [SimpleNamespace(mlp=SimpleNamespace(_mega_moe_chunker=chunker))]
        x = torch.zeros(2, 3)
        with patch(_MODULE + ".dist.all_reduce") as reduce:
            reduce.side_effect = lambda count, **kwargs: count.fill_(9)
            with self.assertRaisesRegex(RuntimeError, "test"):
                with mega_moe_chunk_plan(layers, x):
                    self.assertEqual(chunker.max_tokens(2, x.device), 9)
                    self.assertEqual(chunker.max_tokens(2, x.device), 9)
                    with self.assertRaisesRegex(RuntimeError, "token count changed"):
                        chunker.max_tokens(3, x.device)
                    raise RuntimeError("test")
            self.assertEqual(reduce.call_count, 1)
            chunker.max_tokens(2, x.device)
            self.assertEqual(reduce.call_count, 2)
            self.assertIs(reduce.call_args.kwargs["group"], chunker.group)

    def test_graph_does_not_synchronize(self):
        chunker = MegaMoeChunker(4, 8, object())
        x = torch.ones(6, 2)
        with patch(_MODULE + "._capturing", return_value=True), patch.object(
            chunker, "max_tokens"
        ) as sync:
            torch.testing.assert_close(chunker.forward(x, lambda part: part * 2), x * 2)
            with self.assertRaisesRegex(ValueError, "exceed capacity"):
                chunker.forward(torch.ones(9, 2), lambda part: part)
            sync.assert_not_called()

    def test_four_rank_uneven_empty_and_repeated_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_worker,
                args=(os.path.join(directory, "rendezvous"),),
                nprocs=4,
                join=True,
            )


class MegaMoeChunkCapacityTest(unittest.TestCase):
    def adapter(self, strategy="mega_moe_fp8", role=RoleType.PREFILL, length=1048576):
        config = ModelConfig()
        config.max_seq_len = length
        config.moe_prefill_max_tokens_per_rank = 4 * length
        parallel = ParallelismConfig()
        parallel.role_type = role
        moe = MoeConfig()
        moe.moe_strategy = strategy
        moe.ll_num_max_token = 128
        return MoEConfigAdapter(config, parallel, moe, max_generate_batch_size=256)

    def test_capacity_is_independent_of_long_sequence_length(self):
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se"):
            for length in (32768, 1048576):
                with self.subTest(strategy=strategy, length=length), patch.dict(
                    os.environ, {"RTP_MEGAMOE_CHUNK_TOKENS": "8192"}
                ):
                    adapter = self.adapter(strategy, length=length)
                    self.assertEqual(adapter.max_tokens_per_rank, 8192)
                    self.assertEqual(adapter.mega_moe_chunk_tokens, 8192)
                    self.assertTrue(adapter.warmup_include_capacity)

    def test_decode_other_backends_and_disabled_chunking(self):
        with patch.dict(os.environ, {"RTP_MEGAMOE_CHUNK_TOKENS": "8192"}):
            adapter = self.adapter(role=RoleType.DECODE)
            self.assertEqual(adapter.max_tokens_per_rank, 256)
            self.assertEqual(adapter.mega_moe_chunk_tokens, 0)
            self.assertEqual(self.adapter("auto").max_tokens_per_rank, 4 * 1048576)
        with patch.dict(os.environ, {"RTP_MEGAMOE_CHUNK_TOKENS": "0"}):
            self.assertEqual(self.adapter().max_tokens_per_rank, 4 * 1048576)
        with patch.dict(os.environ, {"RTP_MEGAMOE_CHUNK_TOKENS": "-1"}):
            with self.assertRaises(ValueError):
                self.adapter()


if __name__ == "__main__":
    unittest.main()
