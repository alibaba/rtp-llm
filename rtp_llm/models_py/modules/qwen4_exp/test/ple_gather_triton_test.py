import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PLEGatherTritonTest(unittest.TestCase):
    def _module(self):
        device = torch.device("cuda")
        shards = [
            torch.arange(5 * 160, device=device, dtype=torch.float32)
            .reshape(5, 160)
            .add(i * 1000)
            .to(torch.bfloat16)
            for i in range(2)
        ]
        module = Qwen4ExpNGramEmbedding(
            shards,
            torch.tensor([8, 8], dtype=torch.int64),
            torch.tensor([0, 8], dtype=torch.int64),
            torch.tensor([3, 5], dtype=torch.int64),
            ngram_size=2,
            eos_token_id=0,
            shard_indices=[1, 3],
            total_shards=4,
            distributed_reduce=True,
        )
        return module

    def test_local_and_remote_shards_match_boolean_indexing(self):
        module = self._module()
        ids = torch.tensor([0, 5, 7, 9, 10, 16, 19], device="cuda").reshape(1, 7, 1)
        expected = module._gather_torch(ids)
        actual = module._gather_local(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_replay_uses_updated_ids(self):
        module = self._module()
        ids = torch.tensor([5, 0, 15], device="cuda", dtype=torch.int64)
        module._gather_local(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module._gather_local(ids)
        ids.copy_(torch.tensor([16, 9, 2], device="cuda"))
        graph.replay()
        expected = module._gather_torch(ids)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_cpu_fp8_prefill_tiling_preserves_all_bit_patterns(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_gather_triton import (
            gather_local,
            prepare_cpu_uva_shards,
        )

        raw = (
            torch.arange(256, dtype=torch.int32)
            .to(torch.uint8)
            .repeat(5)
            .reshape(8, 160)
        )
        shards, pointers = prepare_cpu_uva_shards(
            [raw.view(torch.float8_e4m3fn)],
            [0],
            2,
            torch.device("cuda", torch.cuda.current_device()),
        )
        for n in (17, 65536, 131072, 196608):
            ids = torch.arange(n, device="cuda", dtype=torch.int64) % 20 - 2
            for value in (0.00003125, 0.119713, 8.0):
                scale = torch.tensor([value], device="cuda")
                expected = torch.zeros((n, 160), device="cuda", dtype=torch.bfloat16)
                valid = (ids >= 0) & (ids < 8)
                expected[valid] = (
                    raw.to(device="cuda")
                    .view(torch.float8_e4m3fn)
                    .to(torch.bfloat16)
                    .index_select(0, ids[valid])
                    .mul(scale.to(torch.bfloat16))
                )
                actual = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
                self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
                finite = ~torch.isnan(expected)
                self.assertTrue(
                    torch.equal(
                        actual[finite].view(torch.int16),
                        expected[finite].view(torch.int16),
                    )
                )

    def test_cpu_fp8_prefill_tiling_graph_refreshes_ids_and_scale(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_gather_triton import (
            gather_local,
            prepare_cpu_uva_shards,
        )

        shard = (
            torch.arange(8 * 160, dtype=torch.float32)
            .reshape(8, 160)
            .remainder(63)
            .to(torch.float8_e4m3fn)
        )
        shards, pointers = prepare_cpu_uva_shards(
            [shard], [0], 2, torch.device("cuda", torch.cuda.current_device())
        )
        ids = torch.arange(65536, device="cuda", dtype=torch.int64) % 20 - 2
        scale = torch.tensor([0.119713], device="cuda")
        for _ in range(3):
            gather_local(ids, shards, (0,), 2, pointers, scale=scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
        for generation in range(8):
            ids.copy_(
                (torch.arange(65536, device="cuda", dtype=torch.int64) + generation)
                % 20
                - 2
            )
            scale.fill_((generation + 1) * 0.119713)
            graph.replay()
            expected = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
        graph.reset()


if __name__ == "__main__":
    unittest.main()
