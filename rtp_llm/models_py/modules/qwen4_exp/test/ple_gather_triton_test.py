import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import Qwen4ExpNGramEmbedding


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
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_TRITON_PLE_GATHER": "0"}):
            expected = module._gather_local(ids)
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_TRITON_PLE_GATHER": "1"}):
            actual = module._gather_local(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_replay_uses_updated_ids(self):
        module = self._module()
        ids = torch.tensor([5, 0, 15], device="cuda", dtype=torch.int64)
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_TRITON_PLE_GATHER": "1"}):
            module._gather_local(ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = module._gather_local(ids)
            ids.copy_(torch.tensor([16, 9, 2], device="cuda"))
            graph.replay()
            with patch.dict(os.environ, {"RTP_LLM_QWEN4_TRITON_PLE_GATHER": "0"}):
                expected = module._gather_local(ids)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
