import unittest

import torch

try:
    from benchmark.topology_kv_candidate_schedule import benchmark_decode_attention
except ModuleNotFoundError as exc:
    if exc.name not in {"benchmark", "benchmark.topology_kv_candidate_schedule"}:
        raise
    from topology_kv_candidate_schedule import benchmark_decode_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for speed testing")
class TopologyKVCandidateScheduleCUDATest(unittest.TestCase):
    def test_sparse_attention_benchmark_runs_with_topology_schedule(self):
        result = benchmark_decode_attention(
            seq_len=16384,
            selected_tokens=512,
            heads=16,
            head_dim=64,
            rounds=60,
            warmup=20,
            dtype=torch.float16,
            device="cuda",
        )

        self.assertEqual(result.seq_len, 16384)
        self.assertEqual(result.selected_tokens, 512)
        self.assertGreater(result.dense_ms, 0.0)
        self.assertGreater(result.sparse_ms, 0.0)
        self.assertGreater(result.speedup, 0.0)


if __name__ == "__main__":
    unittest.main()
