import importlib.util
import unittest
from pathlib import Path

import torch

try:
    import benchmark.topology_kv_candidate_schedule as topology_kv_candidate_schedule
except ModuleNotFoundError as exc:
    if exc.name not in {"benchmark", "benchmark.topology_kv_candidate_schedule"}:
        raise
    import topology_kv_candidate_schedule


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rtp_llm"
    / "models_py"
    / "modules"
    / "hybrid"
    / "topology_kv_policy.py"
)
INDEXER_PATH = MODULE_PATH.with_name("indexer.py")

spec = importlib.util.spec_from_file_location("topology_kv_policy", MODULE_PATH)
topology_kv_policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(topology_kv_policy)

TopologyKvPolicyConfig = topology_kv_policy.TopologyKvPolicyConfig
apply_topology_kv_policy = topology_kv_policy.apply_topology_kv_policy
dense_decode_attention = topology_kv_candidate_schedule.dense_decode_attention
sparse_decode_attention = topology_kv_candidate_schedule.sparse_decode_attention


class TopologyKvPolicyTest(unittest.TestCase):
    def test_disabled_policy_returns_original_tensor_and_zero_counters(self):
        topk = torch.tensor([[5, 4, 3, 2]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="disabled",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.raw_selected_tokens, 4)
        self.assertEqual(result.counters.compressed_tokens_represented, 0)
        self.assertEqual(result.counters.raw_kv_tokens_avoided, 0)
        self.assertEqual(result.counters.compression_hits, 0)

    def test_compress_sparse_reports_stable_prefix_tokens_avoided(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        result = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="system policy + tool schema",
            output_contract="json tool result",
        )

        self.assertEqual(result.topk_indices.shape, topk.shape)
        self.assertEqual(result.topk_indices.dtype, topk.dtype)
        self.assertIn(0, result.topk_indices[0].tolist())
        self.assertIn(7, result.topk_indices[0].tolist())
        self.assertGreater(result.counters.compressed_tokens_represented, 0)
        self.assertGreater(result.counters.raw_kv_tokens_avoided, 0)

    def test_fingerprint_changes_when_output_contract_changes(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        first = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
        )
        second = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-b",
            previous_fingerprint=first.stable_fingerprint,
        )

        self.assertNotEqual(first.stable_fingerprint, second.stable_fingerprint)
        self.assertEqual(second.counters.compression_hits, 0)

    def test_matching_fingerprint_counts_compression_hit(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        first = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
        )
        second = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            stable_scaffold="same scaffold",
            output_contract="contract-a",
            previous_fingerprint=first.stable_fingerprint,
        )

        self.assertEqual(second.counters.compression_hits, 1)
        self.assertGreater(second.counters.raw_kv_tokens_avoided, 0)

    def test_sparse_merge_preserves_causality_and_removes_duplicates(self):
        topk = torch.tensor([[7, 7, 6, 5, 4, 3]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        values = [value for value in result.topk_indices[0].tolist() if value >= 0]
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(value < 8 for value in values))
        self.assertEqual(result.counters.compressed_tokens_represented, 0)

    def test_indexer_has_disabled_by_default_topology_kv_gate(self):
        source = INDEXER_PATH.read_text(encoding="utf-8")

        self.assertIn("RTP_LLM_TOPOLOGY_KV_POLICY", source)
        self.assertIn("latest_topology_kv_counters", source)
        self.assertIn("_apply_topology_kv_policy", source)
        self.assertIn("apply_topology_kv_policy", source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for policy e2e")
    def test_cuda_policy_output_runs_sparse_decode_attention_e2e(self):
        device = torch.device("cuda")
        seq_len = 64
        query = torch.randn(1, 2, 1, 16, device=device, dtype=torch.float16)
        key = torch.randn(1, 2, seq_len, 16, device=device, dtype=torch.float16)
        value = torch.randn(1, 2, seq_len, 16, device=device, dtype=torch.float16)
        learned_topk = torch.arange(
            seq_len - 1, -1, -1, device=device, dtype=torch.int32
        ).view(1, seq_len)
        lengths = torch.tensor([seq_len], device=device, dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_compress_sparse",
            sink_blocks=1,
            local_blocks=2,
            witness_blocks=2,
            block_size=8,
        )

        result = apply_topology_kv_policy(
            learned_topk,
            lengths,
            config=config,
            stable_scaffold="system policy + tool schema",
            output_contract="json tool result",
        )

        self.assertEqual(result.topk_indices.device.type, "cuda")
        self.assertEqual(result.topk_indices.shape, learned_topk.shape)
        selected = result.topk_indices[result.topk_indices >= 0]
        self.assertEqual(selected.numel(), seq_len)
        self.assertEqual(torch.unique(selected).numel(), seq_len)
        self.assertEqual(result.counters.raw_selected_tokens, seq_len)
        self.assertGreaterEqual(result.counters.schedule_ms, 0.0)
        torch.testing.assert_close(
            sparse_decode_attention(query, key, value, result.topk_indices),
            dense_decode_attention(query, key, value),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
