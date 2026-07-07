import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
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


def _load_indexer_for_unit_test():
    indexer_path = MODULE_PATH.with_name("indexer.py")
    module_names = [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.factory",
        "rtp_llm.models_py.modules.hybrid",
        "rtp_llm.models_py.modules.hybrid.topology_kv_policy",
        "rtp_llm.ops",
        "rtp_llm.ops.compute_ops",
        "rtp_llm.utils",
        "rtp_llm.utils.model_weight",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    modules = types.ModuleType("rtp_llm.models_py.modules")
    modules.IndexerOp = object
    modules.LayerNorm = object
    factory = types.ModuleType("rtp_llm.models_py.modules.factory")
    factory.LinearFactory = object
    ops = types.ModuleType("rtp_llm.ops")
    ops.AttentionConfigs = object
    ops.HWKernelConfig = object
    ops.ParallelismConfig = object
    compute_ops = types.ModuleType("rtp_llm.ops.compute_ops")
    compute_ops.KVCache = object
    model_weight = types.ModuleType("rtp_llm.utils.model_weight")
    model_weight.W = object

    sys.modules["rtp_llm"] = types.ModuleType("rtp_llm")
    sys.modules["rtp_llm.models_py"] = types.ModuleType("rtp_llm.models_py")
    sys.modules["rtp_llm.models_py.modules"] = modules
    sys.modules["rtp_llm.models_py.modules.factory"] = factory
    sys.modules["rtp_llm.models_py.modules.hybrid"] = types.ModuleType(
        "rtp_llm.models_py.modules.hybrid"
    )
    sys.modules["rtp_llm.models_py.modules.hybrid.topology_kv_policy"] = (
        topology_kv_policy
    )
    sys.modules["rtp_llm.ops"] = ops
    sys.modules["rtp_llm.ops.compute_ops"] = compute_ops
    sys.modules["rtp_llm.utils"] = types.ModuleType("rtp_llm.utils")
    sys.modules["rtp_llm.utils.model_weight"] = model_weight

    try:
        indexer_spec = importlib.util.spec_from_file_location(
            "unit_test_indexer", indexer_path
        )
        indexer_module = importlib.util.module_from_spec(indexer_spec)
        assert indexer_spec.loader is not None
        indexer_spec.loader.exec_module(indexer_module)
        return indexer_module.Indexer
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


Indexer = _load_indexer_for_unit_test()


class _PrefillCpConfig:
    def __init__(self, enabled):
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


class _FakeIndexerOp:
    def __init__(self, topk_result):
        self.topk_result = topk_result
        self.called = None

    def _get_topk_paged(self, *args):
        self.called = "decode"
        return self.topk_result

    def _get_topk_ragged(self, *args):
        self.called = "ragged"
        return self.topk_result

    def _get_topk_ragged_cp(self, *args):
        self.called = "cp"
        return self.topk_result


def _make_indexer(topk_result, *, cp_enabled=False):
    indexer = object.__new__(Indexer)
    indexer.indexer_op = _FakeIndexerOp(topk_result)
    indexer.topology_kv_policy = "topology_sparse_merge"
    indexer.topology_sink_blocks = 1
    indexer.topology_local_blocks = 1
    indexer.topology_witness_blocks = 0
    indexer.topology_stable_scaffold = None
    indexer.topology_output_contract = None
    indexer.latest_topology_kv_counters = None
    indexer.latest_topology_kv_fingerprint = None
    indexer.blocksize = 4
    indexer.parallelism_config = SimpleNamespace(
        prefill_cp_config=_PrefillCpConfig(cp_enabled)
    )
    return indexer


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
        self.assertEqual(result.counters.unselected_stable_tokens, 0)
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
        self.assertGreater(result.counters.unselected_stable_tokens, 0)

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
        self.assertGreater(second.counters.unselected_stable_tokens, 0)

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

    def test_rejects_unknown_policy_name(self):
        with self.assertRaisesRegex(ValueError, "unknown topology KV policy"):
            TopologyKvPolicyConfig(
                policy="typo_sparse",
                sink_blocks=1,
                local_blocks=1,
                witness_blocks=0,
                block_size=4,
            )

    def test_rejects_learned_topk_outside_absolute_contract(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        row_starts = torch.tensor([10], dtype=torch.int32)
        topk_indices_offset = torch.tensor([100], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "outside valid topology KV range"):
            apply_topology_kv_policy(
                topk,
                lengths,
                config=config,
                row_starts=row_starts,
                topk_indices_offset=topk_indices_offset,
            )

    def test_indexer_compute_topk_decode_bypasses_token_space_policy(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        fmha_params = SimpleNamespace(expanded_seq_lens=torch.tensor([8]))
        attention_inputs = SimpleNamespace(is_prefill=False)

        result = indexer._compute_topk(
            None, None, None, fmha_params, attention_inputs, None
        )

        self.assertEqual(indexer.indexer_op.called, "decode")
        self.assertIs(result, topk)
        self.assertIsNone(indexer.latest_topology_kv_counters)

    def test_indexer_compute_topk_ragged_requires_offset_absolute_coordinates(self):
        topk = torch.tensor([[117, 116, 115, 114]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=torch.tensor([8]),
            ks=torch.tensor([10]),
            topk_indices_offset=torch.tensor([100]),
        )
        attention_inputs = SimpleNamespace(is_prefill=True)

        result = indexer._compute_topk(
            None, None, None, fmha_params, attention_inputs, None
        )

        self.assertEqual(indexer.indexer_op.called, "ragged")
        values = [value for value in result[0].tolist() if value >= 0]
        self.assertTrue(all(110 <= value < 118 for value in values))

    def test_indexer_compute_topk_cp_requires_offset_absolute_coordinates(self):
        topk = torch.tensor([[117, 116, 115, 114]], dtype=torch.int32)
        indexer = _make_indexer(topk, cp_enabled=True)
        fmha_params = SimpleNamespace()
        attention_inputs = SimpleNamespace(is_prefill=True)
        cp_params = SimpleNamespace(
            total_local_ids=None,
            cu_kv_seqlens_global=None,
            total_kv_len=None,
            precomputed_ks=torch.tensor([10]),
            precomputed_ke=torch.tensor([18]),
            precomputed_lengths=torch.tensor([8]),
            precomputed_topk_off=torch.tensor([100]),
        )

        result = indexer._compute_topk(
            None, None, None, fmha_params, attention_inputs, cp_params
        )

        self.assertEqual(indexer.indexer_op.called, "cp")
        values = [value for value in result[0].tolist() if value >= 0]
        self.assertTrue(all(110 <= value < 118 for value in values))

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
