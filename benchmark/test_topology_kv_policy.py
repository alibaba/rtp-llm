import importlib.util
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "rtp_llm" / "models_py" / "modules" / "hybrid" / "topology_kv_policy.py"
)
INDEXER_PATH = MODULE_PATH.with_name("indexer.py")
METRICS_REPORTER_PATH = (
    REPO_ROOT / "rtp_llm" / "metrics" / "kmonitor_metric_reporter.py"
)

spec = importlib.util.spec_from_file_location("topology_kv_policy", MODULE_PATH)
topology_kv_policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(topology_kv_policy)

metrics_spec = importlib.util.spec_from_file_location(
    "topology_kv_metric_enums", METRICS_REPORTER_PATH
)
topology_kv_metric_enums = importlib.util.module_from_spec(metrics_spec)
assert metrics_spec.loader is not None
metrics_spec.loader.exec_module(topology_kv_metric_enums)

TopologyKvPolicyConfig = topology_kv_policy.TopologyKvPolicyConfig
apply_topology_kv_policy = topology_kv_policy.apply_topology_kv_policy


def _load_indexer_for_unit_test():
    module_names = [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.factory",
        "rtp_llm.models_py.modules.hybrid",
        "rtp_llm.models_py.modules.hybrid.topology_kv_policy",
        "rtp_llm.metrics",
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
    metrics = types.ModuleType("rtp_llm.metrics")
    metrics.AccMetrics = topology_kv_metric_enums.AccMetrics
    metrics.GaugeMetrics = topology_kv_metric_enums.GaugeMetrics
    metrics.kmonitor = SimpleNamespace(report=lambda *args, **kwargs: None)
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
    sys.modules["rtp_llm.metrics"] = metrics
    sys.modules["rtp_llm.ops"] = ops
    sys.modules["rtp_llm.ops.compute_ops"] = compute_ops
    sys.modules["rtp_llm.utils"] = types.ModuleType("rtp_llm.utils")
    sys.modules["rtp_llm.utils.model_weight"] = model_weight

    try:
        indexer_spec = importlib.util.spec_from_file_location(
            "unit_test_indexer", INDEXER_PATH
        )
        indexer_module = importlib.util.module_from_spec(indexer_spec)
        assert indexer_spec.loader is not None
        indexer_spec.loader.exec_module(indexer_module)
        return indexer_module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


indexer_module = _load_indexer_for_unit_test()
Indexer = indexer_module.Indexer
_topology_env_int = indexer_module._topology_env_int
_topology_env_float = indexer_module._topology_env_float


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
    indexer.layer_idx = 7
    indexer.indexer_op = _FakeIndexerOp(topk_result)
    indexer.topology_kv_policy = "topology_sparse_merge"
    indexer.topology_sink_blocks = 1
    indexer.topology_local_blocks = 1
    indexer.topology_witness_blocks = 0
    indexer.topology_max_policy_tokens = 8192
    indexer.topology_max_topk_elements = 262144
    indexer.topology_max_topk_width = 8192
    indexer.topology_max_structural_fraction = 0.5
    indexer.topology_coordinate_mismatch_action = "fallback_disabled"
    indexer.topology_stable_scaffold = None
    indexer.topology_output_contract = None
    indexer.blocksize = 4
    indexer.parallelism_config = SimpleNamespace(
        prefill_cp_config=_PrefillCpConfig(cp_enabled)
    )
    return indexer


class TopologyKvPolicyTest(unittest.TestCase):
    def test_indexer_bypass_metric_is_a_layer_event_with_bounded_layer_tag(self):
        topk = torch.arange(2048, dtype=torch.int32).view(1, -1)
        indexer = _make_indexer(topk)

        with mock.patch.object(indexer_module.logging, "warning"):
            with mock.patch.object(indexer_module.kmonitor, "report") as report:
                indexer._apply_topology_kv_policy(
                    topk, torch.tensor([8193], dtype=torch.int32)
                )

        bypass_call = next(
            call
            for call in report.call_args_list
            if call.args[0]
            is indexer_module.AccMetrics.TOPOLOGY_KV_POLICY_BYPASS_LAYER_EVENT_METRIC
        )
        self.assertEqual(
            bypass_call.args[0].value,
            "py_rtp_topology_kv_policy_bypass_layer_forward_event",
        )
        self.assertEqual(bypass_call.args[2]["layer_idx"], "7")

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
        self.assertIn(3, result.topk_indices[0].tolist())
        self.assertIn(7, result.topk_indices[0].tolist())
        self.assertGreater(result.counters.compressed_tokens_represented, 0)
        self.assertGreater(result.counters.unselected_stable_tokens, 0)
        self.assertGreater(result.counters.learned_kept_tokens, 0)

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
        self.assertGreater(result.counters.learned_kept_tokens, 0)

    def test_structural_fraction_keeps_part_of_learned_budget(self):
        topk = torch.tensor([[11, 10, 9, 8]], dtype=torch.int32)
        lengths = torch.tensor([12], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            max_structural_fraction=0.5,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        values = result.topk_indices[0].tolist()
        self.assertEqual(len([value for value in values if value in {3, 11}]), 2)
        self.assertGreaterEqual(result.counters.learned_kept_tokens, 2)
        self.assertEqual(result.counters.learned_evicted_tokens, 1)

    def test_topology_only_uses_full_structural_budget(self):
        topk = torch.tensor([[-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
        lengths = torch.tensor([6], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_only",
            sink_blocks=3,
            local_blocks=0,
            witness_blocks=0,
            block_size=2,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        values = [value for value in result.topk_indices[0].tolist() if value >= 0]
        self.assertEqual(len(values), min(topk.size(1), int(lengths[0].item())))
        self.assertEqual(set(values), set(range(6)))

    def test_topology_only_rejects_zero_structural_sources(self):
        with self.assertRaisesRegex(
            ValueError, "topology_only requires at least one structural block source"
        ):
            TopologyKvPolicyConfig(
                policy="topology_only",
                sink_blocks=0,
                local_blocks=0,
                witness_blocks=0,
                block_size=4,
            )

    def test_topology_only_single_token_yields_valid_candidate(self):
        topk = torch.tensor([[-1]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_only",
            sink_blocks=1,
            local_blocks=0,
            witness_blocks=0,
            block_size=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertEqual(result.topk_indices.tolist(), [[0]])
        self.assertEqual(result.counters.raw_selected_tokens, 1)

    def test_topology_only_rejects_nonempty_row_without_candidate_capacity(self):
        topk = torch.empty((1, 0), dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_only",
            sink_blocks=1,
            local_blocks=0,
            witness_blocks=0,
            block_size=4,
        )

        with self.assertRaisesRegex(
            RuntimeError, "topology_only produced no valid candidate"
        ):
            apply_topology_kv_policy(topk, lengths, config=config)

    def test_merge_output_satisfies_sparse_kernel_layout_contract(self):
        topk = torch.tensor(
            [
                [2, 2, -1, -1, -1],
                [105, 104, 104, -1, -1],
            ],
            dtype=torch.int32,
        )
        lengths = torch.tensor([3, 6], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            max_structural_fraction=0.5,
        )

        result = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            topk_indices_offset=torch.tensor([0, 100], dtype=torch.int32),
        )

        self.assertEqual(result.topk_indices.device.type, "cpu")
        self.assertEqual(result.topk_indices.shape, topk.shape)
        for row, start, length in zip(result.topk_indices.tolist(), [0, 100], [3, 6]):
            seen_padding = False
            valid_values = []
            for value in row:
                if value < 0:
                    seen_padding = True
                    continue
                self.assertFalse(seen_padding, "valid index after padding")
                self.assertTrue(start <= value < start + length)
                valid_values.append(value)
            self.assertEqual(len(valid_values), len(set(valid_values)))

    def test_rejects_unknown_policy_name(self):
        with self.assertRaisesRegex(ValueError, "unknown topology KV policy"):
            TopologyKvPolicyConfig(
                policy="typo_sparse",
                sink_blocks=1,
                local_blocks=1,
                witness_blocks=0,
                block_size=4,
            )

    def test_normalizes_policy_name_case(self):
        config = TopologyKvPolicyConfig(
            policy="Topology_Sparse_Merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        self.assertEqual(config.policy, "topology_sparse_merge")

    def test_normalizes_coordinate_mismatch_action_case(self):
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            coordinate_mismatch_action="Fallback_Disabled",
        )

        self.assertEqual(config.coordinate_mismatch_action, "fallback_disabled")

    def test_normalizes_coordinate_mismatch_action_whitespace(self):
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            coordinate_mismatch_action=" Raise ",
        )

        self.assertEqual(config.coordinate_mismatch_action, "raise")

    def test_rejects_unknown_coordinate_mismatch_action(self):
        with self.assertRaisesRegex(ValueError, "coordinate_mismatch_action"):
            TopologyKvPolicyConfig(
                policy="topology_sparse_merge",
                sink_blocks=1,
                local_blocks=1,
                witness_blocks=0,
                block_size=4,
                coordinate_mismatch_action="warn",
            )

    def test_rejects_non_integer_topk_indices(self):
        topk = torch.tensor([[7.5, 6.0, 5.0, 4.0]], dtype=torch.float32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        with self.assertRaisesRegex(
            ValueError, "topk_indices must use int32 or int64 dtype"
        ):
            apply_topology_kv_policy(topk, lengths, config=config)

    def test_rejects_unsigned_or_narrow_topk_indices(self):
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        for dtype in (torch.uint8, torch.int8, torch.int16):
            with self.subTest(dtype=dtype):
                topk = torch.tensor([[7, 6, 5, 4]], dtype=dtype)
                with self.assertRaisesRegex(
                    ValueError, "topk_indices must use int32 or int64 dtype"
                ):
                    apply_topology_kv_policy(topk, lengths, config=config)

    def test_bypasses_python_policy_when_topk_elements_exceed_limit(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            max_topk_elements=3,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.policy_bypassed, 1)
        self.assertEqual(result.bypass_reasons, ("topk_elements_limit",))

    def test_default_limits_apply_65_row_production_width_prefill(self):
        topk = torch.full((65, 2048), -1, dtype=torch.int32)
        lengths = torch.arange(1, 66, dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_only",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=64,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertEqual(result.counters.policy_bypassed, 0)
        self.assertIsNot(result.topk_indices, topk)
        self.assertTrue(torch.all((result.topk_indices >= 0).any(dim=1)))

    def test_default_limits_apply_128_row_production_width_prefill(self):
        topk = torch.full((128, 2048), -1, dtype=torch.int32)
        lengths = torch.arange(1, 129, dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_only",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=64,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertEqual(result.counters.policy_bypassed, 0)
        self.assertIsNot(result.topk_indices, topk)
        self.assertTrue(torch.all((result.topk_indices >= 0).any(dim=1)))

    def test_bypasses_python_policy_when_topk_width_exceeds_limit(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            max_topk_width=3,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.policy_bypassed, 1)
        self.assertEqual(result.bypass_reasons, ("topk_width_limit",))

    def test_detaches_block_drift_scores_before_structural_selection(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        drift_scores = torch.ones((1, 2), dtype=torch.float32, requires_grad=True)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=1,
            block_size=4,
        )
        seen_requires_grad = []
        original_structural_tokens = topology_kv_policy._structural_tokens

        def _record_structural_tokens(length, config, drift_row, budget):
            seen_requires_grad.append(
                None if drift_row is None else drift_row.requires_grad
            )
            return original_structural_tokens(length, config, drift_row, budget)

        with mock.patch.object(
            topology_kv_policy, "_structural_tokens", _record_structural_tokens
        ):
            apply_topology_kv_policy(
                topk,
                lengths,
                config=config,
                block_drift_scores=drift_scores,
            )

        self.assertEqual(seen_requires_grad, [False])

    def test_indexer_normalizes_coordinate_mismatch_action(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        indexer.topology_coordinate_mismatch_action = "FALLBACK_DISABLED"

        with mock.patch.object(indexer_module.logging, "warning"):
            with mock.patch.object(indexer_module.kmonitor, "report") as report:
                result = indexer._apply_topology_kv_policy(
                    topk,
                    torch.tensor([8], dtype=torch.int32),
                    row_starts=torch.tensor([10], dtype=torch.int32),
                    topk_indices_offset=torch.tensor([100], dtype=torch.int32),
                )

        self.assertIs(result, topk)
        fallback_call = next(
            call
            for call in report.call_args_list
            if call.args[0]
            is indexer_module.AccMetrics.TOPOLOGY_KV_COORDINATE_FALLBACK_LAYER_EVENT_METRIC
        )
        self.assertEqual(fallback_call.args[2]["action"], "fallback_disabled")
        self.assertEqual(fallback_call.args[2]["layer_idx"], "7")

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

    def test_can_fallback_when_learned_coordinates_do_not_match_contract(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            coordinate_mismatch_action="fallback_disabled",
        )

        result = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            row_starts=torch.tensor([10], dtype=torch.int32),
            topk_indices_offset=torch.tensor([100], dtype=torch.int32),
        )

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.coordinate_mismatch_fallbacks, 1)
        self.assertGreater(result.counters.schedule_ms, 0.0)

    def test_ragged_coordinates_use_topk_offset_not_row_start_plus_offset(self):
        topk = torch.tensor([[107, 106, 105, 104]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        result = apply_topology_kv_policy(
            topk,
            lengths,
            config=config,
            row_starts=torch.tensor([10], dtype=torch.int32),
            topk_indices_offset=torch.tensor([100], dtype=torch.int32),
        )

        values = [value for value in result.topk_indices[0].tolist() if value >= 0]
        self.assertTrue(all(100 <= value < 108 for value in values))

    def test_bypasses_python_policy_when_prefill_length_exceeds_limit(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
            max_policy_tokens=4,
        )

        result = apply_topology_kv_policy(topk, lengths, config=config)

        self.assertIs(result.topk_indices, topk)
        self.assertEqual(result.counters.policy_bypassed, 1)
        self.assertEqual(result.bypass_reasons, ("sequence_length_limit",))
        self.assertEqual(result.max_sequence_length, 8)

    def test_rejects_row_start_shape_mismatch(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "row_starts must have shape"):
            apply_topology_kv_policy(
                topk,
                lengths,
                config=config,
                row_starts=torch.tensor([0, 1], dtype=torch.int32),
            )

    def test_rejects_topk_offset_shape_mismatch(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        config = TopologyKvPolicyConfig(
            policy="topology_sparse_merge",
            sink_blocks=1,
            local_blocks=1,
            witness_blocks=0,
            block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "topk_indices_offset must have shape"):
            apply_topology_kv_policy(
                topk,
                lengths,
                config=config,
                topk_indices_offset=torch.tensor([0, 1], dtype=torch.int32),
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

    def test_indexer_compute_topk_ragged_requires_offset_absolute_coordinates(self):
        topk = torch.tensor([[107, 106, 105, 104]], dtype=torch.int32)
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
        self.assertTrue(all(100 <= value < 108 for value in values))

    def test_indexer_compute_topk_cp_requires_offset_absolute_coordinates(self):
        topk = torch.tensor([[107, 106, 105, 104]], dtype=torch.int32)
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
        self.assertTrue(all(100 <= value < 108 for value in values))

    def test_indexer_falls_back_when_real_op_coordinates_do_not_match_policy(self):
        topk = torch.tensor([[117, 116, 115, 114]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=torch.tensor([8]),
            ks=torch.tensor([10]),
            topk_indices_offset=torch.tensor([100]),
        )
        attention_inputs = SimpleNamespace(is_prefill=True)

        with mock.patch.object(indexer_module.logging, "warning"):
            result = indexer._compute_topk(
                None, None, None, fmha_params, attention_inputs, None
            )

        self.assertIs(result, topk)

    def test_indexer_does_not_reuse_fingerprint_across_policy_calls(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        indexer.topology_kv_policy = "topology_compress_sparse"
        indexer.topology_stable_scaffold = "stable"
        indexer.topology_output_contract = "contract"
        captured_previous = []
        policy_result = topology_kv_policy.TopologyKvPolicyResult(
            topk,
            topology_kv_policy.TopologyKvCounters(
                raw_selected_tokens=4,
                compressed_tokens_represented=0,
                unselected_stable_tokens=0,
                learned_kept_tokens=0,
                learned_evicted_tokens=0,
                compression_hits=0,
                coordinate_mismatch_fallbacks=0,
                policy_bypassed=0,
                schedule_ms=0.0,
            ),
            "fingerprint",
        )

        def _fake_apply(*args, **kwargs):
            captured_previous.append(kwargs.get("previous_fingerprint"))
            return policy_result

        with mock.patch.object(indexer_module, "apply_topology_kv_policy", _fake_apply):
            first = indexer._apply_topology_kv_policy(topk, torch.tensor([8]))
            second = indexer._apply_topology_kv_policy(topk, torch.tensor([8]))

        self.assertIs(first, topk)
        self.assertIs(second, topk)
        self.assertEqual(captured_previous, [None, None])

    def test_indexer_warns_once_when_cuda_policy_sync_is_not_enabled(self):
        class _CudaTopk:
            is_cuda = True

        indexer_module._TOPOLOGY_KV_CUDA_SYNC_WARNING_EMITTED = False
        topk = _CudaTopk()
        indexer = _make_indexer(topk)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(indexer_module.logging, "warning") as warning:
                with mock.patch.object(indexer_module.kmonitor, "report") as report:
                    first = indexer._apply_topology_kv_policy(topk, torch.tensor([8]))
                    second = indexer._apply_topology_kv_policy(topk, torch.tensor([8]))

        self.assertIs(first, topk)
        self.assertIs(second, topk)
        warning.assert_called_once()
        bypass_calls = [
            call
            for call in report.call_args_list
            if call.args[0]
            is indexer_module.AccMetrics.TOPOLOGY_KV_POLICY_BYPASS_LAYER_EVENT_METRIC
        ]
        self.assertEqual(len(bypass_calls), 2)
        self.assertTrue(
            all(
                call.args[2]["reason"] == "cuda_sync_disabled"
                and call.args[2]["layer_idx"] == "7"
                for call in bypass_calls
            )
        )

    def test_indexer_warns_once_when_long_prefix_exceeds_policy_limit(self):
        topk = torch.arange(2048, dtype=torch.int32).view(1, -1)
        indexer = _make_indexer(topk)
        indexer_module._TOPOLOGY_KV_BYPASS_WARNED_REASONS = set()

        with mock.patch.object(indexer_module.logging, "warning") as warning:
            first = indexer._apply_topology_kv_policy(
                topk, torch.tensor([8193], dtype=torch.int32)
            )
            second = indexer._apply_topology_kv_policy(
                topk, torch.tensor([8193], dtype=torch.int32)
            )

        self.assertIs(first, topk)
        self.assertIs(second, topk)
        warning.assert_called_once()
        self.assertIn("sequence_length_limit", warning.call_args.args[0])
        self.assertIn("RTP_LLM_TOPOLOGY_MAX_POLICY_TOKENS", warning.call_args.args[0])

    def test_indexer_reports_coordinate_fallback_from_production_compute_path(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        indexer = _make_indexer(topk)
        indexer_module._TOPOLOGY_KV_FALLBACK_WARNED_REASONS = set()
        attention_inputs = SimpleNamespace(is_prefill=True)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=torch.tensor([8], dtype=torch.int32),
            ks=torch.tensor([10], dtype=torch.int32),
            topk_indices_offset=torch.tensor([100], dtype=torch.int32),
        )

        with mock.patch.object(indexer_module.logging, "warning") as warning:
            with mock.patch.object(indexer_module.kmonitor, "report") as report:
                first = indexer._compute_topk(
                    None, None, None, fmha_params, attention_inputs, None
                )
                second = indexer._compute_topk(
                    None, None, None, fmha_params, attention_inputs, None
                )

        self.assertIs(first, topk)
        self.assertIs(second, topk)
        warning.assert_called_once()
        self.assertIn("coordinate_mismatch", warning.call_args.args[0])
        fallback_calls = [
            call
            for call in report.call_args_list
            if call.args[0]
            is indexer_module.AccMetrics.TOPOLOGY_KV_COORDINATE_FALLBACK_LAYER_EVENT_METRIC
        ]
        self.assertEqual(len(fallback_calls), 2)
        self.assertTrue(all(call.args[1] == 1 for call in fallback_calls))
        self.assertTrue(
            all(
                call.args[2]["reason"] == "coordinate_mismatch"
                and call.args[2]["layer_idx"] == "7"
                for call in fallback_calls
            )
        )

    def test_indexer_reports_schedule_selection_and_eviction_counters(self):
        topk = torch.tensor([[7, 6, 5, 4]], dtype=torch.int32)
        indexer = _make_indexer(topk)

        with mock.patch.object(indexer_module.kmonitor, "report") as report:
            result = indexer._apply_topology_kv_policy(
                topk, torch.tensor([8], dtype=torch.int32)
            )

        self.assertIsNot(result, topk)
        reported = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertIn(
            indexer_module.GaugeMetrics.TOPOLOGY_KV_POLICY_SCHEDULE_MS_METRIC,
            reported,
        )
        self.assertGreaterEqual(
            reported[indexer_module.GaugeMetrics.TOPOLOGY_KV_POLICY_SCHEDULE_MS_METRIC],
            0.0,
        )
        self.assertIn(
            indexer_module.GaugeMetrics.TOPOLOGY_KV_POLICY_SELECTED_TOKENS_METRIC,
            reported,
        )
        self.assertIn(
            indexer_module.GaugeMetrics.TOPOLOGY_KV_POLICY_EVICTED_TOKENS_METRIC,
            reported,
        )
        self.assertTrue(
            all(call.args[2]["layer_idx"] == "7" for call in report.call_args_list)
        )

    def test_indexer_topology_env_int_reports_bad_value(self):
        with mock.patch.dict("os.environ", {"RTP_LLM_TOPOLOGY_SINK_BLOCKS": "true"}):
            with self.assertRaisesRegex(
                ValueError, "RTP_LLM_TOPOLOGY_SINK_BLOCKS must be an integer"
            ):
                _topology_env_int("RTP_LLM_TOPOLOGY_SINK_BLOCKS", 1)

    def test_indexer_topology_env_float_reports_bad_value(self):
        with mock.patch.dict(
            "os.environ", {"RTP_LLM_TOPOLOGY_MAX_STRUCTURAL_FRACTION": "many"}
        ):
            with self.assertRaisesRegex(
                ValueError,
                "RTP_LLM_TOPOLOGY_MAX_STRUCTURAL_FRACTION must be a float",
            ):
                _topology_env_float("RTP_LLM_TOPOLOGY_MAX_STRUCTURAL_FRACTION", 0.5)


if __name__ == "__main__":
    unittest.main()
