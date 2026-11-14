"""Core performance behavior contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import (
    canonical_json,
    load_resolved_config,
    resolve_config,
)
from rtp_llm.test.perf_test.config_schema import (
    ConfigError,
    DecodeGridWorkloadConfig,
    LocalExecutionConfig,
    PrefillWorkloadConfig,
    parse_config,
)
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    CollectionError,
    ConfigurationError,
    EngineLifecycleError,
    PartialRunError,
    PlanningError,
    PostExecutionError,
    ResourceError,
)
from rtp_llm.test.perf_test.test_support import (
    config_resolution_base_config,
    fixture_engine,
    fixture_model,
    fixture_workload,
)


def unified_config_base_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            tokenizer="/models/tokenizer",
            topology={},
        ),
        "engine": fixture_engine(args={}, env={}),
        "workload": fixture_workload(),
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


class ConfigResolutionTest(unittest.TestCase):

    def test_derived_ep_and_explicit_ep_have_the_same_fingerprint(self):
        source = unified_config_base_config()
        source["model"]["topology"] = {"tp": 2, "dp": 2, "pp": 2}
        derived = resolve_config(source)
        source["model"]["topology"]["ep"] = 4
        explicit = resolve_config(source)
        self.assertEqual(derived.resolved_config, explicit.resolved_config)
        self.assertEqual(derived.config_fingerprint, explicit.config_fingerprint)

    def test_equivalent_defaults_and_mapping_order_hash_identically(self):
        minimal = config_resolution_base_config()
        left = resolve_config(minimal)
        explicit = copy.deepcopy(minimal)
        explicit["model"]["topology"] = dict(left.resolved_config["model"]["topology"])
        explicit["workload"]["measure_runs"] = left.resolved_config["workload"][
            "measure_runs"
        ]
        explicit = dict(reversed(list(explicit.items())))
        explicit["engine"]["args"] = dict(
            reversed(list(explicit["engine"]["args"].items()))
        )
        right = resolve_config(explicit)
        self.assertEqual(left.config_fingerprint, right.config_fingerprint)
        self.assertEqual(left.workload_fingerprint, right.workload_fingerprint)
        self.assertEqual(
            canonical_json(left.resolved_config), canonical_json(right.resolved_config)
        )

    def test_engine_and_workload_changes_affect_separate_fingerprints(self):
        original = resolve_config(config_resolution_base_config())
        engine_change = config_resolution_base_config()
        engine_change["engine"]["args"]["reuse_cache"] = 0
        workload_change = config_resolution_base_config()
        workload_change["workload"]["max_input_len"] = 4096
        changed_engine = resolve_config(engine_change)
        changed_workload = resolve_config(workload_change)
        self.assertNotEqual(
            original.config_fingerprint, changed_engine.config_fingerprint
        )
        self.assertEqual(
            original.workload_fingerprint, changed_engine.workload_fingerprint
        )
        self.assertEqual(
            original.config_fingerprint, changed_workload.config_fingerprint
        )
        self.assertNotEqual(
            original.workload_fingerprint, changed_workload.workload_fingerprint
        )

    def test_distribution_file_is_materialized_relative_to_source_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workloads = root / "workloads"
            workloads.mkdir()
            distribution = {"sequence_lengths": [128, 256, 512]}
            (workloads / "requests.json").write_text(
                json.dumps(distribution), encoding="utf-8"
            )
            payload = config_resolution_base_config()
            payload["workload"] = {
                "kind": "distribution",
                "distribution_file": "workloads/requests.json",
                "output_length": 16,
                "batch": {"strategy": "fixed", "sizes": [8]},
            }
            config_path = root / "perf.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            resolved = load_resolved_config(config_path)
            self.assertNotIn("distribution_file", resolved.resolved_config["workload"])
            self.assertEqual(
                distribution, resolved.resolved_config["workload"]["distribution"]
            )
            self.assertNotIn(str(root), canonical_json(resolved.resolved_config))


class UnifiedConfigTest(unittest.TestCase):

    def test_topology_derives_ep_within_each_pipeline_stage(self):
        for tp, dp, pp in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 2)):
            with self.subTest(tp=tp, dp=dp, pp=pp):
                source = unified_config_base_config()
                source["model"]["topology"] = dict(tp=tp, dp=dp, pp=pp)
                self.assertEqual(
                    (tp, dp, pp, tp * dp),
                    parse_config(source).model.topology.as_tuple(),
                )

    def test_topology_preserves_explicit_pure_tp(self):
        source = unified_config_base_config()
        source["model"]["topology"] = dict(tp=4, dp=1, pp=2, ep=1)
        self.assertEqual((4, 1, 2, 1), parse_config(source).model.topology.as_tuple())

    def test_topology_rejects_ep_values_the_engine_cannot_start(self):
        for tp, dp, ep in ((1, 2, 1), (2, 2, 1), (2, 2, 2), (2, 1, 4)):
            with self.subTest(tp=tp, dp=dp, ep=ep):
                source = unified_config_base_config()
                source["model"]["topology"] = dict(tp=tp, dp=dp, pp=1, ep=ep)
                with self.assertRaisesRegex(ConfigError, "model.topology.ep"):
                    parse_config(source)

    def test_rank_batch_unit_is_reserved_for_fixed_grid(self):
        payload = unified_config_base_config()
        payload["workload"] = {
            "kind": "distribution",
            "distribution_file": "requests.json",
            "output_length": 16,
            "batch": {"strategy": "fixed", "sizes": [2], "unit": "per_dp_rank"},
        }
        with self.assertRaisesRegex(ConfigError, "workload.batch.unit"):
            parse_config(payload)

    def test_rank_batch_unit_rejects_pipeline_parallelism(self):
        payload = unified_config_base_config()
        payload["model"]["topology"]["pp"] = 2
        payload["workload"] = {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1], "unit": "per_dp_rank"},
        }
        with self.assertRaisesRegex(ConfigError, "PP=1"):
            parse_config(payload)

    def test_prefill_config_applies_nested_defaults(self):
        config = parse_config(unified_config_base_config())
        self.assertEqual((1, 1, 1, 1), config.model.topology.as_tuple())
        self.assertIsInstance(config.workload, PrefillWorkloadConfig)
        self.assertEqual("dashsc_input_ids", config.workload.transport)
        self.assertIsInstance(config.execution, LocalExecutionConfig)

    def test_decode_grid_preserves_declared_extension_maps(self):
        payload = unified_config_base_config()
        payload["engine"]["args"] = {"reuse_cache": 1, "act_type": "BF16"}
        payload["engine"]["env"] = {"TOKENIZERS_PARALLELISM": "false"}
        payload["workload"] = {
            "kind": "grid",
            "input_lengths": [128, 256],
            "output_lengths": [32],
            "batch": {"strategy": "fixed", "sizes": [1, 4]},
            "generate_config": {"top_k": 1, "stop_words_list": [[1, 2]]},
        }
        config = parse_config(payload)
        self.assertIsInstance(config.workload, DecodeGridWorkloadConfig)
        self.assertEqual((128, 256), config.workload.input_lengths)
        self.assertEqual((32,), config.workload.output_lengths)
        self.assertEqual({"reuse_cache": 1, "act_type": "BF16"}, config.engine.args)
        self.assertEqual({"TOKENIZERS_PARALLELISM": "false"}, config.engine.env)
        self.assertEqual(
            {"top_k": 1, "stop_words_list": [[1, 2]]}, config.workload.generate_config
        )

    def test_unknown_structured_fields_report_complete_path(self):
        payload = unified_config_base_config()
        payload["model"]["topology"]["cp"] = 2
        with self.assertRaisesRegex(ConfigError, "model\\.topology\\.cp"):
            parse_config(payload)

    def test_decode_batch_strategies_are_mutually_exclusive(self):
        for batch, path in (
            (
                {
                    "strategy": "target_tpot",
                    "sizes": [4],
                    "max_size": 16,
                    "candidate_step": 4,
                    "target_tpot_ms": 10.0,
                },
                "workload.batch.sizes",
            ),
            (
                {"strategy": "fixed", "sizes": [4], "target_tpot_ms": 10.0},
                "workload.batch.target_tpot_ms",
            ),
        ):
            with self.subTest(path=path):
                payload = unified_config_base_config()
                payload["workload"] = {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": batch,
                }
                with self.assertRaisesRegex(ConfigError, path):
                    parse_config(payload)


class CapabilitiesTest(unittest.TestCase):

    def test_failure_types_have_stable_exit_code_classes(self):
        expected = {
            ConfigurationError: 2,
            PlanningError: 3,
            ResourceError: 3,
            EngineLifecycleError: 4,
            CaseExecutionError: 5,
            PartialRunError: 5,
            PostExecutionError: 6,
            CollectionError: 6,
        }
        for error_type, exit_code in expected.items():
            with self.subTest(error_type=error_type):
                error = error_type("failure")
                self.assertEqual(exit_code, error.exit_code)
                self.assertEqual("failure", str(error))


if __name__ == "__main__":
    unittest.main()
