"Performance framework behavior and failure regressions."

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
from rtp_llm.test.perf_test.tests.support.fixtures import (
    config_resolution_base_config,
    fixture_config,
    fixture_engine,
    fixture_model,
)


def unified_config_base_config():
    return fixture_config(
        model=fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            tokenizer="/models/tokenizer",
            topology={},
        ),
        engine=fixture_engine(args={}, env={}),
        execution={"backend": "local"},
    )


class ConfigResolutionTest(unittest.TestCase):

    def test_warmup_settings_are_resolved_and_validated(self):
        source = unified_config_base_config()
        defaults = resolve_config(source).resolved_config["workload"]
        self.assertEqual(0.25, defaults["warmup_deviation_threshold"])
        self.assertEqual(1200.0, defaults["warmup_sweep_budget_s"])
        self.assertEqual(3, defaults["warmup_sweep_profiles"])
        custom = dict(
            warmup_deviation_threshold=0.5,
            warmup_sweep_budget_s=60,
            warmup_sweep_profiles=2,
        )
        source["workload"].update(custom)
        resolved = resolve_config(source).resolved_config["workload"]
        for key, expected in custom.items():
            self.assertEqual(expected, resolved[key])
        for key, invalid in (
            ("warmup_deviation_threshold", 0),
            ("warmup_deviation_threshold", 1.1),
            ("warmup_deviation_threshold", True),
            ("warmup_sweep_budget_s", 0),
            ("warmup_sweep_budget_s", float("inf")),
            ("warmup_sweep_profiles", 0),
            ("warmup_sweep_profiles", 1.5),
        ):
            with self.subTest(key=key, invalid=invalid):
                bad = copy.deepcopy(source)
                bad["workload"][key] = invalid
                with self.assertRaises(ConfigError):
                    resolve_config(bad)

    def test_derived_ep_and_explicit_ep_have_the_same_fingerprint(self):
        source = unified_config_base_config()
        source["model"]["topology"] = {"tp": 2, "dp": 2, "pp": 1}
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
        explicit["workload"]["request_timeout_seconds"] = 7200
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

    def test_topology_resolves_automatic_and_explicit_ep(self):
        for topology, expected in (
            ({"tp": 2, "dp": 3}, (2, 3, 1, 6)),
            ({"tp": 4, "dp": 1, "ep": 1}, (4, 1, 1, 1)),
        ):
            with self.subTest(topology=topology):
                source = unified_config_base_config()
                source["model"]["topology"] = topology
                self.assertEqual(
                    expected, parse_config(source).model.topology.as_tuple()
                )

    def test_topology_rejects_ep_values_the_engine_cannot_start(self):
        for tp, dp, ep in ((1, 2, 1), (2, 2, 2), (2, 1, 4)):
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

    def test_pipeline_parallelism_is_rejected(self):
        payload = unified_config_base_config()
        payload["model"]["topology"]["pp"] = 2
        with self.assertRaisesRegex(ConfigError, "PP=1"):
            parse_config(payload)

    def test_prefill_config_applies_nested_defaults(self):
        config = parse_config(unified_config_base_config())
        self.assertEqual((1, 1, 1, 1), config.model.topology.as_tuple())
        self.assertIsInstance(config.workload, PrefillWorkloadConfig)
        self.assertEqual("dashsc_input_ids", config.workload.transport)
        self.assertIsInstance(config.execution, LocalExecutionConfig)
        self.assertEqual(7200, config.workload.request_timeout_seconds)
        source = unified_config_base_config()
        source["workload"]["request_timeout_seconds"] = 12.5
        resolved = resolve_config(source)
        self.assertEqual(
            12.5, resolved.resolved_config["workload"]["request_timeout_seconds"]
        )
        self.assertNotEqual(
            resolve_config(unified_config_base_config()).workload_fingerprint,
            resolved.workload_fingerprint,
        )
        for value in (0, -1, True, "7200", None, float("inf"), float("nan")):
            with self.subTest(value=value):
                source["workload"]["request_timeout_seconds"] = value
                with self.assertRaisesRegex(
                    ConfigError, "workload.request_timeout_seconds"
                ):
                    parse_config(source)

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

        payload["workload"]["generate_config"].update(
            regex="^[a-z]+$", json_schema={"$ref": "#/$defs/answer"}
        )
        payload["engine"]["args"]["grammar_terminate_without_stop_token"] = 1
        self.assertEqual(
            "^[a-z]+$", parse_config(payload).workload.generate_config["regex"]
        )

        payload["publish"] = {
            "oss": {
                "enabled": True,
                "required": True,
                "destination": "oss://bucket/results",
            }
        }
        with self.assertRaisesRegex(ConfigError, "publication is not supported"):
            parse_config(payload)

    def test_unknown_structured_fields_report_complete_path(self):
        payload = unified_config_base_config()
        payload["model"]["topology"]["cp"] = 2
        with self.assertRaisesRegex(ConfigError, "model\\.topology\\.cp"):
            parse_config(payload)

    def test_decode_batch_rejects_search_strategy_and_fields(self):
        for batch, path in (
            (
                {
                    "strategy": "target_tpot",
                    "max_size": 16,
                    "candidate_step": 4,
                    "target_tpot_ms": 10.0,
                },
                "workload.batch.strategy",
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


class SshConfigurationTest(unittest.TestCase):

    def execution(self):
        return {
            "backend": "ssh",
            "hosts": [{"ssh_target": "gpu-alias", "advertise_host": "192.0.2.11"}],
            "gpus_per_machine": 1,
            "source": {"build_policy": "reuse"},
        }

    def test_single_config_accepts_ssh_alias_and_independent_advertised_address(self):
        source = unified_config_base_config()
        source["execution"] = self.execution()
        execution = parse_config(source).execution
        self.assertEqual("gpu-alias", execution.hosts[0].ssh_target)
        self.assertEqual("192.0.2.11", execution.hosts[0].advertise_host)
        self.assertIsNone(execution.run_user)
        self.assertIsNone(execution.ssh_port)

    def test_build_requires_explicit_driver_and_rejects_credential_fields(self):
        from rtp_llm.test.perf_test.ssh.config_schema import parse_execution

        execution = self.execution()
        execution["source"]["build_policy"] = "build"
        with self.assertRaisesRegex(ConfigError, "build_driver"):
            parse_execution(execution)
        execution["source"]["build_policy"] = "reuse"
        execution["password"] = "secret"
        with self.assertRaises(ConfigError):
            parse_execution(execution)

        from rtp_llm.test.perf_test.config_schema import _reject_unsupported_forms

        for value in (
            {"engine": {"args": {"api-key": "fake"}}},
            {"engine": {"env": {"AUTHORIZATION": "fake"}}},
            {"model": {"checkpoint": "https://user:fake@example.com/model"}},
            {"model": {"checkpoint": "https://example.com/model?signature=fake"}},
            {"engine": {"env": {"EXTRA": "Bearer fake"}}},
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                _reject_unsupported_forms(value)

    def test_unknown_host_options_and_duplicate_members_are_rejected(self):
        from rtp_llm.test.perf_test.ssh.config_schema import parse_execution

        execution = self.execution()
        execution["hosts"] *= 2
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            parse_execution(execution)
        execution = self.execution()
        execution["hosts"][0]["ssh_options"] = "-F none"
        with self.assertRaises(ConfigError):
            parse_execution(execution)
        execution = self.execution()
        execution["hosts"][0]["advertise_host"] = "[2001:db8::1]"
        with self.assertRaisesRegex(ConfigError, "advertise_host"):
            parse_execution(execution)


class ExternalInputSnapshotTest(unittest.TestCase):

    def test_experiment_input_snapshot_survives_file_mutation_and_is_not_aliased(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = [[128, 1]]
            path = root / "distribution.json"
            path.write_text(json.dumps(payload))
            source = unified_config_base_config()
            source["workload"] = {
                "kind": "distribution",
                "distribution_file": path.name,
                "output_length": 8,
                "batch": {"strategy": "fixed", "sizes": [1]},
            }
            cache = {}
            first = resolve_config(source, source_dir=root, external_inputs=cache)
            path.write_text("corrupt now")
            second = resolve_config(source, source_dir=root, external_inputs=cache)
            self.assertEqual(first.workload_fingerprint, second.workload_fingerprint)
            first.resolved_config["workload"]["distribution"].append([256, 1])
            self.assertEqual(
                payload, second.resolved_config["workload"]["distribution"]
            )


if __name__ == "__main__":
    unittest.main()
