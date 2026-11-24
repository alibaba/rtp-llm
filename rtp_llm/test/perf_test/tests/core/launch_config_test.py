"""Shared input selection and typed launch overrides."""

import copy
import unittest

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.launch_config import prepare_config
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config


def shared_config():
    source = fixture_config()
    source["schema_version"] = 2
    source["execution"] = {
        "backend": "aone",
        "platform": "cuda13",
        "test_timeout_seconds": 19500,
        "local": {"devices": [0]},
        "aone": {
            "resource": "daily",
            "gpu_type": "H20_CU13",
            "shard_count": 1,
            "max_parallel": 1,
            "exec_properties": {"custom.pool": "benchmark"},
        },
    }
    return source


class LaunchConfigTest(unittest.TestCase):
    def test_one_document_selects_local_without_aone_fields_or_mutation(self):
        source = shared_config()
        original = copy.deepcopy(source)
        local = prepare_config(source, ("execution.backend=local",))
        self.assertEqual(original, source)
        self.assertEqual("local", local["execution"]["backend"])
        self.assertNotIn("resource", local["execution"])
        self.assertNotIn("aone", local["execution"])
        resolved = resolve_config(local).resolved_config
        self.assertEqual(19500, resolved["execution"]["test_timeout_seconds"])
        self.assertEqual("cuda13", resolved["execution"]["platform"])

    def test_old_aone_document_can_switch_to_local(self):
        source = fixture_config(
            execution={
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20_CU13",
                "shard_count": 1,
                "max_parallel": 1,
                "test_timeout_seconds": 19500,
            }
        )
        result = prepare_config(
            source, ("execution.backend=local", "execution.local.devices=[0]")
        )
        self.assertEqual([0], result["execution"]["devices"])
        self.assertEqual("cuda13", result["execution"]["platform"])
        self.assertNotIn("gpu_type", result["execution"])

    def test_typed_overrides_and_null_remove_optional_shards(self):
        result = prepare_config(
            shared_config(),
            (
                "execution.aone.max_parallel=2",
                "execution.aone.gpus_per_machine=1",
                "execution.aone.shard_count=null",
                "engine.args.reuse_cache=true",
            ),
        )
        self.assertEqual(2, result["execution"]["max_parallel"])
        self.assertNotIn("shard_count", result["execution"])
        self.assertIs(True, result["engine"]["args"]["reuse_cache"])

    def test_unknown_inactive_fields_and_malformed_overrides_are_errors(self):
        for override in (
            "execution.aone.max_paralel=2",
            "execution.backend",
            "=local",
            "execution.local.devices.0=1",
        ):
            with self.subTest(override=override), self.assertRaises(ConfigurationError):
                prepare_config(shared_config(), (override, "execution.backend=local"))

    def test_cli_overrides_require_new_config_run(self):
        request = parse_cli(["--config=a.json", "--set=execution.backend=local"])
        self.assertEqual(("execution.backend=local",), request.overrides)
        with self.assertRaises(ConfigurationError):
            parse_cli(
                ["--run-dir=run", "--action=resume", "--set=execution.backend=local"]
            )

    def test_legacy_ssh_input_is_untouched(self):
        source = fixture_config(
            execution={"backend": "ssh", "hosts": [{"ssh_target": "x"}]}
        )
        self.assertEqual(source, prepare_config(source))
        with self.assertRaisesRegex(ConfigurationError, "execution"):
            prepare_config(source, ("execution=[]",))

    def test_cli_switches_backend_without_modifying_shared_input(self):
        import io
        import json
        import tempfile
        from pathlib import Path

        from rtp_llm.test.perf_test.entry import main

        with tempfile.TemporaryDirectory() as directory:
            source = shared_config()
            source["output"] = {"result_root": str(Path(directory) / "results")}
            path = Path(directory) / "input.json"
            original = json.dumps(source)
            path.write_text(original)
            results = []
            for arguments in ([], ["--set=execution.backend=local"]):
                output = io.StringIO()
                self.assertEqual(
                    0,
                    main(
                        ["--config=" + str(path), "--dry-run", *arguments],
                        stdout=output,
                    ),
                )
                results.append(json.loads(output.getvalue()))
            self.assertEqual(["aone", "local"], [item["backend"] for item in results])
            self.assertEqual(
                results[0]["workload_fingerprint"], results[1]["workload_fingerprint"]
            )
            self.assertEqual(results[0]["cases"], results[1]["cases"])
            self.assertEqual(original, path.read_text())
            self.assertFalse((Path(directory) / "results").exists())

    def test_new_format_requires_local_platform_and_rejects_unknown_version(self):
        source = shared_config()
        source["execution"].pop("platform")
        with self.assertRaisesRegex(ConfigurationError, "platform"):
            prepare_config(source, ("execution.backend=local",))
        source["schema_version"] = 9
        with self.assertRaisesRegex(ConfigurationError, "schema_version"):
            prepare_config(source)

    def test_resolver_and_entry_accept_shared_document_directly(self):
        from pathlib import Path

        from rtp_llm.test.perf_test.entry import plan_from_source

        source = shared_config()
        plan, _ = plan_from_source(source, source_dir=Path.cwd())
        self.assertEqual("aone", plan.backend)
        self.assertEqual(
            "benchmark",
            plan.resolved_config["execution"]["exec_properties"]["custom.pool"],
        )

    def test_aone_platform_must_match_gpu_architecture(self):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution

        active = prepare_config(shared_config())
        active["execution"]["platform"] = "cuda13_arm"
        with self.assertRaisesRegex(ConfigurationError, "platform"):
            resolve_config(active, execution_parser=parse_execution)

    def test_explicit_invalid_aone_platform_is_not_treated_as_missing(self):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution

        for value in (False, 0, [], {}, ""):
            with self.subTest(value=value):
                active = prepare_config(shared_config())
                active["execution"]["platform"] = value
                with self.assertRaisesRegex(ConfigurationError, "platform"):
                    resolve_config(active, execution_parser=parse_execution)

    def test_shared_example_plans_all_three_execution_routes(self):
        import json
        from pathlib import Path

        from rtp_llm.test.perf_test.entry import plan_from_source
        from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT

        path = PERF_TEST_ROOT / "examples/prefill_shared.json"
        source = json.loads(path.read_text())
        for overrides, machines in (
            ((), 1),
            (("execution.backend=aone",), 1),
            (
                (
                    "execution.backend=aone",
                    "execution.aone.gpus_per_machine=1",
                    "execution.aone.shard_count=null",
                ),
                2,
            ),
        ):
            with self.subTest(overrides=overrides):
                plan, _ = plan_from_source(
                    prepare_config(source, overrides), source_dir=path.parent
                )
                self.assertEqual(
                    machines, getattr(plan.resource_plan, "machine_count", 1)
                )

    def test_static_prefill_macro_uses_worker_entry(self):
        from unittest.mock import Mock

        from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT

        native = Mock()
        namespace = {"native": native}
        exec((PERF_TEST_ROOT / "aone_defs.bzl").read_text(), namespace)
        namespace["unified_prefill_ci_test"]("case", "config.json", "H20_CU13", 1)
        self.assertEqual(
            "//rtp_llm/test/perf_test:local_main.py",
            native.py_test.call_args.kwargs["main"],
        )

    def test_aone_properties_preserve_full_host_gang_reservation(self):
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.aone.package import _gang_build_file
        from rtp_llm.test.perf_test.aone.plan import aone_exec_properties

        resource = SimpleNamespace(
            gpu_type="H20_CU13", machine_count=2, gpus_per_shard=4
        )
        properties = aone_exec_properties(resource, {"custom.pool": "benchmark"})
        self.assertEqual("8", properties["gpu_count"])
        self.assertEqual("benchmark", properties["custom.pool"])
        with self.assertRaisesRegex(ConfigurationError, "gpu_count"):
            aone_exec_properties(resource, {"gpu_count": "2"})
        build = _gang_build_file(
            resource, test_timeout_seconds=19500, exec_properties=properties
        )
        self.assertIn('"custom.pool": "benchmark"', build)
        self.assertIn("gpus_per_host = 8", build)


if __name__ == "__main__":
    unittest.main()
