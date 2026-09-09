"""Transition ownership and actual launch-probe contract; fixtures do not run Java."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from environment_expectations import configuration
from flexlb_test_framework.harness import render_env
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.actions.environment import (
    _validate,
    mutate_config,
    reconfigure,
    startup_probe,
)
from flexlb_test_framework.scenario.backend import JavaMockBackend
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import environment
from flexlb_test_framework.scenario.contracts import PlanContext
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    _core_action,
)
from test_scenario_backend import lease_manifest
from test_scenario_runtime import source

PROFILE = "single-nonbatch"


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail_cleanup = False
        self.fail_setup = False

    def setup(self, ctx, plan, deadline):
        self.calls.append(("setup", ctx.env_epoch, plan["effective_axes"]["ordering"]))
        if self.fail_setup:
            raise RuntimeError("partial setup")
        return object(), object()

    def teardown(self, ctx, deadline):
        self.calls.append(("cleanup", ctx.env_epoch))
        if self.fail_cleanup:
            raise RuntimeError("owned consumer still running")

    def probe_startup(self, ctx, plan, raw, deadline):
        self.calls.append(("probe", ctx.env_epoch, json.loads(raw)))
        return copy.deepcopy(self.observation)


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_env = {
            "n_prefill": 1,
            "n_decode": 4,
            "config_overrides": {"ordering": "priority"},
        }
        self.plan = PlanContext("test", {}, self.raw_env, (PROFILE,))
        self.backend = FakeBackend()
        self.ctx = RuntimeContext(
            {
                "profile": PROFILE,
                "environment": environment(self.raw_env, "test", PROFILE),
            },
            self.backend,
            self.tmp.name,
            time.monotonic,
            time.sleep,
        )
        self.deadline = Deadline(time.monotonic() + 10)

    def test_compiled_replacement_changes_ordering_not_topology(self):
        doc = source()
        doc["profiles"] = [PROFILE]
        doc["environment"] = self.raw_env
        doc["stages"].insert(
            1,
            {
                "id": "fifo",
                "action": "environment_reconfigure",
                "params": {"config_overrides": {"ordering": "fifo"}},
            },
        )
        instance = compile_scenarios([("test", doc)], handlers=handlers())[0]
        plan = instance["stages"][1]["params"]["environments"][PROFILE]
        self.assertEqual(plan["effective_axes"]["ordering"], "FIFO")
        self.assertEqual((plan["n_prefill"], plan["n_decode"]), (1, 4))
        with self.assertRaises(ValueError):
            _validate({"config_overrides": {"dispatcher": "batch"}}, self.plan)
        with self.assertRaises(ValueError):
            _validate(
                {"config_overrides": {}, "master_layout": "dual_standalone"}, self.plan
            )

    def test_consumers_clean_before_replacement_old_handles_stale(self):
        initial = _core_action("setup", self.ctx, {}, self.deadline).output[
            "environment"
        ]
        self.ctx.add_cleanup(
            "consumer",
            lambda d: self.backend.calls.append(("consumer", self.ctx.env_epoch)),
        )
        historical = self.ctx.register_resource(
            "snapshot", {"first": True}, historical=True
        )
        self.ctx.outputs["old"] = {"passed": True}
        reconfigure(
            self.ctx,
            _validate({"config_overrides": {"ordering": "fifo"}}, self.plan),
            self.deadline,
        )
        self.assertEqual(
            self.backend.calls,
            [
                ("setup", 1, "PRIORITY"),
                ("consumer", 1),
                ("cleanup", 1),
                ("setup", 2, "FIFO"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            self.ctx.resource(initial, "environment")
        self.assertEqual(
            self.ctx.resource(historical, "snapshot", allow_stale=True), {"first": True}
        )
        self.assertTrue(self.ctx.resolve({"$ref": "stages.old.output.passed"}))

    def test_cleanup_failure_prevents_new_setup_and_remains_for_final_retry(self):
        _core_action("setup", self.ctx, {}, self.deadline)
        self.backend.fail_cleanup = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            reconfigure(
                self.ctx, _validate({"config_overrides": {}}, self.plan), self.deadline
            )
        self.assertEqual(self.ctx.env_epoch, 1)
        self.assertEqual(sum(row[0] == "setup" for row in self.backend.calls), 1)
        self.assertTrue(self.ctx._cleanup)
        self.backend.fail_cleanup = False
        self.assertEqual(self.ctx.cleanup(1)[0]["status"], "PASS")

    def test_partial_second_setup_remains_owned(self):
        _core_action("setup", self.ctx, {}, self.deadline)
        self.backend.fail_setup = True
        with self.assertRaisesRegex(RuntimeError, "partial setup"):
            reconfigure(
                self.ctx, _validate({"config_overrides": {}}, self.plan), self.deadline
            )
        self.assertEqual(self.ctx.env_epoch, 2)
        self.assertTrue(self.ctx._cleanup)
        self.ctx.cleanup(1)
        self.assertEqual(self.backend.calls[-1], ("cleanup", 2))

    def test_three_raw_mutations_match_expected_payloads(self):
        cases = [
            ("removed_auto_tpm", configuration("priority", PROFILE)),
            ("fifo_default_priority", configuration("fifo", PROFILE)),
            (
                "removed_engine_cancellation",
                configuration("decode_preemption", PROFILE),
            ),
        ]
        for name, overrides in cases:
            base = overrides
            expected = copy.deepcopy(base)
            if name == "removed_auto_tpm":
                expected["autoTpmEnabled"] = True
            elif name == "fifo_default_priority":
                expected["scheduler"]["ordering"]["defaultPriority"] = 50
            else:
                expected["scheduler"]["ordering"]["preemption"][
                    "engineCancellation"
                ] = {"ackTimeoutMs": 50, "completionTimeoutMs": 1000}
            self.assertEqual(mutate_config(base, name), expected)
        with self.assertRaises(ValueError):
            mutate_config(
                json.loads(render_env(PROFILE)), "removed_engine_cancellation"
            )

    def test_probe_does_not_turn_cleanup_or_generic_failure_into_parser_pass(self):
        params = _validate(
            {
                "config_overrides": {"ordering": "priority"},
                "mutation": "removed_auto_tpm",
            },
            self.plan,
            probe=True,
        )
        cases = [
            (False, "RuntimeError: startup", 1, "Unrecognized field", True, True),
            (True, None, None, "Unrecognized field", False, False),
            (False, "network failure", 1, "BindException", True, False),
            (False, "startup", None, "Unrecognized field", True, False),
        ]
        for started, error, code, log, absent, expected in cases:
            self.backend.observation = dict(
                started=started,
                startup_error=error,
                master_pids=[123],
                master_returncodes=[code],
                logs=[{"tail": log}],
                current_absent_before_cleanup=absent,
            )
            result = startup_probe(self.ctx, params, self.deadline)
            self.assertEqual(result.output["rejected"], expected)
            self.assertEqual(result.output["environment_absent"], absent)
            self.assertIsNone(self.ctx.env)
            self.assertEqual(self.backend.calls[-1][0], "cleanup")
        self.assertEqual(len(list(Path(self.tmp.name).glob("startup-probe-*.json"))), 4)

    def test_backend_epochs_keep_configuration_artifacts(self):
        backend = JavaMockBackend(lease_manifest())
        self.ctx.backend = backend
        self.ctx.instance["resource_budget"] = {
            "max_dynamic_additions": 0,
            "initial_workers": 5,
        }
        env = NS(
            base_grpc_port=55000,
            master_http_port=28000,
            master_management_port=28001,
            mock_http_port=54999,
        )
        with patch(
            "flexlb_test_framework.harness.EnvManager.ensure", return_value=env
        ) as ensure, patch("flexlb_test_framework.engine_ops.EngineOps"):
            self.ctx.env_epoch = 1
            first = environment(self.raw_env, "test", PROFILE)
            backend.setup(self.ctx, first, self.deadline)
            initial = (Path(self.tmp.name) / "environment.json").read_bytes()
            self.ctx.env_epoch = 2
            second = environment(
                {
                    "n_prefill": 1,
                    "n_decode": 4,
                    "config_overrides": {"ordering": "fifo"},
                },
                "test",
                PROFILE,
            )
            backend.setup(self.ctx, second, self.deadline)
            log_args = [
                next(
                    arg
                    for arg in call.args[0].master_extra_args
                    if arg.startswith("--flexlb.log.path=")
                )
                for call in ensure.call_args_list
            ]
            self.assertNotEqual(log_args[0], log_args[1])
            self.assertTrue(
                all(str(Path(self.tmp.name).resolve()) in arg for arg in log_args)
            )
            self.assertEqual(env.master_log_dir, self.ctx.master_log_dir)
        self.assertEqual(
            (Path(self.tmp.name) / "environment.json").read_bytes(), initial
        )
        later = json.loads(
            (Path(self.tmp.name) / "environment-epoch-2/environment.json").read_text()
        )
        self.assertEqual(later["resolved_config"], second["resolved_config"])
        self.assertEqual(later["master_log_dir"], str(self.ctx.master_log_dir))

    def test_backend_probe_requires_owned_master_and_reads_private_files(self):
        backend = JavaMockBackend(lease_manifest())
        directory = Path(self.tmp.name)
        (directory / "master-logs").mkdir()
        (directory / "master-logs/application.log").write_text(
            "ConfigValidationException"
        )
        stdout = directory / "stdout.log"
        stdout.write_text("owned stdout")
        mp = NS(pid=321, proc=NS(poll=lambda: 1), log_file=stdout)

        def launch(*args, **kwargs):
            backend.current_artifact_dir = directory
            backend.manager = NS(current=None)
            backend.environments.append(NS(master=mp))
            raise RuntimeError("master failed")

        with patch.object(backend, "_setup", side_effect=launch):
            result = backend.probe_startup(self.ctx, {}, "{}", self.deadline)
        self.assertEqual(result["master_pids"], [321])
        self.assertEqual(result["master_returncodes"], [1])
        self.assertEqual(
            [x["tail"] for x in result["logs"]],
            ["ConfigValidationException", "owned stdout"],
        )
        with patch.object(backend, "_setup", side_effect=RuntimeError("before JVM")):
            with self.assertRaisesRegex(RuntimeError, "no owned Master"):
                backend.probe_startup(self.ctx, {}, "{}", self.deadline)

    def test_real_harness_startup_failure_tail_uses_isolated_log(self):
        from flexlb_test_framework.harness import EnvManager, EnvSpec

        directory = Path(self.tmp.name)
        log_root = directory / "private"
        log_root.mkdir()
        app_log = log_root / "application.log"
        app_log.write_text("previous bytes\n")
        jar = directory / "placeholder.jar"
        jar.touch()
        spec = EnvSpec(
            label="probe", master_extra_args=[f"--flexlb.log.path={log_root}"]
        )
        env = NS(
            spec=spec,
            master=None,
            master_start_count=0,
            master_http_port=28000,
            master_management_port=28001,
            run_dir=directory,
        )
        manager = EnvManager(directory)

        def launch(*args, **kwargs):
            with app_log.open("a") as stream:
                stream.write("ConfigValidationException from owned start\n")
            return NS(alive=lambda: False, tail_log=lambda: "owned stdout")

        with patch("flexlb_test_framework.harness.API_JAR", jar), patch(
            "flexlb_test_framework.harness.resolve_java21", return_value="unused-java"
        ), patch(
            "flexlb_test_framework.harness.port_in_use", return_value=False
        ), patch.object(
            manager, "_master_env", return_value={}
        ), patch(
            "flexlb_test_framework.harness.ProcessOps.start", side_effect=launch
        ):
            with self.assertRaises(RuntimeError) as error:
                manager.start_master(env)
        self.assertIn(
            "ConfigValidationException from owned start", str(error.exception)
        )
        self.assertNotIn("previous bytes", str(error.exception))
        self.assertIn(str(app_log), str(error.exception))


if __name__ == "__main__":
    unittest.main()
