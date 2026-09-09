"""Exercise owned launch boundaries and actual config writers, with no Java."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_test_framework.harness import _write_master_config
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.actions.environment import mutate_config
from flexlb_test_framework.scenario.backend import JavaMockBackend
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from test_scenario_backend import lease_manifest


class RawMasterProbeTest(unittest.TestCase):
    def plan(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/priority/priority_preemption.yaml"
        )
        return next(
            p
            for p in compile_scenarios(load_scenarios(path), handlers=handlers())
            if p["variant_id"] == "config_strict_reject"
        )

    def test_each_bad_document_reaches_master_while_mock_uses_valid_base(self):
        instance = self.plan()
        probes = [
            s for s in instance["stages"] if s["action"] == "environment_startup_probe"
        ]
        self.assertEqual(len(probes), 3)
        for stage in probes:
            with self.subTest(stage=stage["id"]), tempfile.TemporaryDirectory() as tmp:
                params = stage["params"]
                plan = params["environments"][instance["profile"]]
                invalid = mutate_config(plan["resolved_config"], params["mutation"])
                backend = JavaMockBackend(lease_manifest())
                ctx = RuntimeContext(instance, backend, tmp, time.monotonic, time.sleep)
                ctx.env_epoch = 2
                seen = {}

                def ensure(manager, spec):
                    directory = manager.run_root / "env1_scenario"
                    directory.mkdir(parents=True)
                    endpoint = directory / "endpoints.json"
                    endpoint.write_text(json.dumps({"env": {}}))
                    env = NS(
                        spec=spec,
                        run_dir=directory,
                        endpoint_file=endpoint,
                        master=None,
                        mock=None,
                        zk_helper=None,
                        masters={},
                        victims={},
                        load_clients=[],
                    )
                    manager._start_mock(env)
                    self.assertIs(env.spec, spec)
                    manager.start_master(env)

                def start_mock(manager, env):
                    self.assertIsNone(env.spec.raw_config)
                    config = json.loads(_write_master_config(env).read_text())
                    variables = dict(
                        config["zone_process_setting"]["process_info"]["envs"]
                    )
                    seen["mock"] = json.loads(variables["FLEXLB_CONFIG"])
                    env.endpoint_file.write_text(
                        json.dumps(
                            {
                                "env": {
                                    "MODEL_SERVICE_CONFIG": json.dumps(
                                        {
                                            "hosts": {
                                                "mock.prefill.hosts.address": [],
                                                "mock.decode.hosts.address": [],
                                            }
                                        }
                                    )
                                }
                            }
                        )
                    )
                    env.mock = NS(pid=401)

                def start_master(manager, env):
                    # Use the real Master environment renderer; do not infer
                    # the raw document from the spec without checking its wire.
                    seen["master"] = json.loads(
                        manager._master_env(env)["FLEXLB_CONFIG"]
                    )
                    log_dir = Path(
                        next(
                            arg.split("=", 1)[1]
                            for arg in env.spec.master_extra_args
                            if arg.startswith("--flexlb.log.path=")
                        )
                    )
                    (log_dir / "application.log").write_text(
                        "ConfigValidationException: Unrecognized field"
                    )
                    stdout = env.run_dir / "flexlb_master.log"
                    stdout.write_text("owned Master stdout")
                    env.master = NS(pid=402, proc=NS(poll=lambda: 1), log_file=stdout)
                    raise RuntimeError("Master rejected raw config")

                with patch(
                    "flexlb_test_framework.harness.EnvManager.ensure", ensure
                ), patch(
                    "flexlb_test_framework.harness.EnvManager._start_mock", start_mock
                ), patch(
                    "flexlb_test_framework.harness.EnvManager.start_master",
                    start_master,
                ):
                    result = backend.probe_startup(
                        ctx, plan, json.dumps(invalid), Deadline(time.monotonic() + 2)
                    )
                self.assertEqual(seen["mock"], plan["resolved_config"])
                self.assertEqual(seen["master"], invalid)
                self.assertEqual(result["master_pids"], [402])
                self.assertEqual(result["master_returncodes"], [1])
                self.assertEqual(set(backend.owned_processes), {401, 402})
                evidence = json.loads(
                    (Path(tmp) / "environment-epoch-2/environment.json").read_text()
                )
                self.assertEqual(evidence["raw_config_target"], "master")
                self.assertEqual(json.loads(evidence["raw_config"]), invalid)

    def test_mock_failure_keeps_ownership_and_restores_target_spec_without_master_claim(
        self,
    ):
        instance = self.plan()
        params = next(
            s for s in instance["stages"] if s["action"] == "environment_startup_probe"
        )["params"]
        plan = params["environments"][instance["profile"]]
        raw = json.dumps(mutate_config(plan["resolved_config"], params["mutation"]))
        with tempfile.TemporaryDirectory() as tmp:
            backend = JavaMockBackend(lease_manifest())
            ctx = RuntimeContext(instance, backend, tmp, time.monotonic, time.sleep)
            ctx.env_epoch = 2

            def ensure(manager, spec):
                env = NS(
                    spec=spec,
                    master=None,
                    mock=None,
                    zk_helper=None,
                    masters={},
                    victims={},
                    load_clients=[],
                )
                manager._start_mock(env)

            def fail_mock(manager, env):
                self.assertIsNone(env.spec.raw_config)
                env.mock = NS(pid=501)
                raise RuntimeError("supporting mock failed")

            with patch(
                "flexlb_test_framework.harness.EnvManager.ensure", ensure
            ), patch(
                "flexlb_test_framework.harness.EnvManager._start_mock", fail_mock
            ), patch(
                "flexlb_test_framework.harness.EnvManager.start_master"
            ) as master:
                with self.assertRaisesRegex(RuntimeError, "no owned Master"):
                    backend.probe_startup(
                        ctx, plan, raw, Deadline(time.monotonic() + 2)
                    )
                master.assert_not_called()
            self.assertEqual(backend.environments[0].spec.raw_config, raw)
            self.assertEqual(set(backend.owned_processes), {501})


if __name__ == "__main__":
    unittest.main()
