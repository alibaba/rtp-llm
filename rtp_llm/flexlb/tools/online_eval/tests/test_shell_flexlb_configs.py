import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from flexlb_test_framework.harness import (
    EnvManager,
    EnvSpec,
    MasterSpec,
    build_flexlb_config,
)


class ShellFlexlbConfigTest(unittest.TestCase):
    def assert_v3(self, config):
        self.assertEqual(3, config["schemaVersion"])
        prefill = config["router"]["roles"]["prefill"]
        self.assertNotIn("maxUncachedTokens", prefill.get("availability", {}))
        self.assertNotIn("availability", prefill)
        self.assertGreater(config["requestLifecycle"]["request"]["timeoutMs"], 0)
        self.assertGreaterEqual(config["requestLifecycle"]["decision"]["lifetime"], 1)
        self.assertNotIn("capacity", config["scheduler"])
        self.assertNotIn("lifecycle", config["scheduler"])
        self.assertNotIn("candidateChoice", config["router"]["roles"]["prefill"])
        self.assertNotIn("enqueueRpcTimeoutMs", config["dispatcher"])
        for field in (
            "maxInflightRequestsPerPrefillWorker",
            "maxInflightBatchesPerPrefillWorker",
            "inflightRequestMultiplier",
            "inflightMultiplier",
        ):
            self.assertNotIn(field, config["dispatcher"])
        for key in (
            "kvReservation",
            "decayPerToken",
            "loadDecayPerRequest",
            "outlierRejection",
        ):
            self.assertNotIn(key, config["router"]["roles"]["decode"])

    def test_shell_and_shipped_master_use_complete_v3_contract(self):
        script = (SCRIPT_DIR / "run_online_eval.sh").read_text()
        self.assertNotIn("DEFAULT_FLEXLB_CONFIG=", script)
        from flexlb_cfg import STRESS_PROFILE, render_env, render_process_config

        self.assert_v3(json.loads(render_env(STRESS_PROFILE)))
        master = json.loads(render_process_config(STRESS_PROFILE))
        env = dict(master["zone_process_setting"]["process_info"]["envs"])
        self.assert_v3(json.loads(env["FLEXLB_CONFIG"]))

    def test_matrix_generates_only_active_mode_parameters(self):
        for ordering in ("fifo", "priority"):
            for decision in ("single", "fixed_window"):
                for dispatcher in ("batch", "non_batch"):
                    config = json.loads(
                        build_flexlb_config(
                            ordering=ordering,
                            decision=decision,
                            dispatcher=dispatcher,
                            max_inflight_per_prefill_worker=7,
                            request_timeout_ms=9876,
                            decision_lifetime=3.5,
                        )
                    )
                    self.assert_v3(config)
                    self.assertEqual(
                        9876, config["requestLifecycle"]["request"]["timeoutMs"]
                    )
                    self.assertEqual(
                        {"type": dispatcher.upper(), "maxInflightPerPrefillWorker": 7},
                        config["dispatcher"],
                    )
                    if decision == "single":
                        self.assertEqual(
                            {"type": "SINGLE"}, config["scheduler"]["decision"]
                        )

    def test_shared_concurrency_default_and_positive_integer_validation(self):
        for dispatcher in ("batch", "non_batch"):
            config = json.loads(build_flexlb_config(dispatcher=dispatcher))
            self.assertEqual(2, config["dispatcher"]["maxInflightPerPrefillWorker"])
            for value in (
                None,
                0,
                -1,
                1.5,
                True,
                "2",
                float("inf"),
                float("nan"),
                2_147_483_648,
            ):
                with self.assertRaises(ValueError):
                    build_flexlb_config(
                        dispatcher=dispatcher, max_inflight_per_prefill_worker=value
                    )

    def test_removed_generator_arguments_are_not_silent_noops(self):
        for field in (
            "max_inflight_batches",
            "inflight_request_multiplier",
            "inflight_multiplier",
            "max_predicted_queue_wait_ms",
            "max_uncached_tokens",
            "max_outstanding",
            "max_delivered_not_accepted",
            "stale_inflight_ms",
            "max_waiting_requests_per_prefill_worker",
            "max_inflight_requests_per_worker",
        ):
            with self.assertRaises(TypeError):
                build_flexlb_config(**{field: 1})

    def test_master_env_contains_config_documents_and_role_hosts(self):
        with TemporaryDirectory() as tmp:
            spec = EnvSpec(
                discovery="domain",
                domain_addrs={
                    "prefill": "10.0.0.1:8000,10.0.0.2:8000",
                    "decode": "10.0.0.3:9000",
                },
            )
            env = SimpleNamespace(
                spec=spec, run_dir=Path(tmp), zk_connect_string="127.0.0.1:2181"
            )
            manager = EnvManager(Path(tmp), verbose=False)
            result = manager._master_env(env)
            self.assertTrue({"FLEXLB_CONFIG", "MODEL_SERVICE_CONFIG"} <= set(result))
            model = json.loads(result["MODEL_SERVICE_CONFIG"])
            self.assertEqual(
                ["10.0.0.1:8000", "10.0.0.2:8000"],
                model["hosts"]["mock.prefill.hosts.address"],
            )
            self.assertEqual(
                ["10.0.0.3:9000"], model["hosts"]["mock.decode.hosts.address"]
            )
            spec.zk_consistency = {"zkTimeoutMs": 10000}
            result = manager._master_env(env, MasterSpec(name="A", http_port=18080))
            self.assertEqual(
                {"needConsistency", "zookeeperConfig"},
                set(json.loads(result["FLEXLB_SYNC_CONSISTENCY_CONFIG"])),
            )

    def test_preemption_schema_contains_only_implemented_decode_stages(self):
        for stages in (
            ["DECODE_RESERVED"],
            ["DECODE_ENGINE_OWNED"],
            ["DECODE_RESERVED", "DECODE_ENGINE_OWNED"],
        ):
            config = json.loads(
                build_flexlb_config(
                    ordering="priority", preemption={"allowed_victim_stages": stages}
                )
            )
            self.assertEqual(
                stages,
                config["scheduler"]["ordering"]["preemption"]["allowedVictimStages"],
            )
        for stages in (["PREFILL_QUEUED"], ["DECODE_RESERVED", "PREFILL_QUEUED"]):
            self.assert_v3(
                json.loads(
                    build_flexlb_config(
                        ordering="priority",
                        preemption={"allowed_victim_stages": stages},
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
