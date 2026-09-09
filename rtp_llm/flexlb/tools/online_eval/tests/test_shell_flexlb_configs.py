import json
import re
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from flexlb_ft.harness import EnvManager, EnvSpec, MasterSpec, build_flexlb_config


class ShellFlexlbConfigTest(unittest.TestCase):
    def assert_v3(self, config):
        self.assertEqual(3, config["schemaVersion"])
        prefill = config["router"]["roles"].get("prefill", {})
        self.assertNotIn("maxUncachedTokens", prefill.get("availability", {}))
        self.assertNotIn("availability", prefill)
        self.assertGreater(config["requestLifecycle"]["request"]["timeoutMs"], 0)
        self.assertGreaterEqual(config["requestLifecycle"]["decision"]["lifetime"], 1)
        self.assertNotIn("capacity", config["scheduler"])
        self.assertNotIn("lifecycle", config["scheduler"])
        self.assertNotIn("candidateChoice", prefill)
        self.assertNotIn("enqueueRpcTimeoutMs", config["dispatcher"])
        for field in ("maxInflightRequestsPerPrefillWorker", "maxInflightBatchesPerPrefillWorker",
                      "inflightRequestMultiplier", "inflightMultiplier"):
            self.assertNotIn(field, config["dispatcher"])
        for key in ("kvReservation", "decayPerToken", "loadDecayPerRequest", "outlierRejection"):
            self.assertNotIn(key, config["router"]["roles"]["decode"])

    def test_shell_and_shipped_master_use_complete_v3_contract(self):
        script = (SCRIPT_DIR / "run_online_eval.sh").read_text()
        match = re.search(r"DEFAULT_FLEXLB_CONFIG='(\{.*?\})'", script, re.S)
        self.assertIsNotNone(match)
        self.assert_v3(json.loads(match.group(1)))
        for filename in ("master_fixed_window.json", "master_fixed_window_4g.json",
                         "master_fixed_window_slo500_wait160.json"):
            with self.subTest(filename=filename):
                master = json.loads((SCRIPT_DIR / "data/config" / filename).read_text())
                env = dict(master["zone_process_setting"]["process_info"]["envs"])
                self.assert_v3(json.loads(env["FLEXLB_CONFIG"]))

    def test_smoke_matrix_preserves_its_three_mode_contracts(self):
        script = (SCRIPT_DIR / "run_matrix_smoke.sh").read_text()
        documents = re.findall(r"FLEXLB_CONFIG='(\{.*?\})'", script, re.S)
        self.assertEqual(3, len(documents))
        expected = (
            ("QUEUE", "PRIORITY", "FIXED_WINDOW", "BATCH", 4),
            ("DIRECT", None, None, "NON_BATCH", 2),
            ("QUEUE", "FIFO", "SINGLE", "NON_BATCH", 2),
        )
        for raw, (scheduler, ordering, decision, dispatcher, concurrency) in zip(documents, expected):
            with self.subTest(scheduler=scheduler, ordering=ordering):
                config = json.loads(raw)
                self.assert_v3(config)
                self.assertEqual(scheduler, config["scheduler"]["type"])
                self.assertEqual(ordering, config["scheduler"].get("ordering", {}).get("type"))
                self.assertEqual(decision, config["scheduler"].get("decision", {}).get("type"))
                self.assertEqual({"type": dispatcher, "maxInflightPerPrefillWorker": concurrency},
                                 config["dispatcher"])
                self.assertEqual(132, config["router"]["roles"]["decode"]["availability"]["maxEngineRequests"])
                if scheduler == "DIRECT":
                    self.assertNotIn("queueTimeoutMs", config["scheduler"])

    def test_active_smoke_and_recovery_scripts_use_v3_defaults(self):
        for filename in ("run_cancel_smoke.sh", "run_batch_smoke_only.sh",
                         "engine_kill_restart_test.sh", "engine_disconnect_ttft_test.sh",
                         "master_kill_restart_test.sh", "master_recovery_ttft_test.sh"):
            with self.subTest(filename=filename):
                script = (SCRIPT_DIR / filename).read_text()
                match = re.search(r"(?:DEFAULT_FLEXLB_CONFIG|default_flexlb_config)='(\{.*?\})'",
                                  script, re.S)
                self.assertIsNotNone(match)
                config = json.loads(match.group(1))
                self.assert_v3(config)
                self.assertEqual({"type": "BATCH", "maxInflightPerPrefillWorker": 4},
                                 config["dispatcher"])
                self.assertFalse('"FLEXLB_EXPECT_FETCH_RESPONSE=' in script,
                                 "removed fetch-response environment setting")
                self.assertFalse('"DOMAIN_ADDRESS:' in script,
                                 "service hosts must be inside MODEL_SERVICE_CONFIG")

    def test_behavior_profile_keeps_explicit_lifetime_and_worker_health_inputs(self):
        script = (SCRIPT_DIR / "flexlb_behavior_test.sh").read_text()
        match = re.search(r"DEFAULT_FLEXLB_CONFIG=.*?<<'PY'\n(.*?)\nPY", script, re.S)
        self.assertIsNotNone(match)
        result = subprocess.run([sys.executable, "-c", match.group(1), "12345", "7", "6000"],
                                check=True, capture_output=True, text=True)
        config = json.loads(result.stdout)
        self.assert_v3(config)
        self.assertEqual(12345, config["requestLifecycle"]["request"]["timeoutMs"])
        self.assertEqual(7, config["dispatcher"]["maxInflightPerPrefillWorker"])
        self.assertEqual(12000, config["workerRegistry"]["health"]["statusStaleAfterMs"])

    def test_inflight_experiment_uses_v3_and_retains_its_prefill_formula(self):
        module = runpy.run_path(str(SCRIPT_DIR / "run_inflight_experiment.py"))
        config = json.loads(module["DEFAULT_FLEXLB_CONFIG"])
        self.assert_v3(config)
        self.assertEqual(module["PREFILL_EXECUTION_TIME_EXPRESSION"],
                         config["router"]["roles"]["prefill"]["executionTimeEstimator"]["expression"])
        self.assertEqual(220, config["scheduler"]["decision"]["maxCollectionWaitMs"])

    def test_engine_restart_discovery_keeps_all_hosts_inside_model_config(self):
        script = (SCRIPT_DIR / "engine_kill_restart_test.sh").read_text()
        match = re.search(r"<<'PY_MODEL'\n(.*?)\nPY_MODEL", script, re.S)
        self.assertIsNotNone(match)
        result = subprocess.run([sys.executable, "-c", match.group(1),
                                 "127.0.0.1:55150,127.0.0.1:55151", "127.0.0.1:55250"],
                                check=True, capture_output=True, text=True)
        model = json.loads(result.stdout)
        endpoints = model["role_endpoints"][0]
        self.assertEqual(["127.0.0.1:55150", "127.0.0.1:55151"],
                         model["hosts"][endpoints["prefill_endpoint"]["address"]])
        self.assertEqual(["127.0.0.1:55250"],
                         model["hosts"][endpoints["decode_endpoint"]["address"]])

    def test_matrix_generates_only_active_mode_parameters(self):
        for ordering in ("fifo", "priority"):
            for decision in ("single", "fixed_window"):
                for dispatcher in ("batch", "non_batch"):
                    config = json.loads(build_flexlb_config(
                        ordering=ordering, decision=decision, dispatcher=dispatcher,
                        max_inflight_per_prefill_worker=7, request_timeout_ms=9876, decision_lifetime=3.5))
                    self.assert_v3(config)
                    self.assertEqual(9876, config["requestLifecycle"]["request"]["timeoutMs"])
                    self.assertEqual({"type": dispatcher.upper(), "maxInflightPerPrefillWorker": 7}, config["dispatcher"])
                    if decision == "single":
                        self.assertEqual({"type": "SINGLE"}, config["scheduler"]["decision"])

    def test_shared_concurrency_default_and_positive_integer_validation(self):
        for dispatcher in ("batch", "non_batch"):
            config = json.loads(build_flexlb_config(dispatcher=dispatcher))
            self.assertEqual(2, config["dispatcher"]["maxInflightPerPrefillWorker"])
            for value in (None, 0, -1, 1.5, True, "2", float("inf"), float("nan"), 2_147_483_648):
                with self.assertRaises(ValueError):
                    build_flexlb_config(dispatcher=dispatcher, max_inflight_per_prefill_worker=value)

    def test_removed_generator_arguments_are_not_silent_noops(self):
        for field in ("max_inflight_batches", "inflight_request_multiplier", "inflight_multiplier", "max_predicted_queue_wait_ms", "max_uncached_tokens", "max_outstanding", "max_delivered_not_accepted", "stale_inflight_ms",
                      "max_waiting_requests_per_prefill_worker", "max_inflight_requests_per_worker"):
            with self.assertRaises(TypeError):
                build_flexlb_config(**{field: 1})

    def test_master_env_contains_config_documents_and_role_hosts(self):
        with TemporaryDirectory() as tmp:
            spec = EnvSpec(discovery="domain", domain_addrs={
                "prefill": "10.0.0.1:8000,10.0.0.2:8000", "decode": "10.0.0.3:9000"})
            env = SimpleNamespace(spec=spec, run_dir=Path(tmp), zk_connect_string="127.0.0.1:2181")
            manager = EnvManager(Path(tmp), verbose=False)
            result = manager._master_env(env)
            self.assertEqual({"FLEXLB_CONFIG", "MODEL_SERVICE_CONFIG"}, set(result))
            model = json.loads(result["MODEL_SERVICE_CONFIG"])
            self.assertEqual(["10.0.0.1:8000", "10.0.0.2:8000"], model["hosts"]["mock.prefill.hosts.address"])
            self.assertEqual(["10.0.0.3:9000"], model["hosts"]["mock.decode.hosts.address"])
            spec.zk_consistency = {"zkTimeoutMs": 10000}
            result = manager._master_env(env, MasterSpec(name="A", http_port=18080))
            self.assertEqual({"needConsistency", "zookeeperConfig"},
                             set(json.loads(result["FLEXLB_SYNC_CONSISTENCY_CONFIG"])))

    def test_preemption_schema_contains_only_implemented_decode_stages(self):
        for stages in (["DECODE_RESERVED"], ["DECODE_ENGINE_OWNED"],
                       ["DECODE_RESERVED", "DECODE_ENGINE_OWNED"]):
            config = json.loads(build_flexlb_config(ordering="priority", preemption={
                "allowed_victim_stages": stages}))
            self.assertEqual(stages, config["scheduler"]["ordering"]["preemption"]["allowedVictimStages"])
        for stages in (["PREFILL_QUEUED"], ["DECODE_RESERVED", "PREFILL_QUEUED"]):
            with self.assertRaises(ValueError):
                build_flexlb_config(ordering="priority", preemption={"allowed_victim_stages": stages})


if __name__ == "__main__":
    unittest.main()
