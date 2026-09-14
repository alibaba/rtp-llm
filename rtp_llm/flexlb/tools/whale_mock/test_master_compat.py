import json
from pathlib import Path
import unittest
from master_compat import legacy_discovery, mock_formula_config

class MasterCompatibilityTest(unittest.TestCase):
    def test_legacy_document_is_only_projected_for_mock_formula(self):
        raw = Path(__file__).with_name("glm53-inner-master.json").read_text()
        config = json.loads(raw)
        mock = json.loads(mock_formula_config(raw, True))
        self.assertEqual(1, config["schemaVersion"])
        self.assertEqual(3, mock["schemaVersion"])
        self.assertEqual(config["router"]["roles"]["prefill"]["executionTimeEstimator"],
                         mock["router"]["roles"]["prefill"]["executionTimeEstimator"])
        self.assertNotIn("scheduler", mock)
        with self.assertRaises(ValueError): mock_formula_config(raw, False)
        with self.assertRaises(ValueError): mock_formula_config('{"schemaVersion":3}', True)

    def test_old_static_discovery_uses_each_rpc_port_minus_one(self):
        endpoints = {"env": {"MODEL_SERVICE_CONFIG": '{"discovery_file":"unused"}'},
                     "prefill_domain":"p", "decode_domain":"d", "engines":[
                         {"role":"prefill", "http_addr":"10.0.0.1:7050"},
                         {"role":"prefill", "http_addr":"10.0.0.1:7051"},
                         {"role":"decode", "http_addr":"10.0.0.1:7052"}]}
        env = legacy_discovery(endpoints)
        self.assertEqual("10.0.0.1:7050,10.0.0.1:7051", env["DOMAIN_ADDRESS:p"])
        self.assertEqual("10.0.0.1:7052", env["DOMAIN_ADDRESS:d"])
        self.assertNotIn("discovery_file", json.loads(env["MODEL_SERVICE_CONFIG"]))
        self.assertIn("discovery_file", json.loads(endpoints["env"]["MODEL_SERVICE_CONFIG"]))

    def test_file_adapter_receives_path_without_changing_old_master_schema(self):
        endpoints = {"env": {"MODEL_SERVICE_CONFIG": '{"discovery_file":"/tmp/discovery.json"}'},
                     "prefill_domain":"p", "decode_domain":"d", "engines":[
                         {"role":"prefill", "http_addr":"127.0.0.1:7050"},
                         {"role":"decode", "http_addr":"127.0.0.1:7051"}]}
        env = legacy_discovery(endpoints, True)
        self.assertEqual("/tmp/discovery.json", env["MOCK_DISCOVERY_FILE"])
        self.assertNotIn("discovery_file", json.loads(env["MODEL_SERVICE_CONFIG"]))
        self.assertNotIn("MOCK_DISCOVERY_FILE", legacy_discovery(endpoints))
