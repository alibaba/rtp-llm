import importlib.util
import json
from pathlib import Path
import unittest

module_path = Path(__file__).resolve().parents[1] / "navi_sched/tair_kvcm_config.py"
spec = importlib.util.spec_from_file_location("tair_kvcm_config", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
make_config = module.make_tair_kvcm_resource_config


class BootstrapConfigTest(unittest.TestCase):
    def test_absent_topology_stays_dormant(self):
        for env in ({}, {"MODEL_SERVICE_CONFIG": ""}, {"MODEL_SERVICE_CONFIG": '{"service_id":"s"}'}):
            self.assertEqual({}, make_config(env=env))

    def test_flexlb_topology_maps_to_resource_without_worker_address(self):
        env = {"MODEL_SERVICE_CONFIG": json.dumps({
            "service_id": "model", "role_endpoints": [],
            "kvcm": {"address": "v-kvcm", "discovery": {"type": "dashscope"}},
        })}
        self.assertEqual({"address": "v-kvcm", "port": 6381,
                          "discovery": {"type": "spectrum"}, "thread_num": 4},
                         make_config(env=env))

    def test_explicit_resource_config_overrides_environment(self):
        explicit = {"address": "127.0.0.1", "port": 17001, "thread_num": 2,
                    "discovery": {"type": "direct"}, "namespace": " test "}
        result = make_config(explicit, {"MODEL_SERVICE_CONFIG": "invalid"})
        self.assertEqual(2, result["thread_num"])
        self.assertEqual(17001, result["port"])
        self.assertEqual("test", result["namespace"])
        self.assertEqual({}, make_config({}, {"MODEL_SERVICE_CONFIG": "invalid"}))

    def test_invalid_topology_rejected_before_startup(self):
        valid = {"address": "v-kvcm", "discovery": {"type": "dashscope"}}
        for key, value in (("port", 0), ("port", 65536), ("port", True),
                           ("thread_num", 0), ("thread_num", 1.5),
                           ("address", "bad/service"), ("namespace", 4),
                           ("discovery", {})):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                make_config(dict(valid, **{key: value}))
        with self.assertRaises(ValueError):
            make_config(env={"MODEL_SERVICE_CONFIG": '{"kvcm":{},"kvcm":{}}'})

    def test_vipserver_uses_resource_or_existing_gig_platform_domain(self):
        result = make_config({"address": "kvcm.test.vip", "discovery": {"type": "VIPSERVER"}})
        self.assertEqual({"type": "vipserver"}, result["discovery"])
        result = make_config({"address": "kvcm.test.vip", "discovery": {
            "type": "vipserver", "jmenv_dom": " test.jmenv "}})
        self.assertEqual("test.jmenv", result["discovery"]["jmenv_dom"])
        with self.assertRaises(ValueError):
            make_config({"address": "kvcm.test.vip", "discovery": {"type": "vipserver", "jmenv_dom": 1}})

    def test_static_env_hosts_are_validated_and_deduplicated(self):
        result = make_config({"address": "kvcm-static", "port": 6381, "discovery": {
            "type": "static-env", "hosts": ["127.0.0.1:6382", "127.0.0.1:6382", "127.0.0.2:6382"]}})
        self.assertEqual(["127.0.0.1:6382", "127.0.0.2:6382"], result["discovery"]["hosts"])
        self.assertEqual(6381, result["port"])
        for hosts in ([], "127.0.0.1:6382", ["127.0.0.1"], ["127.0.0.1:0"], ["127.0.0.1:65536"], [None]):
            with self.subTest(hosts=hosts), self.assertRaises(ValueError):
                make_config({"address": "kvcm-static", "discovery": {"type": "static-env", "hosts": hosts}})

    def test_does_not_copy_other_flexlb_settings(self):
        result = make_config({"address": "v-kvcm", "discovery": {"type": "dashscope"},
                              "unused_secret": "not-copied", "enabled": True})
        self.assertNotIn("unused_secret", result)
        self.assertNotIn("enabled", result)


if __name__ == "__main__":
    unittest.main()
