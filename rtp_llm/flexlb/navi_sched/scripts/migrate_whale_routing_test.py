import copy
import json
import unittest

from migrate_whale_routing import ROUTING_ENVS, Whale, clean_template, env_map, plan_deployment


class RoutingMigrationTest(unittest.TestCase):
    def setUp(self):
        self.prefix = "com.aicheng.whale.prod.model."
        self.master = self.prefix + "master.deployment"
        self.agent = self.prefix + "navi_sched.deployment"
        self.deployment = {
            "biz_name": "Model", "deployment_name": "deployment",
            "zone_deployments": [
                {"zone_name": "inference", "zone_type": "PY_INFERENCE",
                 "service_discoveries": [{"type": "vip", "domain": self.prefix + "publish",
                                           "online_service_discovery_id": "provider", "enable": False}],
                 "zone_process_setting": {"process_info": {"envs": [["UNRELATED", "keep"]],
                                                            "args": [["x", "y"]]},
                                          "resource_plan": {"unchanged": 1}}},
                {"zone_name": "master", "zone_type": "PY_MASTER",
                 "service_discoveries": [{"domain": self.master}]},
            ],
        }
        config = json.dumps({"scheduler_name": "cost", "runtime_config": {
            "local_nodes": [{"ip": "foreign"}], "vip_subscription": {"services": ["foreign"]}},
            "scheduler_config": {"learning_rate": 16}})
        self.content = {
            "image_packages": {"amd_image": "preserve-image"},
            "inference_zone_plan": {"default_process_envs": [
                ["VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN", "foreign"], ["VLLM_NAVI_SCHED_ENABLE", "1"]],
                "zone_resource_templates": [{"process_info": {"envs": [
                    ["VLLM_NAVI_SCHED_BIZ_NAME", "foreign"], ["VLLM_NAVI_SCHED_ENABLE", "1"],
                    ["UNRELATED", "keep"]]}}]},
            "master_zone_plan": {"default_process_envs": [["NAVI_SCHED_MASTER_CONFIG_JSON", config]],
                                 "zone_resource_templates": [{"process_info": {"envs": []}}]},
        }

    def test_both_ends_and_private_discovery(self):
        before = copy.deepcopy(self.deployment)
        plan = plan_deployment(self.deployment, {"content": self.content}, self.master, self.agent, True)
        self.assertEqual(before, self.deployment)
        master, client = plan["processes"]
        config = json.loads(dict(master["zone_process_setting"]["process_info"]["envs"])["NAVI_SCHED_MASTER_CONFIG_JSON"])
        self.assertEqual([self.agent], config["runtime_config"]["vip_subscription"]["services"])
        self.assertNotIn("local_nodes", config["runtime_config"])
        self.assertEqual({"learning_rate": 16}, config["scheduler_config"])
        self.assertEqual("deployment", config["kmon_tags"]["deployment_name"])
        setting = client["zone_process_setting"]
        envs = dict(setting["process_info"]["envs"])
        self.assertEqual(self.master, envs["VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN"])
        self.assertEqual("vllm_navi_sched_master", envs["VLLM_NAVI_SCHED_MASTER_BIZ_NAME"])
        self.assertEqual(self.agent, envs["VLLM_NAVI_SCHED_TARGET_BIZ_NAME"])
        self.assertEqual(self.agent, envs["VLLM_NAVI_SCHED_BIZ_NAME"])
        self.assertEqual("", envs["VLLM_NAVI_SCHED_MASTER_ADDRESSES"])
        self.assertEqual("keep", envs["UNRELATED"])
        self.assertEqual({"unchanged": 1}, setting["resource_plan"])
        self.assertEqual([["x", "y"]], setting["process_info"]["args"])
        discoveries = plan["discoveries"][0]["service_discoveries"]
        self.assertFalse(discoveries[0]["enable"])
        self.assertEqual(self.agent, discoveries[1]["domain"])
        self.assertTrue(discoveries[1]["enable"])

    def test_template_has_no_routing_and_preserves_images(self):
        before = copy.deepcopy(self.content)
        cleaned = clean_template(self.content)
        self.assertEqual(before, self.content)
        self.assertEqual(before["image_packages"], cleaned["image_packages"])
        for name in ROUTING_ENVS:
            self.assertNotIn(name, json.dumps(cleaned))
        self.assertEqual([["VLLM_NAVI_SCHED_ENABLE", "0"]], cleaned["inference_zone_plan"]["default_process_envs"])
        self.assertEqual(cleaned, clean_template(cleaned))

    def test_cross_service_and_unregistered_master_rejected(self):
        for domain in ["com.aicheng.whale.prod.foreign.master", self.prefix + "master.unknown"]:
            with self.assertRaises(ValueError):
                plan_deployment(self.deployment, {"content": self.content}, domain, self.agent, True)

    def test_worker_only_and_existing_discovery(self):
        self.deployment["zone_deployments"].pop()
        existing = self.deployment["zone_deployments"][0]["service_discoveries"][0]
        existing["domain"] = self.agent
        plan = plan_deployment(self.deployment, {"content": self.content}, self.master, self.agent, True)
        self.assertEqual(1, len(plan["processes"]))
        services = plan["discoveries"][0]["service_discoveries"]
        self.assertEqual(1, len(services))
        self.assertTrue(services[0]["enable"])
        self.assertFalse(existing["enable"])

    def test_snapshot_version_string_matches_deployment_integer(self):
        api = object.__new__(Whale)
        api.request = lambda *args: [{"biz_version": "15"}, {"biz_version": "16"}]
        self.assertEqual({"biz_version": "16"}, api.snapshot({"id": "deployment", "biz_version": 16}))

    def test_draft_and_snapshot_environment_formats(self):
        self.assertEqual({"x": "value"}, env_map([["x", "value"]]))
        self.assertEqual({"x": "value"}, env_map([{"name": "x", "value": "value", "desc": "metadata"}]))


if __name__ == "__main__":
    unittest.main()
