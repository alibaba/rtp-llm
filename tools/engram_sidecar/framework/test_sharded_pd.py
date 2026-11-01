import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.engram_sidecar.framework import sharded_pd as pd
from rtp_llm.models_py.modules.dsv41.engram_capacity import CapacityError, validate_capacity


def topology():
    return {"schema": 1, "nodes": {
        name: {"ip": f"192.0.2.{i+1}", "port": 18180,
               "gpu_uuids": [f"GPU-00000000-0000-0000-0000-{i*4+j:012x}" for j in range(4)]}
        for i, name in enumerate(pd.NODES)}}


class ShardedPDTest(unittest.TestCase):
    def test_preflight_capacity_uses_explicit_arguments(self):
        args = SimpleNamespace(max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        limits = validate_capacity(args.max_allocation_bytes, args.budget_bytes)
        self.assertEqual(limits, (128 << 30, 256 << 30))
        args.budget_bytes = 0
        with self.assertRaises(CapacityError):
            validate_capacity(args.max_allocation_bytes, args.budget_bytes)

    def env(self, node, graph=False, storage="external-egm", decode_moe="mega"):
        args = SimpleNamespace(checkpoint=Path("/model"),
                               storage_mode=storage, cuda_graph=graph,
                               max_allocation_bytes=128*1024**3, budget_bytes=256*1024**3,
                               port=18180, decode_moe_strategy=decode_moe)
        return pd.launch_environment(topology(), node, args, inherited={
            "TP_SIZE": "8", "DP_SIZE": "1", "WORLD_RANK": "99",
            "GANG_CONFIG_STRING": "stale", "GANG_CONFIG_PATH": "/old",
            "PREFILL_CP_KV_CACHE_SHARDED": "0", "LD_LIBRARY_PATH": "/native"})

    def test_decode_is_one_dpep8_gang(self):
        envs = [self.env(n) for n in ("d0", "d1")]
        for e in envs:
            self.assertEqual([e[k] for k in ("TP_SIZE", "DP_SIZE", "EP_SIZE")], ["1", "8", "8"])
            self.assertEqual(e["WORLD_SIZE"], "8")
            self.assertEqual(e["LOCAL_WORLD_SIZE"], "4")
            self.assertEqual(e["DSV4_MOE_STRATEGY"], "mega")
            self.assertEqual(e["MOE_STRATEGY"], "mega_moe")
            self.assertEqual(e["PREFILL_CP_SIZE"], "4")
            self.assertEqual(e["PREFILL_CP_KV_CACHE_SHARDED"], "1")
            self.assertEqual(e["LD_LIBRARY_PATH"], "/native")
        self.assertEqual([e["WORLD_RANK"] for e in envs], ["0", "4"])
        self.assertEqual(envs[0]["GANG_CONFIG_STRING"], envs[1]["GANG_CONFIG_STRING"])
        self.assertIn("name:decode_part1,ip:192.0.2.4", envs[0]["GANG_CONFIG_STRING"])

    def test_decode_backend_is_independent_of_parallel_topology(self):
        for name in ("d0", "d1"):
            env = self.env(name, decode_moe="deepep")
            self.assertEqual(env["DSV4_MOE_STRATEGY"], "deepep")
            self.assertEqual(env["MOE_STRATEGY"], "auto")
            self.assertEqual([env[k] for k in ("TP_SIZE", "DP_SIZE", "EP_SIZE")], ["1", "8", "8"])
        self.assertEqual(self.env("p0", decode_moe="deepep")["DSV4_MOE_STRATEGY"], "mega")

    def test_two_independent_prefills_and_local_frontends(self):
        for i, name in enumerate(("p0", "p1")):
            e = self.env(name, graph=True)
            self.assertEqual([e[k] for k in ("TP_SIZE", "DP_SIZE", "EP_SIZE", "WORLD_SIZE")], ["4", "1", "4", "4"])
            self.assertEqual(e["WORLD_RANK"], "0")
            self.assertEqual(e["DSV4_MOE_STRATEGY"], "mega")
            self.assertNotIn("GANG_CONFIG_STRING", e)
            self.assertNotIn("GANG_CONFIG_PATH", e)
            self.assertEqual(e["ENABLE_CUDA_GRAPH"], "0")
            route = json.loads(e["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
            self.assertEqual(route["prefill_endpoint"]["address"], f"192.0.2.{i+1}:18180")
            self.assertEqual(route["decode_endpoint"]["address"], "192.0.2.3:18180")

    def test_pd_cache_contract_matches_and_graph_is_decode_only(self):
        envs = [self.env(n, graph=True) for n in pd.NODES]
        for key in ("PREFILL_CP_SIZE", "PREFILL_CP_KV_CACHE_SHARDED", "SEQ_SIZE_PER_BLOCK",
                    "KERNEL_SEQ_SIZE_PER_BLOCK", "CACHE_STORE_RDMA_MODE", "DSV41_CED",
                    "DSV41_SWA_BOUNDED_REPLAY", "RTP_ENGRAM_TOPOLOGY_SHA256"):
            self.assertEqual(len({e[key] for e in envs}), 1, key)
        self.assertEqual([e["ENABLE_CUDA_GRAPH"] for e in envs], ["0", "0", "1", "1"])
        for node in pd.NODES:
            local = self.env(node, graph=True, storage="local-pinned")
            for key in pd.full_model_http.EXTERNAL_ENV:
                self.assertNotIn(key, local)

    def test_external_cohort_contract_reaches_the_server(self):
        env = pd.launch_environment(
            topology(), "d0", SimpleNamespace(
                checkpoint=Path("/model"), storage_mode="external-egm", cuda_graph=False,
                max_allocation_bytes=128 * 1024**3, budget_bytes=256 * 1024**3,
                port=18180, decode_moe_strategy="mega"),
            inherited={"RTP_ENGRAM_COHORT": "192.0.2.1:19780,192.0.2.2:19780",
                       "RTP_ENGRAM_INSTANCE_INDEX": "1", "ENGRAM_SHARD_COUNT": "2"})
        self.assertEqual(env["RTP_ENGRAM_COHORT"], "192.0.2.1:19780,192.0.2.2:19780")
        self.assertEqual(env["RTP_ENGRAM_INSTANCE_INDEX"], "1")
        self.assertEqual(env["ENGRAM_SHARD_COUNT"], "2")

    def test_reject_colocated_hosts_duplicate_devices_and_wrong_gang_ports(self):
        original = topology()
        for key, value in (("ip", original["nodes"]["p0"]["ip"]),
                           ("gpu_uuids", original["nodes"]["p0"]["gpu_uuids"]),
                           ("port", 18181)):
            bad = copy.deepcopy(original)
            bad["nodes"]["d1"][key] = value
            with self.assertRaises(ValueError):
                pd.validate_topology(bad)


if __name__ == "__main__":
    unittest.main()
