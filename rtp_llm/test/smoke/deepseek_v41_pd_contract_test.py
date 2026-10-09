"""CED bounded replay over main P/D routing with CP2 prefill and TP2 decode."""

import logging
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke import deepseek_v41_contract_test as contract
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV41PDContractTest(contract.DeepSeekV41ContractTest):
    pd_separation = True
    compare_generation_logits = True
    # Decode reuse includes the freshly transferred prefill cache. Only the
    # prefill's initial reuse measures a hit from an earlier request.
    reuse_field = "prefill_total_reuse_len"

    def start_servers(self):
        gpu_ids = list(get_gpu_ids())
        self.assertGreaterEqual(len(gpu_ids), 4)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="v41_contract",
            use_local=True,
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{prefill_port}",
                        protocol="http",
                        path="/",
                    ),
                    decode_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{decode_port}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
        )
        common_env = {
            "LOAD_PYTHON_MODEL": "1",
            "LOG_LEVEL": "DEBUG",
            "FT_SERVER_TEST": "1",
            "DSV4_MOE_STRATEGY": "mega",
            "DSV4_USE_MEGA_MOE_SE": "0",
            "DSV4_USE_MEGA_MOE_FUSED": "0",
            "DSV41_CED": "1",
            "DSV41_SWA_BOUNDED_REPLAY": "1",
            "WORLD_SIZE": "2",
            "REMOTE_RPC_SERVER_IP": "localhost",
        }
        # The macro reserves four cards; each role owns two disjoint cards.
        common_args = os.environ["SMOKE_ARGS"].replace(
            "--world_size 4", "--world_size 2"
        )
        for role, port, peer, devices, args in (
            (
                "prefill",
                prefill_port,
                decode_port,
                gpu_ids[:2],
                "--role_type PREFILL --enable_cuda_graph 0 --cp_rotate_method ALL_GATHER",
            ),
            (
                "decode",
                decode_port,
                prefill_port,
                gpu_ids[2:4],
                "--role_type DECODE --enable_cuda_graph 1 --decode_capture_config 1,2,4 --cp_rotate_method PREFILL_CP",
            ),
        ):
            manager = MagaServerManager(
                env_args={
                    **common_env,
                    "REMOTE_SERVER_PORT": peer,
                    "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                },
                device_ids=devices,
                port=port,
                role_name=f"deepseek_v41_{role}_contract",
                smoke_args_str=f"{common_args} {args}",
            )
            self.addCleanup(manager.stop_server)
            if role == "prefill":
                self.server = manager
            else:
                self.graph_server = manager
        # Use the same independent server lifecycle and endpoint routing as
        # PdSeperationCaseRunner. Register cleanup before parallel startup so
        # failures and assertion errors always stop both processes.
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=self.model_path,
                    model_type="deepseek_v41",
                    timeout=3600,
                )
                for manager in (self.server, self.graph_server)
            ]
            for result in starts:
                self.assertTrue(result.result())

    def _long_prefix_reuse_checks(self, prompt):
        # Fresh SWA uses BF16 while a reused prefix uses MXFP8-dequantized KV.
        # Independent cache-format/attention references validate those values;
        # bounded replay does not promise cross-path token equality. Each path
        # must reproduce its complete output and observed generation logits.
        baseline = self.request("long_uncached", prompt, False)
        self.assertEqual(baseline["aux_info"][self.reuse_field], 0)
        cold = self.request("long_cache_fill", prompt, True)
        self.assert_same(baseline, cold)
        self.assertEqual(cold["aux_info"][self.reuse_field], 0)
        repeated = self.request("long_cache_hit", prompt, True)
        second = self.request("long_second_cache_hit", prompt, True)
        self.assertGreaterEqual(repeated["aux_info"][self.reuse_field], 256)
        self.assertEqual(
            repeated["aux_info"][self.reuse_field],
            second["aux_info"][self.reuse_field],
        )
        self.assert_same(repeated, second)
        bypass = self.request("populated_cache_bypass", prompt, False)
        self.assert_same(baseline, bypass)
        self.assertEqual(bypass["aux_info"][self.reuse_field], 0)
        self.assert_decode_graph_replayed_and_record_coverage()

    def test_graph_engram_and_prefix_reuse(self):
        super().test_graph_engram_and_prefix_reuse()
        prefill_log = Path(self.server.log_file_path).read_text(errors="replace")
        compactions = re.findall(
            r"V4.1 CED compacted: bounded=True cp=2 rows=(\d+)->(\d+)",
            prefill_log,
        )
        self.assertTrue(
            any(int(before) > int(after) for before, after in compactions),
            "No actual bounded CED row reduction observed on CP2 prefill",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
