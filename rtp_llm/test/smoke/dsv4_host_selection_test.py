"""Four-card DeepSeek-V4 P/D smoke after whole-host decode rank selection."""

import json
import os
import re
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import takewhile
from pathlib import Path

import grpc
import requests

from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortsContext


def local_slots(base_port):
    slots = []
    for rank in range(2):
        server = ServerConfig()
        server.start_port = base_port
        server.set_local_rank(rank)
        slots.append(server)
    return slots


def endpoint(ports):
    return EndPoint(
        type="Vipserver",
        address=",".join(f"127.0.0.1:{port}" for port in ports),
        protocol="http",
        path="/",
    )


class DSV4HostSelectionTest(unittest.TestCase):
    def test_decode_host_exclusion_and_pd_handoff(self):
        gpu_ids = list(get_gpu_ids())
        self.assertGreaterEqual(len(set(gpu_ids[:4])), 4, f"Need four GPUs: {gpu_ids}")
        decode_devices, prefill_devices = gpu_ids[:2], gpu_ids[2:4]
        fixture = Path(__file__).parent / os.environ["DSV4_SMOKE_FIXTURE"]
        model = json.loads(fixture.read_text())
        # Wildcard listeners on the same physical host need separate port ranges.
        excluded_base, active_base, prefill_base = (
            PortsContext.reserve_for_test(self, 200)[100] for _ in range(3)
        )
        active_slots = local_slots(active_base)
        route = ServiceRoute(
            service_id="test",
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=endpoint([prefill_base]),
                    decode_endpoint=endpoint(
                        [slot.server_port for slot in active_slots]
                    ),
                )
            ],
            use_local=True,
        )
        common = {
            "LOCAL_WORLD_SIZE": "2",
            "FRONTEND_SERVER_COUNT": "1",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            "REMOTE_RPC_SERVER_IP": "localhost",
        }
        decode_env = dict(
            common,
            WORLD_SIZE="4",
            ENABLE_STARTUP_RANK_SELECTION="1",
            FT_SERVER_TEST="1",  # Capture the engine configs in process.log.
            GANG_CONFIG_STRING=f"name:test_part0,ip:127.0.0.1,port:{excluded_base}",
            REMOTE_SERVER_PORT=str(prefill_base),
        )
        excluded = MagaServerManager(
            role_name="decode_excluded",
            port=str(excluded_base),
            device_ids=prefill_devices,
            env_args=dict(decode_env, WORLD_RANK="0", STARTUP_HEALTHY_GPU_IDS="0"),
            smoke_args_str=os.environ["SMOKE_ARGS"],
            health_check_path="/liveness",
        )
        active = MagaServerManager(
            role_name="decode_active",
            port=str(active_base),
            device_ids=decode_devices,
            env_args=dict(decode_env, WORLD_RANK="2", STARTUP_HEALTHY_GPU_IDS="0,1"),
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        prefill = MagaServerManager(
            role_name="prefill",
            port=str(prefill_base),
            device_ids=prefill_devices,
            env_args=dict(
                common,
                WORLD_SIZE="2",
                WORLD_RANK="0",
                ENABLE_STARTUP_RANK_SELECTION="0",
                REMOTE_SERVER_PORT=str(active_base),
            ),
            smoke_args_str=os.environ["PREFILL_SMOKE_ARGS"],
        )
        for manager in (excluded, active, prefill):
            self.addCleanup(manager.stop_server)
        self.http = requests.Session()
        self.http.trust_env = False
        self.addCleanup(self.http.close)

        def start(manager):
            self.assertTrue(
                manager.start_server(
                    model["model_path"], model["model_type"], timeout=3600
                ),
                f"Startup failed; see {manager.log_file_path}",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [executor.submit(start, manager) for manager in (excluded, active)]
            try:
                for ready in as_completed(starts):
                    ready.result()
            except BaseException:
                for manager in (prefill, excluded, active):
                    manager.stop_server()
                raise

        self.assert_decode_workers(active_slots)
        self.assert_engine_configs(active.log_file_path)
        print(
            f"Decode GPUs {decode_devices}: original ranks 2,3 -> 0,1; "
            "WORLD=LOCAL_WORLD=DP=EP=2, TP=1; both workers alive",
            flush=True,
        )

        self.assert_excluded_probe(excluded_base, "/liveness", 200)
        self.assert_excluded_probe(excluded_base, "/health", 500)

        start(prefill)
        for slot in local_slots(prefill_base):
            with grpc.insecure_channel(f"127.0.0.1:{slot.rpc_server_port}") as channel:
                grpc.channel_ready_future(channel).result(timeout=120)
        print(f"Prefill ready on excluded GPUs {prefill_devices}", flush=True)
        request = {
            "prompt": "Write the integers from 1 to 20, in order, separated by commas.",
            "generate_config": {
                "max_new_tokens": 16,
                "min_new_tokens": 16,
                "top_k": 1,
                "can_use_pd_separation": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        response = self.http.post(
            f"http://127.0.0.1:{active_base}/", json=request, timeout=600
        )
        self.assertEqual(response.status_code, 200, response.text)
        actual = response.json()
        self.assertNotIn("error_code", actual)
        self.assertIs(actual["finished"], True, actual)
        self.assertIs(actual["aux_info"]["pd_sep"], True, actual)
        self.assertEqual(len(actual["output_ids"][0]), 16, actual)
        self.assert_excluded_probe(excluded_base, "/liveness", 200)
        print(
            f"PD response: HTTP {response.status_code}, finished={actual['finished']}, "
            f"pd_sep={actual['aux_info']['pd_sep']}, output_ids={actual['output_ids'][0]}",
            flush=True,
        )

    def assert_excluded_probe(self, base_port, path, expected_status):
        for slot in local_slots(base_port):
            url = f"http://127.0.0.1:{slot.server_port}{path}"
            probe = self.http.get(url, allow_redirects=False, timeout=5)
            self.assertEqual(probe.status_code, expected_status, f"{url}: {probe.text}")

    def assert_decode_workers(self, slots):
        # Each real backend reports its configured DP/TP and rank via ModelRPC.
        # A frontend can return partial status while a peer starts listening.
        expected = {rank: slot.rpc_server_port for rank, slot in enumerate(slots)}
        for slot in slots:
            url = f"http://127.0.0.1:{slot.server_port}/worker_status"
            deadline = time.monotonic() + 120
            while True:
                try:
                    response = self.http.get(url, timeout=5)
                    if response.ok:
                        workers = response.json()["results"]
                        if len(workers) == len(expected):
                            break
                    last_status = response.text
                except requests.RequestException as error:
                    last_status = str(error)
                if time.monotonic() >= deadline:
                    self.fail(f"Incomplete decode topology at {url}: {last_status}")
                time.sleep(1)
            self.assertEqual(
                {
                    int(w["dp_rank"]): int(w["address"].rsplit(":", 1)[1])
                    for w in workers
                },
                expected,
            )
            for worker in workers:
                self.assertEqual(worker["role"], "RoleType.DECODE")
                self.assertEqual((worker["dp_size"], worker["tp_size"]), (2, 1))
                self.assertIs(worker["alive"], True)

    def assert_engine_configs(self, log_file):
        # /worker_status omits EP and local/world sizes. Check the config actually
        # passed to each C++ engine at initialization.
        seen = set()
        with open(log_file, errors="replace") as stream:
            for line in stream:
                if line.strip() != "[ParallelismConfig]":
                    continue
                block = "".join(takewhile(str.strip, stream))
                config = dict(re.findall(r"^(\w+): (\d+)$", block, re.MULTILINE))
                rank = int(config["world_rank"])
                for key, size in dict(
                    world_size=2, local_world_size=2, dp_size=2, ep_size=2, tp_size=1
                ).items():
                    self.assertEqual(int(config[key]), size, (rank, key, config))
                for key in ("local_rank", "ep_rank", "dp_rank"):
                    self.assertEqual(int(config[key]), rank, (key, config))
                seen.add(rank)
        self.assertEqual(seen, {0, 1}, log_file)


if __name__ == "__main__":
    unittest.main()
