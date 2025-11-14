import logging
import multiprocessing
import os
import random
import socket
import time
import unittest
from multiprocessing import Process
from typing import List
from unittest import mock

import requests
import torch

torch.cuda.set_device = lambda x: None

from rtp_llm.distribute.gang_info import get_c2_members, get_gang_info
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info
from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM, GangConfig
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.frontend.frontend_server import FrontendWorker
from rtp_llm.start_backend_server import main


def fake_init(self, *args, **kwargs):
    self.model_config = None
    self.tokenizer = None
    self.model_cls = None
    self.pipeline = None
    self.backend_rpc_server_visitor = None


FrontendWorker.__init__ = fake_init
OpenaiEndpoint.__init__ = fake_init


class GangTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super().__init__(*args, **kwargs)

    def get_self_ip(self):
        return socket.gethostbyname(socket.gethostname())

    @mock.patch.dict(
        "os.environ",
        {"GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation"},
    )
    def test_annocation(self):
        gang_annocation_path = os.environ.get("GANG_ANNOCATION_PATH", "rtp_llm/distribute/test/testdata/annocation")
        gang_members = get_c2_members(gang_annocation_path)
        self.assertEqual(len(gang_members), 2)
        self.assertEqual(gang_members[0].name, "llama_7b_a10_part2_new_inference_part0")
        self.assertEqual(gang_members[0].ip, "33.115.125.211")
        self.assertEqual(gang_members[1].name, "llama_7b_a10_part2_new_inference_part1")
        self.assertEqual(gang_members[1].ip, "33.115.37.164")

    @mock.patch.dict(
        "os.environ",
        {
            "GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation",
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
        },
    )
    def test_multi_gpu_gang_info(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        gang_config = GangConfig()
        gang_info = get_gang_info(start_port=g_worker_info.rpc_server_port, gang_config=gang_config)
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, self.get_self_ip())
        self.assertEqual(gang_info.members[0].name, "local_0")
        self.assertEqual(
            gang_info.members[0].server_port, WorkerInfo.server_port_offset(0)
        )
        self.assertEqual(gang_info.members[1].ip, self.get_self_ip())
        self.assertEqual(gang_info.members[1].name, "local_1")
        self.assertEqual(
            gang_info.members[1].server_port, WorkerInfo.server_port_offset(1)
        )

    @mock.patch.dict(
        "os.environ",
        {
            "GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation",
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
        },
    )
    def test_multi_worker_gang_info(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        gang_annocation_path = os.environ.get("GANG_ANNOCATION_PATH", "rtp_llm/distribute/test/testdata/annocation")
        gang_info = get_gang_info(gang_annocation_path=gang_annocation_path)
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, "33.115.125.211")
        self.assertEqual(
            gang_info.members[0].name, "llama_7b_a10_part2_new_inference_part0_0"
        )
        self.assertEqual(
            gang_info.members[0].server_port, WorkerInfo.server_port_offset(0)
        )
        self.assertEqual(gang_info.members[1].ip, "33.115.37.164")
        self.assertEqual(
            gang_info.members[1].name, "llama_7b_a10_part2_new_inference_part1_0"
        )
        self.assertEqual(
            gang_info.members[1].server_port, WorkerInfo.server_port_offset(0)
        )

    @mock.patch.dict(
        "os.environ",
        {
            "GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation",
            "TP_SIZE": "2",
            "PP_SIZE": "2",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
        },
    )
    def test_multi_worker_gpu_gang_info(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        gang_annocation_path = os.environ.get("GANG_ANNOCATION_PATH", "rtp_llm/distribute/test/testdata/annocation")
        gang_config = GangConfig()
        gang_config.gang_annocation_path = gang_annocation_path
        gang_info = get_gang_info(start_port=g_worker_info.rpc_server_port, gang_config=gang_config)
        self.assertEqual(len(gang_info.members), 4)
        self.assertEqual(gang_info.members[0].ip, "33.115.125.211")
        self.assertEqual(
            gang_info.members[0].name, "llama_7b_a10_part2_new_inference_part0_0"
        )
        self.assertEqual(
            gang_info.members[0].server_port, WorkerInfo.server_port_offset(0)
        )

        self.assertEqual(gang_info.members[1].ip, "33.115.125.211")
        self.assertEqual(
            gang_info.members[1].name, "llama_7b_a10_part2_new_inference_part0_1"
        )
        self.assertEqual(
            gang_info.members[1].server_port, WorkerInfo.server_port_offset(1)
        )

        self.assertEqual(gang_info.members[2].ip, "33.115.37.164")
        self.assertEqual(
            gang_info.members[2].name, "llama_7b_a10_part2_new_inference_part1_0"
        )
        self.assertEqual(
            gang_info.members[2].server_port, WorkerInfo.server_port_offset(0)
        )

        self.assertEqual(gang_info.members[3].ip, "33.115.37.164")
        self.assertEqual(
            gang_info.members[3].name, "llama_7b_a10_part2_new_inference_part1_1"
        )
        self.assertEqual(
            gang_info.members[3].server_port, WorkerInfo.server_port_offset(1)
        )

    @mock.patch.dict(
        "os.environ",
        {
            "DISTRIBUTE_CONFIG_FILE": "rtp_llm/distribute/test/testdata/parallel.json",
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
        },
    )
    def test_multi_worker_gang_info_from_json(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        distribute_config_file = os.environ.get("DISTRIBUTE_CONFIG_FILE", "rtp_llm/distribute/test/testdata/parallel.json")
        gang_config = GangConfig()
        gang_config.distribute_config_file = distribute_config_file
        gang_info = get_gang_info(start_port=g_worker_info.rpc_server_port, gang_config=gang_config)
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, "11.161.48.116")
        self.assertEqual(
            gang_info.members[0].name, "llama13B_2A10_PCIE_1_inference_part0_0"
        )
        self.assertEqual(gang_info.members[0].server_port, 10000)

        self.assertEqual(gang_info.members[1].ip, "11.161.48.116")
        self.assertEqual(
            gang_info.members[1].name, "llama13B_2A10_PCIE_1_inference_part1_0"
        )
        self.assertEqual(gang_info.members[1].server_port, 20000)

    @mock.patch("torch.cuda.device_count")
    @mock.patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "START_PORT": str(random.randint(10000, 40000)),
            "GANG_SLEEP_TIME": "1",
            "FAKE_GANG_ENV": "1",
            "MODEL_TYPE": "fake_model",
            "TOKENIZER_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/tokenizer"
            ),
            "CHECKPOINT_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/cpt"
            ),
            "DIST_BARRIER_TIMEOUT": "10",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
    )
    def test_server_start(self, torch_device_count):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError as e:
            logging.warn(str(e))

        torch_device_count.return_value = 2
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        procs: List[Process] = list()
        procs = main()

        try:
            while True:
                try:
                    for i in range(0, int(os.environ["WORLD_SIZE"])):
                        gang_hb_port = WorkerInfo.gang_hb_port_offset(i)
                        hb_response = requests.post(
                            f"http://localhost:{gang_hb_port}/heartbeat",
                            json={"name": "fake_name", "ip": "fake_ip"},
                            timeout=5,
                        )
                        self.assertEqual(hb_response.json()["initializing"], False)
                    break
                except:
                    time.sleep(1)

            # test gang heartbeat loss will cause other process terminate
            if torch_device_count.return_value > 1:
                procs[0].terminate()
                time.sleep(10)
                for proc in procs:
                    self.assertTrue(proc.is_alive() == False)
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()

    @mock.patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "PP_SIZE": "2",
            "WORLD_SIZE": "4",
            "WORLD_INDEX": "1",
            "LOCAL_WORLD_SIZE": "2",
        },
    )
    def test_get_world_rank_from_world_index(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        self.assertEqual(g_parallel_info.world_rank, 2)

    @mock.patch.dict(
        "os.environ",
        {
            "LEADER_ADDRESS": "33.115.125.211",
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "ZONE_NAME": "prefill",
            "WORLD_SIZE": "2",
            "WORLD_INDEX": "1",
            "LOCAL_WORLD_SIZE": "1",
        },
    )
    def test_multi_worker_gang_info_from_leader(self):
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        leader_address = os.environ.get("LEADER_ADDRESS", "33.115.125.211")
        zone_name = os.environ.get("ZONE_NAME", "prefill")
        gang_config = GangConfig()
        gang_config.leader_address = leader_address
        gang_config.zone_name = zone_name
        gang_info = get_gang_info(start_port=g_worker_info.rpc_server_port, gang_config=gang_config)
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, "33.115.125.211")
        self.assertEqual(gang_info.members[0].name, "prefill_part0_0")
        self.assertEqual(
            gang_info.members[0].server_port, WorkerInfo.server_port_offset(0)
        )
        self.assertEqual(gang_info.members[1].ip, self.get_self_ip())
        self.assertEqual(gang_info.members[1].name, "prefill_part1_0")
        self.assertEqual(
            gang_info.members[1].server_port, WorkerInfo.server_port_offset(0)
        )

    @mock.patch("torch.cuda.device_count")
    @mock.patch.dict(
        "os.environ",
        {
            "LEADER_ADDRESS": socket.gethostbyname(socket.gethostname()),
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_INDEX": "0",
            "LOCAL_WORLD_SIZE": "1",
            "START_PORT": str(random.randint(10000, 40000)),
            "GANG_SLEEP_TIME": "1",
            "FAKE_GANG_ENV": "1",
            "MODEL_TYPE": "fake_model",
            "TOKENIZER_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/tokenizer"
            ),
            "CHECKPOINT_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/cpt"
            ),
            "DIST_BARRIER_TIMEOUT": "10",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
    )
    def test_server_start_leader(self, torch_device_count):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError as e:
            logging.warn(str(e))

        torch_device_count.return_value = 2
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        procs: List[Process] = list()
        procs = main()

        try:
            gang_hb_port = WorkerInfo.gang_hb_port_offset(0)
            while True:
                try:
                    hb_response = requests.post(
                        f"http://localhost:{gang_hb_port}/heartbeat",
                        json={"name": "part1_0", "ip": self.get_self_ip()},
                        timeout=5,
                    )
                    self.assertEqual(hb_response.json()["initializing"], True)
                    break
                except:
                    time.sleep(1)

            time.sleep(2)
            gang_hb_port = WorkerInfo.gang_hb_port_offset(0)
            hb_response = requests.post(
                f"http://localhost:{gang_hb_port}/heartbeat",
                json={"name": "part1_0", "ip": self.get_self_ip()},
                timeout=5,
            )
            self.assertEqual(hb_response.json()["initializing"], False)

        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()

    @mock.patch("torch.cuda.device_count")
    @mock.patch.dict(
        "os.environ",
        {
            "LEADER_ADDRESS": "127.0.0.1",
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_INDEX": "1",
            "LOCAL_WORLD_SIZE": "1",
            "START_PORT": str(random.randint(10000, 40000)),
            "GANG_SLEEP_TIME": "1",
            "FAKE_GANG_ENV": "1",
            "MODEL_TYPE": "fake_model",
            "TOKENIZER_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/tokenizer"
            ),
            "CHECKPOINT_PATH": os.path.join(
                os.getcwd(), "rtp_llm/distribute/test/testdata/cpt"
            ),
            "DIST_BARRIER_TIMEOUT": "10",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
    )
    def test_server_start_worker(self, torch_device_count):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError as e:
            logging.warn(str(e))

        torch_device_count.return_value = 2
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", str(MIN_WORKER_INFO_PORT_NUM)))
        g_parallel_info.reload(worker_info_port_num)
        procs: List[Process] = list()
        procs = main()

        try:
            while True:
                try:
                    gang_hb_port = WorkerInfo.gang_hb_port_offset(0)
                    hb_response = requests.post(
                        f"http://localhost:{gang_hb_port}/heartbeat",
                        json={"name": "part1_0", "ip": self.get_self_ip()},
                        timeout=5,
                    )
                    self.assertEqual(hb_response.json()["initializing"], True)
                    hb_response = requests.post(
                        f"http://localhost:{gang_hb_port}/broadcast_parts",
                        json={
                            "part0": {"name": "part0", "ip": "127.0.0.1"},
                            "part1": {"name": "part1", "ip": self.get_self_ip()},
                            "part2": {"name": "part2", "ip": self.get_self_ip()},
                            "part3": {"name": "part3", "ip": self.get_self_ip()},
                        },
                        timeout=5,
                    )
                    break
                except:
                    time.sleep(1)

            time.sleep(2)
            gang_hb_port = WorkerInfo.gang_hb_port_offset(0)
            hb_response = requests.post(
                f"http://localhost:{gang_hb_port}/heartbeat",
                json={"name": "part1_0", "ip": self.get_self_ip()},
                timeout=5,
            )
            self.assertEqual(hb_response.json()["initializing"], False)

        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()


if __name__ == "__main__":
    unittest.main()
