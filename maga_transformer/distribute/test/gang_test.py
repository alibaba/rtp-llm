import os
import time
import random
import socket
import signal
import logging
import unittest
import requests
from unittest import mock
from threading import Thread
import multiprocessing
from multiprocessing import Process
from typing import List
from maga_transformer.distribute.gang_info import get_c2_members
from maga_transformer.distribute.gang_info import get_gang_info
from maga_transformer.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info
from maga_transformer.distribute.test.fake_model import FakeModel
from maga_transformer.start_server import main

class GangTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super().__init__(*args, **kwargs)

    def get_self_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def test_annocation(self):
        os.environ['GANG_ANNOCATION_PATH'] = "maga_transformer/distribute/test/testdata/annocation"
        gang_members = get_c2_members()
        self.assertEqual(len(gang_members), 2)
        self.assertEqual(gang_members[0].name, 'llama_7b_a10_part2_new_inference_part0')
        self.assertEqual(gang_members[0].ip, "33.115.125.211")
        self.assertEqual(gang_members[1].name, 'llama_7b_a10_part2_new_inference_part1')
        self.assertEqual(gang_members[1].ip, "33.115.37.164")

    def test_multi_gpu_gang_info(self):
        os.environ['TP_SIZE'] = '2'
        os.environ['PP_SIZE'] = '1'
        os.environ['WORLD_SIZE'] = '2'
        os.environ['WORLD_RANK'] = '0'
        os.environ['LOCAL_WORLD_SIZE'] = '2'
        os.environ['GANG_ANNOCATION_PATH'] = "maga_transformer/distribute/test/testdata/annocation"
        g_parallel_info.reload()
        gang_info = get_gang_info()
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, self.get_self_ip())
        self.assertEqual(gang_info.members[0].name, 'local_0')
        self.assertEqual(gang_info.members[0].server_port, WorkerInfo.server_port_offset(0))
        self.assertEqual(gang_info.members[1].ip, self.get_self_ip())
        self.assertEqual(gang_info.members[1].name, 'local_1')
        self.assertEqual(gang_info.members[1].server_port, WorkerInfo.server_port_offset(1))

    def test_multi_worker_gang_info(self):
        os.environ['TP_SIZE'] = '2'
        os.environ['PP_SIZE'] = '1'
        os.environ['WORLD_SIZE'] = '2'
        os.environ['WORLD_RANK'] = '0'
        os.environ['LOCAL_WORLD_SIZE'] = '1'
        os.environ['GANG_ANNOCATION_PATH'] = "maga_transformer/distribute/test/testdata/annocation"
        g_parallel_info.reload()
        gang_info = get_gang_info()
        self.assertEqual(len(gang_info.members), 2)
        self.assertEqual(gang_info.members[0].ip, '33.115.125.211')
        self.assertEqual(gang_info.members[0].name, 'llama_7b_a10_part2_new_inference_part0_0')
        self.assertEqual(gang_info.members[0].server_port, WorkerInfo.server_port_offset(0))
        self.assertEqual(gang_info.members[1].ip, '33.115.37.164')
        self.assertEqual(gang_info.members[1].name, 'llama_7b_a10_part2_new_inference_part1_0')
        self.assertEqual(gang_info.members[1].server_port, WorkerInfo.server_port_offset(0))

    def test_multi_worker_gpu_gang_info(self):
        os.environ['TP_SIZE'] = '2'
        os.environ['PP_SIZE'] = '2'
        os.environ['WORLD_SIZE'] = '4'
        os.environ['WORLD_RANK'] = '0'
        os.environ['LOCAL_WORLD_SIZE'] = '2'
        os.environ['GANG_ANNOCATION_PATH'] = "maga_transformer/distribute/test/testdata/annocation"
        g_parallel_info.reload()
        gang_info = get_gang_info()
        self.assertEqual(len(gang_info.members), 4)
        self.assertEqual(gang_info.members[0].ip, '33.115.125.211')
        self.assertEqual(gang_info.members[0].name, 'llama_7b_a10_part2_new_inference_part0_0')
        self.assertEqual(gang_info.members[0].server_port, WorkerInfo.server_port_offset(0))

        self.assertEqual(gang_info.members[1].ip, '33.115.125.211')
        self.assertEqual(gang_info.members[1].name, 'llama_7b_a10_part2_new_inference_part0_1')
        self.assertEqual(gang_info.members[1].server_port, WorkerInfo.server_port_offset(1))

        self.assertEqual(gang_info.members[2].ip, '33.115.37.164')
        self.assertEqual(gang_info.members[2].name, 'llama_7b_a10_part2_new_inference_part1_0')
        self.assertEqual(gang_info.members[2].server_port, WorkerInfo.server_port_offset(0))

        self.assertEqual(gang_info.members[3].ip, '33.115.37.164')
        self.assertEqual(gang_info.members[3].name, 'llama_7b_a10_part2_new_inference_part1_1')
        self.assertEqual(gang_info.members[3].server_port, WorkerInfo.server_port_offset(1))

    @mock.patch('torch.cuda.device_count')
    def test_server_start(self, torch_device_count):
        os.environ['WORLD_SIZE'] = '4'
        os.environ['TP_SIZE'] = '4'
        os.environ['PP_SIZE'] = '1'
        os.environ['START_PORT'] = str(random.randint(10000, 40000))
        os.environ['GANG_SLEEP_TIME'] = '1'
        os.environ['FAKE_GANG_ENV'] = '1'
        os.environ['MODEL_TYPE'] = "fake_model"
        os.environ['TOKENIZER_PATH'] = os.path.join(os.getcwd(), "maga_transformer/distribute/test/testdata/tokenizer")
        os.environ['CHECKPOINT_PATH'] = os.path.join(os.getcwd(), "maga_transformer/distribute/test/testdata/cpt")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        torch_device_count.return_value = 4
        g_parallel_info.reload()

        procs: List[Process] = list()
        if torch_device_count.return_value == 1:
            proc = multiprocessing.Process(target=main)
            proc.start()
            procs.append(proc)
        else:
            procs = main()

        time.sleep(30)

        try:
            for i in range(0, int(os.environ['WORLD_SIZE'])):
                gang_hb_port = WorkerInfo.gang_hb_port_offset(i)
                hb_response = requests.post(f"http://localhost:{gang_hb_port}/heartbeat", json={"name": 'fake_name', "ip": 'fake_ip'}, timeout=5)
                self.assertEqual(hb_response.json()['initializing'], False)

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

if __name__ == '__main__':
    unittest.main()
