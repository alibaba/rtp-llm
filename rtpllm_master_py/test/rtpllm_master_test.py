import json
import time
import logging
import requests
from typing import List, Any, Optional
from unittest import TestCase, main
from fake_server import FakeServer
import multiprocessing
from multiprocessing import Process
from io import TextIOWrapper
import subprocess
import os

from maga_transformer.test.utils.port_util import get_consecutive_free_ports
from concurrent.futures import ThreadPoolExecutor

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

def _start_fake_sever_func(port: int, running_task_list: List[Any]):
    server = FakeServer(running_task_list)
    server.start(port)

class ServerProcess(object):
    def __init__(self, process: subprocess.Popen, port: int, file_stream: TextIOWrapper, log_file_path: str):
        self.process = process
        self.port = port
        self.file_stream = file_stream
        self.log_file_path = log_file_path

    def stop(self):
        self.process.terminate()
        self.file_stream.close()

class MasterStartTest(TestCase):
    master_proc: Optional[ServerProcess] = None
    worker_port = -1
    worker_process: Optional[Process] = None
 
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.worker_port = get_consecutive_free_ports(1)[0]
        cls.worker_process = cls._start_fake_sever(cls.worker_port, [])
        cls.master_proc = cls._start_master(cls.worker_port)
        try:
            cls._check_server_start(cls.master_proc.port)
        except:
            with open(cls.master_proc.log_file_path) as f:
                content = f.read()
            logging.warning("----------------server output--------------------")
            logging.warning(f"{content}")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.worker_process:
            cls.worker_process.terminate()
        if cls.master_proc:
            cls.master_proc.stop()

    @staticmethod
    def _start_fake_sever(port: int, running_task_list: List[Any]):
        multiprocessing.set_start_method("spawn")
        proc = Process(target=_start_fake_sever_func, args=(port, running_task_list, ))
        proc.start()
        return proc

    @staticmethod
    def _start_master(worker_port: int):
        server_log_path = os.path.join(os.getcwd(), 'server')
        os.system(f"rm -rf {server_log_path}")
        os.makedirs(server_log_path, exist_ok=True)        
        log_file_path = os.path.join(server_log_path, "process.log")

        random_port = get_consecutive_free_ports(1)[0]
        args = ["/opt/conda310/bin/python", "-m", "rtpllm_master_py.entry"]
        args.extend(["--local_port", str(worker_port)])
        args.extend(["--port", str(random_port)])
        args.extend(["--use_local", "1"])
        args.extend(["--model_size", "7"])
        args.extend(["--model_type", "FAKE_MODEL"])
        args.extend(["--force_replace_data_dir", str(os.path.join(CUR_PATH, "testdata"))])
        f = open(log_file_path, 'w')        
        p = subprocess.Popen(args, stdout=f, stderr=f)
        logging.info(f"master log write in file: {log_file_path}")
        return ServerProcess(p, random_port, f, log_file_path)

    @staticmethod
    def _check_server_start(port: int, timeout: int = 10):
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                 raise Exception(f"failed to check server health in timeout=={timeout}")
            try:
                ret = requests.get(f"http://localhost:{port}/health", timeout=1)
                if ret.status_code == 200:
                    break
            except Exception as e:
                logging.warning(f"start server failed with error: {str(e)}")
            time.sleep(1)
        # wait 3s for master to sync worker status
        time.sleep(3)

    def _check_get_ip_success(self, master_port: int, worker_port: int):
        ret = requests.post(f"http://localhost:{master_port}", json={"prompt": "hello"})
        print("response: ", ret.status_code, ret.text)
        self.assertEqual(ret.status_code, 200)        
        response = ret.json()
        self.assertEqual(response['ip'], "127.0.0.1")
        self.assertEqual(response['port'], worker_port)
        self.assertEqual(response['master_info']['expect_execute_time_ms'], 50)
        self.assertEqual(response['master_info']['expect_wait_time_ms'], 0)
        self.assertEqual(response['master_info']['input_length'], 5)
        self.assertTrue(response['master_info']['request_id'] != "")

        ret = requests.post(f"http://localhost:{master_port}", json={"prompt": "hello"})
        print("response: ", ret.status_code, ret.text)
        self.assertEqual(ret.status_code, 200)        
        response = ret.json()
        self.assertEqual(response['ip'], "127.0.0.1")
        self.assertEqual(response['port'], worker_port)
        self.assertEqual(response['master_info']['expect_execute_time_ms'], 50)
        self.assertEqual(response['master_info']['expect_wait_time_ms'], 50)
        self.assertEqual(response['master_info']['input_length'], 5)
        self.assertTrue(response['master_info']['request_id'] != "")

    def test_all(self):
        self._test_simple()
        self._test_concurrent_access()

    def _test_simple(self):
        self.assertTrue(self.master_proc is not None)
        self._check_get_ip_success(self.master_proc.port, self.worker_port)

    def _test_concurrent_access(self):
        t = ThreadPoolExecutor(100)
        def curl():
            ret = requests.post(f"http://localhost:{self.master_proc.port}", json={"prompt": "hello"})
            self.assertEqual(ret.status_code, 200)        
        ress = []
        for i in range(100):
            ress.append(t.submit(curl))
        for i in range(100):
            ress[i].result()

if __name__ == '__main__':
    
    main()