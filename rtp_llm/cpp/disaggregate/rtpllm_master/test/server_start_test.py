import multiprocessing
import os
import subprocess
from multiprocessing import Process
from unittest import TestCase, main

from fake_server import FakeServer

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

from master.librtpllm_master import MasterInitParameter, RtpLLMMasterEntry


class MasterStartTest(TestCase):
    def _start_fake_sever(self, port: int):
        def _start_fake_sever_func():
            server = FakeServer()
            server.start(port)

        multiprocessing.set_start_method("spawn")
        proc = Process(target=_start_fake_sever_func)
        proc.start()
        return proc

    def _start_master(self):
        args = ["/opt/conda310/bin/python", "-m", "entry"]
        args.extend(["--local_port", "10000"])
        args.extend(["--port", "20000"])
        args.extend(["--model_size", "7"])
        args.extend(
            ["--force_replace_data_dir", str(os.path.join(CUR_PATH, "testdata"))]
        )
        p = subprocess.Popen(args)
        return p

    def test_simple(self):
        fake_server_proc = None
        master_proc = None
        try:
            fake_server_proc = self._start_fake_sever(12345)
            master_proc = self._start_master()
        finally:
            if fake_server_proc:
                fake_server_proc.terminate()
            if master_proc:
                master_proc.terminate()


if __name__ == "__main__":
    main()
