import os
import time
import socket
import random
import torch
import unittest
from contextlib import closing
from unittest import TestCase, main, mock
from concurrent.futures import ThreadPoolExecutor

from rtp_llm.utils.weight_type import WEIGHT_TYPE
from rtp_llm.test.model_test.test_util.fake_model_loader import FakeModelLoader
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.distribute.worker_info import update_master_info, g_worker_info
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller

os.environ['KV_CACHE_MEM_MB'] = '100'
@mock.patch.dict('os.environ', {'RESERVER_RUNTIME_MEM_MB': '1', 'DEVICE_RESERVE_MEMORY_BYTES': str(64 * 1024 * 1024)})
class RpcModelTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_path = os.path.join(os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.tokenizer_path = "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"
        self.ckpt_path = "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"
        
    def create_pipeline(self, max_seq_len: int = 100):
        set_global_controller(init_controller())
        ports, _ = PortManager().get_consecutive_ports(1)
        free_port = ports[0]
        os.environ['START_PORT'] = str(free_port)
        update_master_info("", free_port)
        g_worker_info.reload()
        self.fake_model_loader = FakeModelLoader(model_type='qwen_2',
                                                 tokenizer_path=self.tokenizer_path,
                                                 ckpt_path=self.ckpt_path,
                                                 weight_type=WEIGHT_TYPE.FP16,
                                                 max_seq_len=max_seq_len)
        model = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.config, model.tokenizer)
        return pipeline

    def test_simple(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(10)
            def func():
                [_ for _ in pipeline("hello, what's your name?", max_new_tokens=10)]
            result = []
            for i in range(0, 10):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 10):
                result[i].result()
        finally:
            pipeline.stop()

if __name__ == '__main__':
    main()
