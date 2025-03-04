import os
import time
import socket
import random
import torch
from contextlib import closing
import unittest
from unittest import TestCase, main
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader
from maga_transformer.test.utils.port_util import get_consecutive_free_ports
from maga_transformer.config.exceptions import FtRuntimeException
from maga_transformer.pipeline.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
from maga_transformer.distribute.worker_info import update_master_info, g_worker_info
from maga_transformer.utils.concurrency_controller import init_controller, set_global_controller

from unittest import mock

os.environ['KV_CACHE_MEM_MB'] = '100'
@mock.patch.dict('os.environ', {'RESERVER_RUNTIME_MEM_MB': '1', 'DEVICE_RESERVE_MEMORY_BYTES': str(64 * 1024 * 1024)})
class RpcModelTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")

    def create_pipeline(self, max_seq_len: int = 100):
        set_global_controller(init_controller())
        free_port = get_consecutive_free_ports(1)[0]
        os.environ['START_PORT'] = str(free_port)
        update_master_info("", free_port)
        g_worker_info.reload()
        self.fake_model_loader = FakeModelLoader(model_type='llama',
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
