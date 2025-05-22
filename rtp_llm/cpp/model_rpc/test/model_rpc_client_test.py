import os
import logging
import asyncio
import unittest
from unittest import TestCase, main
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient

import torch
from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.cpp.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import GenerateOutputPB
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.proto.model_rpc_service_pb2 import TensorPB


class FakeStub:

    async def generate_stream(self, input: GenerateInputPB):
        res = GenerateOutputPB()
        output_ids = res.output_ids
        output_ids.data_type = TensorPB.DataType.INT32
        output_ids.shape.extend([1, 0])
        res.aux_info.iter_count = 1
        for i in range(2):
            res.aux_info.iter_count += 1
            res.aux_info.output_len += 1
            output_ids.shape[1] += 1
            output_ids.data_int32.append(i)
            yield res
        res.finished = True
        yield res


class FakeModelRpcClient(ModelRpcClient):

    def __init__(self):
        self.stub = FakeStub()


class ModelRpcClientTest(TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # self.client = FakeModelRpcClient()

    @staticmethod
    async def _run(client, input):
        responses = []
        async for res in client.enqueue(input):
            responses.append(res)
        return responses

    @unittest.skip("need fix")
    def test_generate_stream(self):
        client = FakeModelRpcClient()
        generate_config: GenerateConfig = GenerateConfig(
            using_hf_sampling=False)
        input = GenerateInput(token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                              generate_config=generate_config)
        res = asyncio.run(self._run(client, input))
        self.assertEqual(len(res), 3)
        self.assertEqual(list(res[0].output_ids.shape), [1, 1])
        self.assertEqual(res[0].output_ids.tolist(), [[0]])
        self.assertEqual(res[0].finished, False)
        self.assertEqual(res[0].aux_info.iter_count, 2)
        self.assertEqual(res[0].aux_info.output_len, 1)

        self.assertEqual(list(res[1].output_ids.shape), [1, 2])
        self.assertEqual(res[1].output_ids.tolist(), [[0, 1]])
        self.assertEqual(res[1].finished, False)
        self.assertEqual(res[1].aux_info.iter_count, 3)
        self.assertEqual(res[1].aux_info.output_len, 2)

        self.assertEqual(res[2].finished, True)


if __name__ == '__main__':
    if os.environ.get('FT_SERVER_TEST', None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()
