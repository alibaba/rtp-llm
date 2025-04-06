import os
import asyncio
import logging
from unittest import TestCase, main

from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.server.frontend_worker import FrontendWorker, BatchPipelineResponse
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.structure.request_extractor import request_id_field_name
from maga_transformer.distribute.worker_info import DEFAULT_START_PORT, update_master_info, g_worker_info

from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader
from maga_transformer.test.utils.port_util import get_consecutive_free_ports

class FakeFrontendWorker(FrontendWorker):
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline

class FrontendWorkerTest(TestCase):
    def setUp(self):
        os.environ['KV_CACHE_MEM_MB'] = '100'
        os.environ['RESERVER_RUNTIME_MEM_MB'] = '1'
        os.environ['DEVICE_RESERVE_MEMORY_BYTES'] = str(64 * 1024 * 1024)
        self.tokenizer_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.frontend_worker = self.create_frontend_worker()

    def tearDown(self):
        self.frontend_worker.stop()

    def create_frontend_worker(self):
        port_list = get_consecutive_free_ports(1)
        os.environ['START_PORT'] = str(port_list[0])
        update_master_info('0.0.0.0', int(port_list[0]))
        g_worker_info.reload()
        self.fake_model_loader = FakeModelLoader(model_type='llama',
                                                 tokenizer_path=self.tokenizer_path,
                                                 ckpt_path=self.ckpt_path,
                                                 weight_type=WEIGHT_TYPE.FP16,
                                                 max_seq_len=2048)
        model: AsyncModel = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.config, model.tokenizer)
        return FakeFrontendWorker(model, pipeline)

    async def _run(self, frontend_worker, **kwargs):
        count = 0
        kwargs[request_id_field_name] = 1
        gen = frontend_worker.inference(**kwargs)
        result = []
        aux_info = []
        finished = []
        async for response in gen:
            if isinstance(response, BatchPipelineResponse):
                self.assertTrue("prompt_batch" in kwargs)
                logging.info(f"batch stream reponse: {response.response_batch}")
            else:
                logging.info(f"stream response:{response.response}")

        # collect log response
        log_response = await gen.gen_complete_response_once()
        if isinstance(log_response, BatchPipelineResponse):
            result = [_.response for _ in log_response.response_batch]
            aux_info = [_.aux_info for _ in log_response.response_batch]
            finished = [_.finished for _ in log_response.response_batch]
        else:
            result = log_response.response
            aux_info = log_response.aux_info
            finished = log_response.finished
        return result, aux_info, finished

    def test_simple(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker, prompt="please write a story about dog", generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1}))
        # just ensure every input has result
        result_text, aux_info, finished = func()
        logging.info(f"result_text: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertTrue(len(result_text) > 0)

    def test_text_input(self):
        result_text, aux_info, finished = asyncio.run(self._run(self.frontend_worker, text="please write a story about dog", generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1}))
        # logging.info(f"test text input : {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertTrue(len(result_text) > 0)

    def test_num_batch(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker,
                                         prompt_batch=["please write a story about dog",
                                                       "please write a story about dog"],
                                         yield_generator=True,
                                         generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True, "num_return_sequences": 3}))
        result_text, aux_info, finished = func()
        logging.info(f"batch * num: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(2, len(result_text))
        self.assertEqual(3, len(result_text[0]))
        self.assertEqual(3, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(3, len(aux_info[0]))
        self.assertEqual(3, len(aux_info[1]))
        self.assertEqual([True, True], finished)
        self.assertTrue(len(result_text) > 0)

    def test_num_return_sequences_1(self):
        result_text, aux_info, finished = asyncio.run(self._run(self.frontend_worker,
                                                                prompt="please write a story about dog",
                                                                yield_generator=True,
                                                                generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True, "num_return_sequences": 1}))
        # logging.info(f"no batch * num_return_sequences 1 : {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(1, len(result_text))
        self.assertEqual(1, len(aux_info))
        self.assertEqual(True, finished)

    def test_batch_num_return_sequences_1(self):
        result_text, aux_info, finished = asyncio.run(self._run(self.frontend_worker,
                                                                prompt_batch=["please write a story about dog",
                                                                              "please write a story about dog"],
                                                                yield_generator=True,
                                                                generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True, "num_return_sequences": 1}))
        # logging.info(f"batch 2 * num_return_sequences 1: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(2, len(result_text))
        self.assertEqual(1, len(result_text[0]))
        self.assertEqual(1, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(1, len(aux_info[0]))
        self.assertEqual(1, len(aux_info[1]))
        self.assertEqual([True, True], finished)

    def test_incremental(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker,
                                         prompt="please write a story about dog",
                                         yield_generator=True,
                                         generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True}))
        result = []
        result_text, aux_info, finished = func()
        self.assertEqual(3, aux_info.get("iter_count"))
        self.assertEqual(True, finished)
        logging.info(f"incremental : {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertTrue(len(result_text) > 0)

    def test_batch_incremental(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker,
                                         prompt_batch=["please write a story about dog",
                                                       "please write a story about dog"],
                                         yield_generator=True,
                                         generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True}))
        result_text, aux_info, finished = func()
        self.assertEqual(2, len(aux_info))
        self.assertEqual([True, True], finished)
        logging.info(f"batch incremental: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertTrue(len(result_text) > 0)


    def test_num_return_incremental(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker,
                                         prompt="please write a story about dog",
                                         yield_generator=True,
                                         generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True, "num_return_sequences": 3}))
        result_text, aux_info, finished = func()
        logging.info(f"incremental num return: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(3, len(result_text))
        self.assertEqual(3, len(aux_info))
        self.assertEqual(True, finished)
        self.assertTrue(len(result_text) > 0)

    def test_batch_num_return_incremental(self):
        def func():
            return asyncio.run(self._run(self.frontend_worker,
                                         prompt_batch=["please write a story about dog",
                                                       "please write a story about dog"],
                                         yield_generator=True,
                                         generate_config={"top_k": 1, "max_new_tokens": 3, "top_p": 1, "return_incremental": True, "num_return_sequences": 3}))

        result_text, aux_info, finished = func()
        logging.info(f"incremental batch * num return: {result_text}, aux_info: {aux_info}, finished:{finished}")
        self.assertEqual(2, len(result_text))
        self.assertEqual(3, len(result_text[0]))
        self.assertEqual(3, len(result_text[1]))
        self.assertEqual(2, len(aux_info))
        self.assertEqual(3, len(aux_info[0]))
        self.assertEqual(3, len(aux_info[1]))
        self.assertEqual([True, True], finished)
        self.assertTrue(len(result_text) > 0)

    def test_encode(self):
        token_ids, tokens = self.frontend_worker.tokenizer_encode('a b c')
        self.assertEqual(token_ids, [1, 263, 289, 274])
        self.assertEqual(tokens, ['<s>', 'a', 'b', 'c'])

if __name__ == '__main__':
    if os.environ.get('FT_SERVER_TEST', None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()
