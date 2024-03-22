import os
import torch
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, main
from unittest import mock
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.server.inference_worker import InferenceWorker, BatchPipelineResponse
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader

class MockMemInfo:
    free: int  = 12 * 1024 * 1024 # byte
    used: int  = 0

class FakeInferenceWorker(InferenceWorker):
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline


@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
@mock.patch.dict('os.environ', {'RESERVER_RUNTIME_MEM_MB': '2'})
class InferenceWorkerTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")

    def create_inference_worker(self):
        self.fake_model_loader = FakeModelLoader(model_type='llama',
                                                 tokenizer_path=self.tokenizer_path,
                                                 ckpt_path=self.ckpt_path,
                                                 weight_type=WEIGHT_TYPE.FP16,
                                                 max_seq_len=2048)
        model: AsyncModel = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.tokenizer)
        return FakeInferenceWorker(model, pipeline)

    async def _run(self, inference_worker, **kwargs):
        count = 0
        gen = inference_worker.inference(**kwargs)
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
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker, prompt="please write a story about dog", generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                logging.info(f"{i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_text_input(self):
        inference_worker = self.create_inference_worker()
        try:
            result_text, aux_info, finished = asyncio.run(self._run(inference_worker, text="please write a story about dog", generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1}))
            # logging.info(f"test text input : {result_text}, aux_info: {aux_info}, finished:{finished}")
            expect_text = "ProductsProductsProducts"
            self.assertEqual(expect_text, result_text)
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_num_batch(self):
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker,
                                             prompt_batch=["please write a story about dog", "please write a story about dog"],
                                             yield_generator=True,
                                             generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True, "num_return_sequences":3}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                logging.info(f"batch * num {i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                self.assertEqual(2, len(result_text))
                self.assertEqual(3, len(result_text[0]))
                self.assertEqual(3, len(result_text[1]))
                self.assertEqual(2, len(aux_info))
                self.assertEqual(3, len(aux_info[0]))
                self.assertEqual(3, len(aux_info[1]))
                self.assertEqual([True, True], finished)
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_num_return_sequences_1(self):
        inference_worker = self.create_inference_worker()
        try:
            result_text, aux_info, finished = asyncio.run(self._run(inference_worker,
                                                                    prompt="please write a story about dog",
                                                                    yield_generator=True,
                                                                    generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True, "num_return_sequences": 1}))
            # logging.info(f"no batch * num_return_sequences 1 : {result_text}, aux_info: {aux_info}, finished:{finished}")
            self.assertEqual(1, len(result_text))
            self.assertEqual(1, len(aux_info))
            self.assertEqual(True, finished)
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_batch_num_return_sequences_1(self):
        inference_worker = self.create_inference_worker()
        try:
            result_text, aux_info, finished = asyncio.run(self._run(inference_worker,
                                                                    prompt_batch=["please write a story about dog", "please write a story about dog"],
                                                                    yield_generator=True,
                                                                    generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True, "num_return_sequences": 1}))
            # logging.info(f"batch 2 * num_return_sequences 1: {result_text}, aux_info: {aux_info}, finished:{finished}")
            self.assertEqual(2, len(result_text))
            self.assertEqual(1, len(result_text[0]))
            self.assertEqual(1, len(result_text[1]))
            self.assertEqual(2, len(aux_info))
            self.assertEqual(1, len(aux_info[0]))
            self.assertEqual(1, len(aux_info[1]))
            self.assertEqual([True, True], finished)
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_incremental(self):
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker,
                                                            prompt="please write a story about dog",
                                                            yield_generator=True,
                                                            generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                self.assertEqual(3, aux_info.get("iter_count"))
                self.assertEqual(True, finished)
                logging.info(f"incremental {i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_batch_incremental(self):
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker,
                                                            prompt_batch=["please write a story about dog", "please write a story about dog"],
                                                            yield_generator=True,
                                                            generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                self.assertEqual(2, len(aux_info))
                self.assertEqual([True, True], finished)
                logging.info(f"batch incremental {i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()


    def test_num_return_incremental(self):
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker,
                                             prompt="please write a story about dog",
                                             yield_generator=True,
                                             generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True, "num_return_sequences":3}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                logging.info(f"incremental num return {i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                self.assertEqual(3, len(result_text))
                self.assertEqual(3, len(aux_info))
                self.assertEqual(True, finished)
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_batch_num_return_incremental(self):
        inference_worker = self.create_inference_worker()
        try:
            thread_count = 10
            t = ThreadPoolExecutor(thread_count)
            def func():
                return asyncio.run(self._run(inference_worker,
                                             prompt_batch=["please write a story about dog", "please write a story about dog"],
                                             yield_generator=True,
                                             generate_config={"top_k":1, "max_new_tokens":3, "top_p": 1, "return_incremental": True, "num_return_sequences":3}))
            result = []
            for i in range(0, thread_count):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, thread_count):
                result_text, aux_info, finished = result[i].result()
                expect_text = "ProductsProductsProducts"
                logging.info(f"incremental batch * num return {i} th : {result_text}, aux_info: {aux_info}, finished:{finished}")
                self.assertEqual(2, len(result_text))
                self.assertEqual(3, len(result_text[0]))
                self.assertEqual(3, len(result_text[1]))
                self.assertEqual(2, len(aux_info))
                self.assertEqual(3, len(aux_info[0]))
                self.assertEqual(3, len(aux_info[1]))
                self.assertEqual([True, True], finished)
                # self.assertEqual(result_text, expect_text, f"{i} th result is not same: expect is :[{expect_text}] actual is :[{result_text}]")
            self.assertFalse(inference_worker.pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            inference_worker.stop()

    def test_encode(self):
        inference_worker = self.create_inference_worker()
        token_ids, tokens = inference_worker.tokenizer_encode('a b c')
        self.assertEqual(token_ids, [1, 263, 289, 274])
        self.assertEqual(tokens, ['<s>', 'a', 'b', 'c'])

if __name__ == '__main__':
    if os.environ.get('FT_SERVER_TEST', None) is None:
        logging.config.dictConfig(LOGGING_CONFIG)
    main()
