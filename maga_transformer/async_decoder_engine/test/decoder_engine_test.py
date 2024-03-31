import os
import torch
import asyncio
import logging
import threading
import time
from unittest import TestCase, main
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.pipeline.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
class MockMemInfo:
    free: int  = 12 * 1024 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
@mock.patch.dict('os.environ', {'RESERVER_RUNTIME_MEM_MB': '2'})
class DecoderEngineTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")

    def create_pipeline(self):
        self.fake_model_loader = FakeModelLoader(model_type='llama',
                                                 tokenizer_path=self.tokenizer_path,
                                                 ckpt_path=self.ckpt_path,
                                                 weight_type=WEIGHT_TYPE.FP16,
                                                 max_seq_len=2048)
        model: AsyncModel = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.tokenizer)
        return pipeline

    def test_simple(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(10)
            def func():
                gen = pipeline("hello, what's your name?", max_new_tokens=10)
                results = [result for result in gen]

            result = []
            for i in range(0, 10):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 10):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    def test_timeout(self) -> None:
        pipeline = self.create_pipeline()
        try:
            gen = pipeline("what's your name?", max_new_tokens=100, timeout_ms=10)
            with self.assertRaisesRegex(Exception, "it's timeout"):
                results = [result for result in gen]
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    def test_cancel_request(self) -> None:
        pipeline = self.create_pipeline()
        origin_block = pipeline.model.decoder_engine_.scheduler_._stream_cache_manager.cache_manager_.free_block_nums
        
        async def _simplified_generator():
            try:
                # 假设 self.pipeline_async 是一个异步生成器
                async for x in pipeline.pipeline_async("please write a story about dog", max_new_tokens=1000, top_k = 1):
                    yield x
            except Exception as e:
                raise e
            
        async def _run():
            count = 0
            last_element = ""
            gen =  _simplified_generator()
            async for result in gen:
                # 处理每个生成的结果
                if count == 3:
                    await gen.aclose()
                    break
                else:
                    last_element = result.generate_texts
                    logging.info(f"{count}: {last_element}")
                count = count + 1
            self.assertEqual(last_element, ['ProductsProductsProducts'])
            await asyncio.sleep(0.01)
            
        try:
            asyncio.run(_run())
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
            remain_block = pipeline.model.decoder_engine_.scheduler_._stream_cache_manager.cache_manager_.free_block_nums
            self.assertEqual(origin_block, remain_block)
        finally:
            pipeline.model.stop()

    def test_stress(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(32)
            def func():
                [_ for _ in pipeline("please write a story about dog", max_new_tokens=32)]
            result = []
            for i in range(0, 64):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 64):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    def test_guarante_generate(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(32)
            def func():
                [_ for _ in pipeline(" ".join("hello, what's your name?" * 80), max_new_tokens=64)]
            result = []
            for i in range(0, 32):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 32):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    @mock.patch('maga_transformer.async_decoder_engine.normal_model_executor.NormalModelExecutor._process')
    def test_error_internal(self, process) -> None:
        pipeline = self.create_pipeline()
        try:
            process.side_effect = Exception("test exception")
            with self.assertRaisesRegex(Exception, "test exception"):
                [_ for _ in pipeline("hello, what's your name?")]
            # just ensure every input has result
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    @mock.patch('maga_transformer.async_decoder_engine.cache_manager.CacheManager.malloc')
    @mock.patch('maga_transformer.async_decoder_engine.cache_manager.CacheManager.malloc_with_cache')
    def test_failed_to_malloc_block(self, malloc, malloc_with_cache) -> None:
        pipeline = self.create_pipeline()
        try:
            malloc.side_effect = Exception("test exception")
            malloc_with_cache.side_effect = Exception("test exception")
            with self.assertRaisesRegex(Exception, "test exception"):
                gen = pipeline("hello, what's your name?")
                results = [result for result in gen]
            # just ensure every input has result
            time.sleep(0.01)
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.stop()

    @mock.patch('maga_transformer.async_decoder_engine.batch_query.BatchQuery.generate_model_input')
    def test_free_query_when_generate_batch_query_error(self, generate_model_input) -> None:
        generate_model_input.side_effect = Exception("test exception")
        pipeline = self.create_pipeline()
        try:
            origin_block = pipeline.model.decoder_engine_.scheduler_._stream_cache_manager.cache_manager_.free_block_nums
            gen = pipeline("hello, what's your name?")
            with self.assertRaisesRegex(Exception, "test exception"):
                results = [result for result in gen]
            remain_block = pipeline.model.decoder_engine_.scheduler_._stream_cache_manager.cache_manager_.free_block_nums
            self.assertEqual(origin_block, remain_block)
        finally:
            pipeline.model.stop()

if __name__ == '__main__':
    main()
