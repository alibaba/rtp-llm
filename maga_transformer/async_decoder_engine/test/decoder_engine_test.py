import os
import torch
from unittest import TestCase, main
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.pipeline.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
class MockMemInfo:
    free: int  = 24 * 1024 * 1024 # byte
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
                                             async_mode=True)
        model: AsyncModel = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.tokenizer)
        return pipeline

    def test_simple(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(10)
            def func():
                gen = pipeline(["hello, what's your name?"], [[]], max_new_tokens=10)
                results = [result for result in gen]

            result = []
            for i in range(0, 10):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 10):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.has_query())
        finally:
            pipeline.model.stop()

    def test_timeout(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(10)
            gen = pipeline(["hello, what's your name?"], [[]], max_new_tokens=100, timeout_ms=10)
            with self.assertRaisesRegex(Exception, "ms timeout"):
                results = [result for result in gen]
            self.assertTrue(pipeline.model.decoder_engine_.scheduler_.has_query())
        finally:
            pipeline.model.stop()

    def test_batch(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(10)
            def func():
                gen = pipeline(["hello, what's your name?", "please write a story about dog", "hi", "what"], [[], [], [], []], max_new_tokens=10)
                results = [result for result in gen]
            result = []
            for i in range(0, 10):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 10):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.has_query())
        finally:
            pipeline.model.stop()

    def test_stress(self) -> None:
        pipeline = self.create_pipeline()
        try:
            t = ThreadPoolExecutor(100)
            def func():
                try:
                    gen = pipeline(["hello, what's your name?", "please write a story about dog"], [[], []],max_new_tokens=64)
                    results = [result for result in gen]
                except:
                    pass
            result = []
            for i in range(0, 100):
                result.append(t.submit(func))
            # just ensure every input has result
            for i in range(0, 100):
                result[i].result()
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.has_query())
        finally:
            pipeline.model.stop()

    @mock.patch('maga_transformer.models.base_model.BaseModel.create_context_decoder_mask')
    def test_error_internal(self, create_context_decder_mask) -> None:
        pipeline = self.create_pipeline()
        try:
            create_context_decder_mask.side_effect = Exception("test exception")
            with self.assertRaisesRegex(Exception, "test exception"):
                gen = pipeline(["你好", "hello, what's your name?"], [[], []])
                results = [result for result in gen]
            # just ensure every input has result
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.has_query())
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
                gen = pipeline(["你好", "hello, what's your name?"], [[], []])
                results = [result for result in gen]
            # just ensure every input has result
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.has_query())
        finally:
            pipeline.model.stop()

    @mock.patch('maga_transformer.async_decoder_engine.batch_query.BatchQuery.generate')
    def test_free_query_when_generate_batch_query_error(self, generate) -> None:
        generate.side_effect = Exception("test exception")
        pipeline = self.create_pipeline()
        try:
            origin_block = len(pipeline.model.decoder_engine_.scheduler_.cache_manager_.free_blocks_index)
            gen = pipeline(["hello, what's your name?"], [[]], max_new_tokens=10)
            with self.assertRaisesRegex(Exception, "test exception"):
                results = [result for result in gen]
            remain_block = len(pipeline.model.decoder_engine_.scheduler_.cache_manager_.free_blocks_index)
            self.assertEqual(origin_block, remain_block)
        finally:
            pipeline.model.stop()

if __name__ == '__main__':
    main()
