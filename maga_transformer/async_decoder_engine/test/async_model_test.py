import os
import torch
from unittest import TestCase, main
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.config.exceptions import FtRuntimeException
from maga_transformer.pipeline.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

class MockMemInfo:
    free: int  = 2 * 1024 * 1024 # byte
    used: int  = 0

@mock.patch('maga_transformer.config.cache_config.get_mem_info', MockMemInfo)
@mock.patch.dict('os.environ', {'RESERVER_RUNTIME_MEM_MB': '1'})
class AsyncDecoderModelTest(TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")
        self.ckpt_path = os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source")

    def create_pipeline(self, max_seq_len: int = 100):
        self.fake_model_loader = FakeModelLoader(model_type='llama',
                                                 tokenizer_path=self.tokenizer_path,
                                                 ckpt_path=self.ckpt_path,
                                                 weight_type=WEIGHT_TYPE.FP16,
                                                 max_seq_len=max_seq_len)
        model: AsyncModel = self.fake_model_loader.load_model()
        pipeline = Pipeline(model, model.tokenizer)
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
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.decoder_engine_.stop()

    def test_max_new_tokens_error(self) -> None:
        pipeline = self.create_pipeline(10)
        try:
            gen = pipeline("hello, what's your name?\nI'm a 20 year old girl from the Netherlands.")
            with self.assertRaisesRegex(FtRuntimeException, "model max tokens is "):
                [_ for _ in gen]
            self.assertFalse(pipeline.model.decoder_engine_.scheduler_.have_streams())
        finally:
            pipeline.model.decoder_engine_.stop()

if __name__ == '__main__':
    main()
