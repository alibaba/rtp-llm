import os
import logging
import logging.config
import sys
from unittest import TestCase, main

from maga_transformer.models.llama import Llama, LlamaTokenizer

class LlamaTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/models/test/testdata/llama/')


    def test_load_llama_tokenizer(self):
        model_path = os.path.join(LlamaTest._testdata_path(), "fake")
        config = Llama._create_config(model_path)
        config.lora_infos = None
        config.tokenizer_path=model_path
        model = Llama(config)
        model.load_tokenizer()
        self.assertEqual([1, 29871, 29900, 29900, 29900], model.tokenizer.encode("000"))
        self.assertTrue(isinstance(model.tokenizer, LlamaTokenizer))
        
    def test_load_aquila_tokenizer(self):
        model_path = os.path.join(LlamaTest._testdata_path(), "AquilaChat2-7B")
        config = Llama._create_config(model_path)
        config.lora_infos = None
        config.tokenizer_path=model_path
        model = Llama(config)
        model.load_tokenizer()
        self.assertEqual([1457], model.tokenizer.encode("000"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
