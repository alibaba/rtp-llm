import os
import logging
import logging.config
import sys
from unittest import TestCase, main

from maga_transformer.models.qwen import *

class QwenTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/models/test/testdata/qwen/')
        
    def test_config_from_hf(self):
        config = QWen_7B._create_config(QwenTest._testdata_path())
        self.assertEqual(11008, config.inter_size)
        self.assertEqual(303872, config.vocab_size)
        self.assertEqual(32, config.head_num)
        self.assertEqual(128, config.size_per_head)
        self.assertEqual(32, config.layer_num)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
