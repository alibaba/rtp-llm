import os
import logging
import logging.config
import sys
from unittest import TestCase, main

from maga_transformer.models.starcoder import *

class StarCodeTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/models/test/testdata/starcode/')
        
    def test_config_from_hf(self):
        config = StarCoder._create_config(StarCodeTest._testdata_path())
        self.assertEqual(16384, config.inter_size)
        self.assertEqual(49152, config.vocab_size)
        self.assertEqual(32, config.head_num)
        self.assertEqual(128, config.size_per_head)
        self.assertEqual(4096, config.hidden_size)
        self.assertEqual(42, config.layer_num)
        self.assertEqual(1, config.head_num_kv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
