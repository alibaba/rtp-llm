import os
import logging
import logging.config
import sys
from unittest import TestCase, main

import torch

from maga_transformer.utils.model_weight import b_half_merge


print(os.getcwd())
print('PYTHONPATH=' + os.environ['PYTHONPATH'] + ' LD_LIBRARY_PATH=' + os.environ['LD_LIBRARY_PATH'] + ' ' + sys.executable + ' ')

os.environ['FT_SERVER_TEST'] = "1"

class ModelWeightsTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/utils/test/testdata/model_weights_testdata/')

    def test_b_half_merge(self):   
        a =  torch.rand(12).cuda()
        m = b_half_merge([a]).cuda()
        self.assertTrue(torch.equal(a, m))

        a = torch.tensor([11, 12, 13, 14]).cuda()
        b = torch.tensor([21, 22, 23, 24]).cuda()
        m = b_half_merge([a, b])
        e_m = torch.tensor([11, 12, 21, 22, 13, 14, 23, 24]).cuda()
        print(m)
        self.assertTrue(torch.equal(e_m, m))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
