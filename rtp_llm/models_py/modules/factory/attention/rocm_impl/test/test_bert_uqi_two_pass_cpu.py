import unittest
import torch
from bert_uqi_test_utils import check_cases, cpu_varlen


class TestBertUqiCPU(unittest.TestCase):
    def test_explicit_mask_and_profile_isolation(self):
        check_cases(self, "cpu", torch.float32, cpu_varlen)

if __name__ == "__main__":
    unittest.main()
