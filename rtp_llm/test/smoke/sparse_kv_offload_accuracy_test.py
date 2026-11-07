"""Greedy-token and full-vocabulary logit parity with real Host KV reads."""

import logging
import unittest

from rtp_llm.test.smoke import sparse_kv_offload_test


class SparseKVOffloadAccuracyTest(unittest.TestCase):
    _compare_model = sparse_kv_offload_test.SparseKVOffloadModelTest._compare_model

    def test_glm_fp8_accuracy(self):
        self._compare_model("GLM-5-FP8-4layer", "glm_5", True, accuracy=True)

    def test_deepseek_v32_accuracy(self):
        self._compare_model("DeepSeek-V3.2-4layer", "deepseek_v32", True, accuracy=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
