import os
from typing import Any
from unittest import TestCase, main

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


class AllFakeModelTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(cur_path, "testdata/qwen2_tokenizer")

    def test_simple(self):
        # test load success from bad tokenizer file
        tokenizer = Qwen2Tokenizer.from_pretrained(self.data_path)
        # test special tokens
        res = tokenizer.encode("<|im_start|>hello<|im_end|>")
        self.assertEqual(res, [151644, 14990, 151645])


if __name__ == "__main__":
    main()
