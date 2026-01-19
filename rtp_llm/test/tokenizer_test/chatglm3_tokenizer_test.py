from typing import Any
from unittest import TestCase, main

from rtp_llm.tokenizer_factory.tokenizers import ChatGLMV3Tokenizer


class AllFakeModelTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_path = "rtp_llm/test/tokenizer_test/testdata/chatglm3_tokenizer"

    def test_simple(self):
        # test load success from bad tokenizer file
        tokenizer = ChatGLMV3Tokenizer(self.data_path, {})
        # test special tokens
        res = tokenizer.encode("<|assistant|>")
        self.assertEqual(res, [64790, 64792, 64796])


if __name__ == "__main__":
    main()
