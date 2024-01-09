from typing import Any
from unittest import TestCase, main
from maga_transformer.tokenizer.tokenization_chatglm3 import ChatGLMTokenizer

class AllFakeModelTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_path = "maga_transformer/test/tokenizer_test/testdata/chatglm3_tokenizer"

    def test_simple(self):
        # test load success from bad tokenizer file
        tokenizer = ChatGLMTokenizer.from_pretrained(self.data_path, encode_special_tokens=True)
        # test special tokens
        res = tokenizer.encode("<|assistant|>")
        self.assertEqual(res, [64790, 64792, 64796])
        
if __name__ == '__main__':
    main()