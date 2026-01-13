import os
from unittest import TestCase, main

from rtp_llm.frontend.tokenizer_factory.tokenizers import ChatGLMV3Tokenizer


from pytest import mark
@mark.A10
@mark.cuda
@mark.gpu
class AllFakeModelTest(TestCase):
    def setUp(self) -> None:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(cur_path, 'testdata/chatglm3_tokenizer')

    def test_simple(self):
        # test load success from bad tokenizer file
        tokenizer = ChatGLMV3Tokenizer(self.data_path, {})
        # test special tokens
        res = tokenizer.encode("<|assistant|>")
        self.assertEqual(res, [64790, 64792, 64796])


if __name__ == "__main__":
    main()
