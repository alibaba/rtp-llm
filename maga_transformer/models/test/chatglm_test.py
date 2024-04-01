import os
import logging
import logging.config
from unittest import TestCase, main

from maga_transformer.models.chat_glm import ChatGlm
from maga_transformer.utils.model_weight import W

class ChatGlmTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/models/test/testdata/chatglm/')

    def _tokenizer_path():
        return os.path.join(os.getcwd(), 'maga_transformer/test/model_test/fake_test/testdata/chatglm/tokenizer')
        
    def test_load_aquila_tokenizer(self):
        model_path = os.path.join(ChatGlmTest._testdata_path(), "4bit")
        logging.info(model_path)
        config = ChatGlm._create_config(model_path)
        config.ckpt_path = model_path
        config.lora_infos = None
        config.tokenizer_path=ChatGlmTest._tokenizer_path()
        model = ChatGlm(config)
        self.assertEqual([128], list(model.weight.weights[1][W.post_ln_beta].shape))
        self.assertEqual([128, 384],  list(model.weight.weights[1][W.attn_qkv_w].shape))
        self.assertAlmostEqual(-0.05078125, model.weight.weights[1][W.attn_qkv_w][0][0].item())
        self.assertAlmostEqual(0.006118, model.weight.weights[1][W.attn_qkv_b][0].item(), delta = 0.0001)
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
