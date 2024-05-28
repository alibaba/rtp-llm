
import os
import torch
from unittest import TestCase, main
from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.models.starcoder import StarcoderTokenizer
from maga_transformer.ops import SpecialTokens

class ChatapiTest(TestCase):
    def _get_tokenizer(self):
        tokenizer_path = os.path.join(os.getcwd(), 'maga_transformer/test/model_test/fake_test/testdata/starcoder/tokenizer')
        tokenizer = StarcoderTokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def test_simple(self):
        tokenizer = self._get_tokenizer()
        special_tokens = SpecialTokens()
        special_tokens.user.token_ids = [1]
        special_tokens.assistant.eos_token_ids = [2]
        prompt = [
            {'role':'system',
             'content':'system'},
            {'role':'user',
             'content':'user_'},
            {'role':'assistant',
             'content':'ass_'},
        ]
        
        self.assertEqual(tokenizer.encode('system') + \
                         [1] + tokenizer.encode('user_') + [] + \
                         [] + tokenizer.encode('ass_') + [2],
                         encode_chatapi(prompt, special_tokens, tokenizer))

    def test_bos(self):
        tokenizer = self._get_tokenizer()
        special_tokens = SpecialTokens()
        special_tokens.bos_token_id = 5
        special_tokens.user.token_ids = [1]
        special_tokens.assistant.eos_token_ids = [2]
        prompt = [
            {'role':'system',
             'content':'system'},
            {'role':'user',
             'content':'user_'},
            {'role':'assistant',
             'content':'ass_'},
        ]
        self.assertEqual([5] + tokenizer.encode('system') + \
                         [1] + tokenizer.encode('user_') + [] + \
                         [] + tokenizer.encode('ass_') + [2],
                         encode_chatapi(prompt, special_tokens, tokenizer))

    def test_no_system(self):
        tokenizer = self._get_tokenizer()
        special_tokens = SpecialTokens()
        special_tokens.user.token_ids = [1]
        special_tokens.assistant.eos_token_ids = [2]
        prompt = [
            {'role':'user',
             'content':'user_'},
            {'role':'assistant',
             'content':'ass_'},
        ]
        self.assertEqual([1] + tokenizer.encode('user_') + [] + \
                         [] + tokenizer.encode('ass_') + [2],
                         encode_chatapi(prompt, special_tokens, tokenizer))

    def test_multi_round(self):
        tokenizer = self._get_tokenizer()
        special_tokens = SpecialTokens()
        special_tokens.user.token_ids = [1]
        special_tokens.user.eos_token_ids = [2]
        special_tokens.assistant.token_ids = [3]
        special_tokens.assistant.eos_token_ids = [4]
        prompt = [
            {'role':'user',
             'content':'user1'},
            {'role':'assistant',
             'content':'ass1'},
            {'role':'user',
             'content':'user2'},
            {'role':'assistant',
             'content':'ass2'},
            {'role':'user',
             'content':'user3'},
        ]
        self.assertEqual([1] + tokenizer.encode('user1') + [2] + \
                         [3] + tokenizer.encode('ass1') + [4] + \
                         [1] + tokenizer.encode('user2') + [2] + \
                         [3] + tokenizer.encode('ass2') + [4] + \
                         [1] + tokenizer.encode('user3') + [2, 3],
                         encode_chatapi(prompt, special_tokens, tokenizer))

if __name__ == '__main__':
    main()
