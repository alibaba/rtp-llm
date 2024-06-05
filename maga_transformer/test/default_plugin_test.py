
from maga_transformer.pipeline.default_plugin import DefaultPlugin
from maga_transformer.utils.tokenizer_utils import DecodingState, IncrementDecodingUtils

from unittest import TestCase, main
from typing import Any, List

class FakeTokenizer(object):
    def decode(self, ids: List[int]):
        return " ".join([str(x) for x in ids])

class DefaultPluginTest(TestCase):
    def test_incremental_decode(self):
        tokenizer = FakeTokenizer()
        state = DecodingState()
        output_ids = [5, 9, 11, 13, 15, 17, 19, 21]
        res = []
        for i in range(1, len(output_ids) + 1):
            ids = output_ids[:i]
            text, all_text = DefaultPlugin.tokenids_decode_func(ids, tokenizer, state, True)
            res.append(text)
        self.assertEqual(res, ['5', ' 9', ' 11', ' 13', ' 15', ' 17', ' 19', ' 21'])


if __name__ == '__main__':
    main()
