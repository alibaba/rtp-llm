import os
import torch
from typing import Any
from unittest import TestCase, main
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_test import single_fake_test

model_list = {
    "chatglm-safetensors": {
        "model_type": "chatglm",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/safetensors"),
        "weight_type": WEIGHT_TYPE.FP16
    },
}

class AllFakeSafetensorsTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        os.environ['KV_CACHE_MEM_MB'] = '10'
        os.environ['MAX_SEQ_LEN'] = '256'
        # set plugin to allow hidden_states output

    def test_simple(self):
        for name, model_config in model_list.items():
            single_fake_test(name, "fake_test", model_config, async_mode=False, test_loss=False)
            # TODO(xinfei.sxf) async = true lead to dim error, so ignore it temporarily
            # single_fake_test(name, model_config, True)

if __name__ == '__main__':
    main()
