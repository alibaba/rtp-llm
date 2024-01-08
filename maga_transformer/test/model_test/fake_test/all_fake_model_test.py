import os
from typing import Any
from unittest import TestCase, main
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.test.model_test.test_util.fake_model_test import single_fake_test

model_list = {
    "chatglm": {
        "model_type": "chatglm",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/fake"),
        "weight_type": WEIGHT_TYPE.FP16
    },
    "chatglm-ptuning": {
        "model_type": "chatglm",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm/ptuning"),
        "weight_type": WEIGHT_TYPE.FP16
    },
    "chatglm2": {
        "model_type": "chatglm2",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm2/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/chatglm2/fake"),
        "weight_type": WEIGHT_TYPE.FP16
    },
    "llama": {
        "model_type": "llama",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/llama/fake/hf_source"),
        "weight_type": WEIGHT_TYPE.FP16
    },
    "bloom": {
        "model_type": "bloom",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/bloom/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/bloom/fake/"),
        "weight_type": WEIGHT_TYPE.FP16
    },
    "qwen": {
        "model_type": "qwen_7b",
        "tokenizer_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/qwen_7b/tokenizer"),
        "ckpt_path": os.path.join(os.getcwd(), "maga_transformer/test/model_test/fake_test/testdata/qwen_7b/fake/"),
        "weight_type": WEIGHT_TYPE.FP16
    }
}

loss_model_list = {
}

class AllFakeModelTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        os.environ['KV_CACHE_MEM_MB'] = '1'
        # set plugin to allow hidden_states output
    def test_simple(self):
        for name, model_config in model_list.items():
            single_fake_test(name, "fake_test", model_config, async_mode=False, test_loss=False)
            single_fake_test(name, "fake_test", model_config, async_mode=True, test_loss=False)

        for name, model_config in loss_model_list.items():
            single_fake_test(name, "fake_test", model_config, async_mode=False, test_loss=True)
            single_fake_test(name, "fake_test", model_config, async_mode=True, test_loss=True)

if __name__ == '__main__':
    main()
