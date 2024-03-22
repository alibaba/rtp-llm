import os
import logging
import logging.config
import torch
from transformers import AutoModelForCausalLM
from unittest import TestCase, main
from maga_transformer.utils.model_weight import W
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.utils.model_weights_loader import ModelWeightsLoader
from maga_transformer.model_factory import ModelFactory, ModelConfig
from maga_transformer.utils.database import CkptDatabase, ModuleDatabase

class ModelWeihgtsLoaderTest(TestCase):
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/utils/test/testdata/model_weights_loader_testdata/')

    @staticmethod
    def create_config(model_type, ckpt_path,  tokenizer_path = None, weight_type=WEIGHT_TYPE.FP16,
                      max_seq_len=2049, seq_size_per_block=1, tp_size=1,
                      ptuning_path=None, ref_model=None):
        model_cls = ModelFactory.get_model_cls(model_type)
        model_config = ModelConfig(model_type=model_type,
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,            
            weight_type=weight_type,
            max_seq_len=max_seq_len,
            seq_size_per_block=seq_size_per_block,
            gen_num_per_circle=1,
            ptuning_path=ptuning_path,
            ref_model=ref_model
        )
        config = model_cls.create_config(model_config)
        return config

    @staticmethod
    def load_ckpt(model_type, ckpt_path, tp_size=1, tp_rank=1, pp_size=1, pp_rank=1, num_layers = None, ptuning_path=None):
        config = ModelWeihgtsLoaderTest.create_config(model_type, ckpt_path, tp_size=tp_size, ptuning_path=ptuning_path)
        config.num_layers = num_layers if num_layers else config.num_layers
        logging.info(f"config.num_layers:{config.num_layers}")
        weights_info = ModelFactory.get_weight_cls(model_type)(config, tp_size, tp_rank)
        database = CkptDatabase(ckpt_path)
        model_weights_loader = ModelWeightsLoader(weights_info, database)
        model_weights_loader.set_data_type(torch.float16)
        weights = model_weights_loader.load_weights_from_scratch(config.quant_algo, device="cpu")
        model_weights_loader.show_warns()
        return config, weights
    
    @staticmethod
    def load_module(model_type, ckpt_path, tp_size=1, tp_rank=1, pp_size=1, pp_rank=1, num_layers = None, ptuning_path=None):
        ref_model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
        config = ModelWeihgtsLoaderTest.create_config(model_type, ckpt_path, tp_size=tp_size, ptuning_path=ptuning_path, ref_model=ref_model)
        config.num_layers = num_layers if num_layers else config.num_layers
        logging.info(f"config.num_layers:{config.num_layers}")
        weights_info = ModelFactory.get_weight_cls(model_type)(config, tp_size, tp_rank)
        database = ModuleDatabase(ref_model)
        model_weights_loader = ModelWeightsLoader(weights_info, database)
        model_weights_loader.set_data_type(torch.float16)
        weights = model_weights_loader.load_weights_from_scratch(config.quant_algo, device="cpu")
        model_weights_loader.show_warns()
        return config, weights

    def test_load_from_module(self):
        model_type = "qwen_7b"
        ckpt_path = "/mnt/nas1/smoke/Qwen-7B-Chat"
        config, weights = ModelWeihgtsLoaderTest.load_module(model_type, ckpt_path, num_layers=1)
        self.assertEqual(config.num_layers, len(weights.weights))
        
        self.assertEqual([151936, 4096], list(weights.steal_pytorch_weight(W.embedding).shape))

        self.assertEqual([4096], list(weights.steal_pytorch_weight(W.final_ln_gamma).shape))

        self.assertEqual([12288], list(weights.weights[0][W.attn_qkv_w][0].shape))

        self.assertEqual([4096], list(weights.weights[0][W.ffn_w2][0].shape))

    def test_qwen_megatron_model_load(self):
        ckpt_path = os.path.join(ModelWeihgtsLoaderTest._testdata_path(), "qwen_14b_megatron_model")
        model_type = "qwen_13b"
        config, weights = ModelWeihgtsLoaderTest.load_ckpt(model_type, ckpt_path, num_layers = 1)
        self.assertEqual(config.num_layers, len(weights.weights))
        self.assertEqual([4, 5120], list(weights.steal_pytorch_weight(W.embedding).shape))
        self.assertEqual([4, 5120], list(weights.steal_pytorch_weight(W.lm_head).shape))
        self.assertEqual([3, 640], list(weights.weights[0][W.attn_qkv_w][0].shape))


    def test_qwen_megatron_pp_model_load(self):
        ckpt_path = os.path.join(ModelWeihgtsLoaderTest._testdata_path(), "yanghao-qwen-13b-sft")
        model_type = "qwen_13b"
        config, weights = ModelWeihgtsLoaderTest.load_ckpt(model_type, ckpt_path, num_layers = 4)
        self.assertEqual(config.num_layers, len(weights.weights))
        self.assertEqual([1, 5120], list(weights.steal_pytorch_weight(W.embedding).shape))
        self.assertEqual([1, 5120], list(weights.steal_pytorch_weight(W.lm_head).shape))
        self.assertEqual([3, 640], list(weights.weights[0][W.attn_qkv_w][0].shape))
        self.assertEqual([3, 640], list(weights.weights[1][W.attn_qkv_w][0].shape))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    main()
