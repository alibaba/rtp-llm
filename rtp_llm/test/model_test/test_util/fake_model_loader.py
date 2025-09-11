import json
import logging
import os

from rtp_llm.async_decoder_engine.async_model import AsyncModel
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory import ModelConfig, ModelFactory
from rtp_llm.utils.weight_type import WEIGHT_TYPE


class FakeModelLoader(object):
    def __init__(
        self,
        model_type: str,
        tokenizer_path: str,
        ckpt_path: str,
        act_type: str = "bf16",
        max_seq_len: int = 0,
        quantization: str = "",
        data_type: str = WEIGHT_TYPE.AUTO.to_str(),
        kv_cache_type: str = WEIGHT_TYPE.AUTO.to_str(),
        load_py_model: bool = False,
        device_reserve_memory_bytes: int = 0,
        warm_up: bool = False,
        is_causal: bool = True,
    ) -> None:
        self.model_type = model_type
        self.tokenizer_path = tokenizer_path
        self.ckpt_path = ckpt_path
        self.max_seq_len: int = max_seq_len
        self.quantization = quantization
        self.load_py_model = load_py_model
        self.device_reserve_memory_bytes = device_reserve_memory_bytes
        self.warm_up = warm_up
        self.data_type = data_type
        self.kv_cache_type = kv_cache_type
        self.act_type = act_type
        self.is_causal = is_causal

        logging.info(f"tokenizer path: {self.tokenizer_path}")
        logging.info(f"check point path: {self.ckpt_path}")

    def load_model(self):
        config_path = os.path.join(self.ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
        else:
            raise Exception("not existed config_path ", config_path)
        os.environ["WARM_UP"] = "0"
        model_cls = ModelFactory.get_model_cls(self.model_type)
        model_config = ModelConfig(
            ckpt_path=self.ckpt_path,
            model_type=self.model_type,
            tokenizer_path=self.tokenizer_path,
            act_type=self.act_type,
            max_seq_len=64,
            seq_size_per_block=64,
            gen_num_per_circle=1,
            quantization=self.quantization,
        )

        raw_config: GptInitModelParameters = model_cls.create_config(model_config)
        raw_config.model_specific_config.load_python_model = self.load_py_model
        raw_config.device_resource_config.device_reserve_memory_bytes = (
            self.device_reserve_memory_bytes
        )
        raw_config.warm_up = self.warm_up
        raw_config.head_num = config_json.get(
            "num_attention_heads", raw_config.head_num
        )
        raw_config.head_num_kv = config_json.get(
            "num_key_value_heads", raw_config.head_num_kv
        )
        if config_json.get("multi_query_attention", False):
            raw_config.head_num_kv = config_json["multi_query_group_num"]
        raw_config.size_per_head = config_json.get(
            "kv_channels", raw_config.size_per_head
        )
        raw_config.inter_size = config_json.get(
            "ffn_hidden_size", raw_config.inter_size
        )
        raw_config.inter_padding_size = config_json.get(
            "ffn_inter_padding_size", raw_config.inter_padding_size
        )
        raw_config.layer_num = config_json.get("num_layers", raw_config.layer_num)
        raw_config.vocab_size = config_json.get(
            "padded_vocab_size", raw_config.vocab_size
        )
        raw_config.pre_seq_len = config_json.get("pre_seq_len", raw_config.pre_seq_len)

        raw_config.update_common(
            ckpt_path=self.ckpt_path,
            lora_infos=None,
            tokenizer_path=self.tokenizer_path,
            quantization=model_config.quantization,
            data_type=self.data_type,
            kv_cache_type=self.kv_cache_type,
            max_seq_len=self.max_seq_len,
            seq_size_per_block=64,
            gen_num_per_circle=1,
            ptuning_path=None,
        )
        raw_config.is_causal = self.is_causal
        model = model_cls.from_config(raw_config)
        model = AsyncModel(model, None)
        return model
