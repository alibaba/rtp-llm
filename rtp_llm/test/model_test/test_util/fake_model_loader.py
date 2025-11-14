import json
import logging
import os

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    QuantizationConfig,
    RenderConfig,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
)
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
        device_reserve_memory_bytes: int = -1073741824,
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

    def init_model(self) -> BaseModel:
        config_path = os.path.join(self.ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
        else:
            raise Exception("not existed config_path ", config_path)
        model_cls = ModelFactory.get_model_cls(self.model_type)

        config: ModelConfig = model_cls._create_config(self.ckpt_path)
        # Apply model_config settings
        config.ckpt_path = self.ckpt_path
        config.tokenizer_path = self.tokenizer_path
        config.max_seq_len = (
            self.max_seq_len if self.max_seq_len > 0 else config.max_seq_len
        )
        # Set data_type instead of act_type (act_type is not a direct attribute)
        if self.act_type:
            config.data_type = self.act_type
        config.model_type = self.model_type

        # Update from config.json
        config.attn_config.head_num = config_json.get(
            "num_attention_heads", config.attn_config.head_num
        )
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.kv_head_num
        )
        if config_json.get("multi_query_attention", False):
            config.attn_config.kv_head_num = config_json["multi_query_group_num"]
        config.attn_config.size_per_head = config_json.get(
            "kv_channels", config.attn_config.size_per_head
        )
        config.inter_size = config_json.get("ffn_hidden_size", config.inter_size)
        config.num_layers = config_json.get("num_layers", config.num_layers)
        config.vocab_size = config_json.get("padded_vocab_size", config.vocab_size)
        config.pre_seq_len = config_json.get("pre_seq_len", config.pre_seq_len)
        config.attn_config.is_causal = self.is_causal

        # Create ModelArgs from config
        model_args = ModelArgs()
        model_args.ckpt_path = self.ckpt_path
        model_args.tokenizer_path = self.tokenizer_path
        model_args.model_type = self.model_type
        model_args.act_type = self.act_type
        model_args.max_seq_len = self.max_seq_len

        quantization_config = QuantizationConfig()
        quantization_config.quantization = self.quantization

        # Create minimal configs for testing
        kv_cache_config = KVCacheConfig()
        kv_cache_config.seq_size_per_block = (
            64  # Set seq_size_per_block in KVCacheConfig
        )

        # Create minimal config objects for BaseModel
        parallelism_config = ParallelismConfig()
        hw_kernel_config = HWKernelConfig()
        fmha_config = FMHAConfig()
        moe_config = MoeConfig()

        # Build model config
        build_model_config(
            model_config=config,
            model_args=model_args,
            kv_cache_config=kv_cache_config,
            quantization_config=quantization_config,
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
            embedding_config=None,  # Fake loader doesn't need embedding_config
        )
        config.render_config = RenderConfig()
        config.generate_env_config = GenerateEnvConfig()

        model = model_cls(
            model_config=config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            load_python_model=self.load_py_model,
            max_generate_batch_size=0,
            vit_config=None,
            merge_lora=False,
        )
        model.load()
        return model
