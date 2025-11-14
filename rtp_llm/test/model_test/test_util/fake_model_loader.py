import json
import logging
import os

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.engine_creator import create_engine
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
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

    def init_engine(self) -> BaseEngine:
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

        raw_config: ModelConfig = model_cls._create_config(self.ckpt_path)
        # Apply model_config settings
        raw_config.ckpt_path = self.ckpt_path
        raw_config.tokenizer_path = self.tokenizer_path
        raw_config.max_seq_len = self.max_seq_len if self.max_seq_len > 0 else raw_config.max_seq_len
        raw_config.quantization = model_config.quantization
        raw_config.act_type = self.act_type
        raw_config.model_type = self.model_type
        
        # Update from config.json
        raw_config.head_num_ = config_json.get(
            "num_attention_heads", raw_config.head_num_
        )
        raw_config.head_num_kv_ = config_json.get(
            "num_key_value_heads", raw_config.head_num_kv_
        )
        if config_json.get("multi_query_attention", False):
            raw_config.head_num_kv_ = config_json["multi_query_group_num"]
        raw_config.size_per_head_ = config_json.get(
            "kv_channels", raw_config.size_per_head_
        )
        raw_config.inter_size = config_json.get(
            "ffn_hidden_size", raw_config.inter_size
        )
        raw_config.inter_padding_size_ = config_json.get(
            "ffn_inter_padding_size", raw_config.inter_padding_size_
        )
        raw_config.num_layers = config_json.get("num_layers", raw_config.num_layers)
        raw_config.vocab_size = config_json.get(
            "padded_vocab_size", raw_config.vocab_size
        )
        raw_config.pre_seq_len = config_json.get("pre_seq_len", raw_config.pre_seq_len)
        raw_config.is_causal_ = self.is_causal
        
        # Create ModelArgs for build_model_config
        from rtp_llm.config.model_args import ModelArgs
        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.config.model_config import build_model_config
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.ops import HWKernelConfig, ProfilingDebugLoggingConfig, ParallelismConfig
        
        # Create ModelArgs from raw_config
        model_args = ModelArgs()
        model_args.ckpt_path = self.ckpt_path
        model_args.tokenizer_path = self.tokenizer_path
        model_args.model_type = self.model_type
        model_args.act_type = self.act_type
        
        # Create minimal configs for testing
        engine_config = EngineConfig()
        kv_cache_config = KVCacheConfig()
        py_hw_kernel_config = HWKernelConfig()
        profiling_debug_logging_config = ProfilingDebugLoggingConfig()
        parallelism_config = ParallelismConfig()
        
        # Build engine config (minimal setup for testing)
        # Note: EngineConfig.create() requires full py_env_configs, so we'll skip it for fake loader
        # and just set minimal values
        engine_config.kv_cache_config = kv_cache_config
        engine_config.parallelism_config = parallelism_config
        
        # Build model config
        model_config = raw_config
        build_model_config(
            model_config=model_config,
            model_args=model_args,
            kv_cache_config=kv_cache_config,
            py_hw_kernel_config=py_hw_kernel_config,
            profiling_debug_logging_config=profiling_debug_logging_config,
            parallelism_config=parallelism_config,
        )

        model = model_cls(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=None,
            merge_lora=False,
        )
        model.load()
        
        # Create engine using create_engine function
        from rtp_llm.async_decoder_engine.engine_creator import create_engine
        engine = create_engine(
            model=model,
            alog_conf_path=profiling_debug_logging_config.ft_alog_conf_path,
            gang_info=None,
            propose_model=None
        )
        engine.start()
        return engine
