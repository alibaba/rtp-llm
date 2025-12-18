import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

import rtp_llm.models
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_device,
    get_typemeta,
    init_device,
)
from rtp_llm.utils.base_model_datatypes import ModelConfig
from rtp_llm.utils.model_weight import W

rtp_opensouce_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(rtp_opensouce_path))


class AutoModel:
    def __init__(
        self,
        model_path_or_name: str,
        **kwargs,
    ):
        # get some initial parameters
        revision: Optional[str] = kwargs.get("revision", None)
        max_total_tokens: int = kwargs.get("max_total_tokens", 4096)
        tokens_per_block: int = kwargs.get("tokens_per_block", 64)
        stop_id_list: list[int] = kwargs.get("stop_id_list", [])

        # set some env and config
        self._set_env()
        StaticConfig.model_config.checkpoint_path = model_path_or_name
        StaticConfig.update_from_env()
        # hf_model_info = get_hf_model_info(model_path_or_name)

        # init C++ logger
        # Note: In standalone mode, libth_transformer is not loaded,
        # so torch.ops.rtp_llm.init_engine is not available.
        # alog_conf_path = os.environ.get("FT_ALOG_CONF_PATH", "")
        # print(f"  alog_conf_path: '{alog_conf_path}'")
        # torch.ops.rtp_llm.init_engine(alog_conf_path)

        # load model
        self.factory_model_config = ModelFactory.create_normal_model_config()
        self.gpt_model = ModelFactory.creat_standalone_py_model_from_huggingface(
            model_path_or_name=model_path_or_name,
            revision=revision,
            model_config=self.factory_model_config,
        )
        self.compute_dtype = self.gpt_model.compute_dtype
        self.model = self.gpt_model.py_model
        self.model_config = self.model.config

        # init device
        init_device(self.gpt_model.config)
        self.device = get_device().get_device_type().name.lower()

        # init kv cache and bind it to py model
        self.block_nums = math.ceil(max_total_tokens / tokens_per_block)
        # since block_id start from 1, so we should add 1 in the corner case
        self.block_nums += 1
        logging.info(f"total block nums: {self.block_nums}")
        self.tokens_per_block = tokens_per_block
        self._init_kv_cache()
        self.model.kv_cache = self.kv_cache

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.ckpt_path)

        # add eos_token_id to stop_id_list
        self.stop_id_list = stop_id_list.copy() if stop_id_list else []
        if (
            self.tokenizer.eos_token_id is not None
            and self.tokenizer.eos_token_id not in self.stop_id_list
        ):
            self.stop_id_list.append(self.tokenizer.eos_token_id)

        # convert hidden_states(after final layernorm) to logits
        self.lm_head_weight = self.model.weight.get_global_weight(W.lm_head)

    @classmethod
    def from_pretrained(cls, model_path_or_name: str, **kwargs) -> "AutoModel":
        return cls(model_path_or_name, **kwargs)

    def _set_env(self):
        os.environ["LOAD_PYTHON_MODEL"] = "1"

        if os.getenv("ACT_TYPE") is None:
            os.environ["ACT_TYPE"] = "AUTO"
        if os.getenv("DEVICE_RESERVE_MEMORY_BYTES") is None:
            os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(2 * 1024 * 1024 * 1024)
        # if os.getenv("FT_ALOG_CONF_PATH") is None:
        #     os.environ["FT_ALOG_CONF_PATH"] = os.path.join(
        #         str(rtp_opensouce_path), "rtp_llm/test/alog_conf.json"
        #     )
        # if os.getenv("RTP_LLM_LOG_LEVEL") is None:
        #     os.environ["RTP_LLM_LOG_LEVEL"] = "INFO"

    def _init_kv_cache(self):
        self.kv_cache = KVCache()
        self.layer_num = self.model_config.gpt_init_params.layer_num
        self.model_config.gpt_init_params.seq_size_per_block = self.tokens_per_block
        self.kv_head_num = self.model_config.gpt_init_params.head_num_kv
        self.size_per_head = self.model_config.gpt_init_params.size_per_head
        kv_shape = [
            self.layer_num,
            self.block_nums,
            2,
            self.kv_head_num,
            self.tokens_per_block,
            self.size_per_head,
        ]
        kv_cache_dtype = self._get_kv_cache_dtype(self.factory_model_config)
        kv_cache_total = torch.zeros(kv_shape, dtype=kv_cache_dtype, device=self.device)
        k_cache_base = kv_cache_total
        v_cache_base = torch.empty(
            self.layer_num,
            0,
            self.kv_head_num,
            self.tokens_per_block,
            self.size_per_head,
            device=self.device,
        )
        self.kv_cache.k_cache_base = k_cache_base
        self.kv_cache.v_cache_base = v_cache_base

    def _get_kv_cache_dtype(self, factory_model_config: ModelConfig) -> torch.dtype:
        kv_cache_dtype_str = factory_model_config.kv_cache_type
        if kv_cache_dtype_str == "auto":
            return self.compute_dtype
        if kv_cache_dtype_str not in ["FP16", "BF16", "FP32"]:
            raise ValueError(f"Invalid kv cache dtype: {kv_cache_dtype_str}")
        str_to_dtype = {
            "FP16": torch.float16,
            "BF16": torch.bfloat16,
            "FP32": torch.float32,
        }
        return str_to_dtype[kv_cache_dtype_str]

    def _prepare_prefill_attention_inputs(self, input_length: int) -> PyAttentionInputs:
        need_block_nums = self._check_block_nums(input_length)
        attention_inputs = PyAttentionInputs()
        attention_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        attention_inputs.sequence_lengths = torch.tensor([], dtype=torch.int32)
        attention_inputs.cu_seqlens = torch.tensor(
            [0, input_length], dtype=torch.int32, device=self.device
        )
        attention_inputs.cu_seqlens_without_prefix = torch.tensor(
            [0, input_length], dtype=torch.int32, device=self.device
        )
        attention_inputs.prefix_lengths = torch.tensor([0], dtype=torch.int32)
        attention_inputs.padding_offset = torch.tensor(
            [0 for _ in range(input_length)], dtype=torch.int32, device=self.device
        )
        attention_inputs.kv_cache_block_id_device = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        attention_inputs.dtype = get_typemeta(self.kv_cache.k_cache_base)
        attention_inputs.is_prefill = True
        return attention_inputs

    def _check_block_nums(self, sequence_length: int) -> int:
        need_block_nums = math.ceil(sequence_length / self.tokens_per_block)
        # plus one for zero case
        assert need_block_nums + 1 <= self.block_nums, "sequence_length is too long"
        return need_block_nums

    def _prepare_decode_attention_inputs(
        self, attention_inputs: PyAttentionInputs, sequence_length: int
    ) -> PyAttentionInputs:
        need_block_nums = self._check_block_nums(sequence_length)
        attention_inputs.is_prefill = False
        attention_inputs.padding_offset = torch.tensor(
            [0], dtype=torch.int32, device=self.device
        )
        attention_inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
        # sequence_lengths is index, so minus 1
        attention_inputs.sequence_lengths = torch.tensor(
            [sequence_length - 1], dtype=torch.int32
        ).pin_memory()
        attention_inputs.kv_cache_block_id_device = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs.kv_cache_block_id_host = torch.tensor(
            [[i for i in range(1, need_block_nums + 1)]], dtype=torch.int32
        )
        return attention_inputs

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 10,
        sampling_params: dict = None,
    ) -> list[int]:
        output_ids = []

        # prefill
        input_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        attention_inputs = self._prepare_prefill_attention_inputs(input_length)
        model_inputs = PyModelInputs(
            input_ids=input_ids,
            attention_inputs=attention_inputs,
        )
        model_outputs = self.model.forward(model_inputs)
        next_token_id = self._sample_next_token(model_outputs, sampling_params)
        next_token_id_cpu = next_token_id.cpu().item()
        # check if the next token is a stop token
        if next_token_id_cpu in self.stop_id_list:
            return output_ids
        output_ids.append(next_token_id_cpu)

        # decode
        gen_tokens = 1
        while gen_tokens < max_new_tokens:
            attention_inputs = self._prepare_decode_attention_inputs(
                attention_inputs, input_length + gen_tokens
            )
            model_inputs = PyModelInputs(
                input_ids=next_token_id,
                attention_inputs=attention_inputs,
            )
            model_outputs = self.model.forward(model_inputs)
            next_token_id = self._sample_next_token(model_outputs, sampling_params)
            next_token_id_cpu = next_token_id.cpu().item()
            gen_tokens += 1
            # check if the next token is a stop token
            if next_token_id_cpu in self.stop_id_list:
                break
            output_ids.append(next_token_id_cpu)

        return output_ids

    def _sample_next_token(
        self, model_outputs: PyModelOutputs, sampling_params: dict = None
    ) -> torch.Tensor:
        hidden_states = model_outputs.hidden_states[-1:, :]
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        next_token_id = torch.argmax(logits, dim=-1)
        return next_token_id
