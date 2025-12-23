import logging
import math
import os
import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models.base_model import BaseModel

# Add rtp_llm root path
rtp_opensouce_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.append(str(rtp_opensouce_path))

import rtp_llm.models
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_decode_ops import (
    CudaGraphDecodePaddingOp,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import KVCache, get_device, get_typemeta, init_device
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf


class TestCudaGraphDecodePadding(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        os.environ["RESERVER_RUNTIME_MEM_MB"] = "10240"

        # Test parameters (can be configured)
        self.max_seq_len = 64
        self.tokens_per_block = 64
        self.max_batch_size = 128

        # Generate decode_capture_batch_sizes from 1 to max_batch_size, excluding some values
        excluded_batch_sizes = {80, 87, 102}
        self.decode_capture_batch_sizes = [
            bs
            for bs in range(1, self.max_batch_size + 1)
            if bs not in excluded_batch_sizes
        ]

        # Build model using ModelFactory (similar to rtp_auto_model.py)
        model = self.build_model()
        logging.info("build model successfully")

        # Get model parameters for CudaGraphRunner
        self.hidden_size = self.model_config.gpt_init_params.hidden_size

        self.op = CudaGraphDecodePaddingOp()
        self.op.init(
            model,
            self.hidden_size,
            self.max_seq_len,
            self.tokens_per_block,
            self.decode_capture_batch_sizes,
        )
        logging.info(
            f"CUDA Graph initialized with batch sizes: 1 to {self.max_batch_size}"
        )
        self.normal_model = self.build_model()

    def build_model(self) -> GptModelBase:
        """Build model using ModelFactory, similar to auto_model.py"""
        model_path = "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"

        # Set configs (similar to auto_model.py)
        self._set_configs(model_path)

        # Create EngineConfig from py_env_configs
        engine_config = EngineConfig.create(self.py_env_configs)

        # Create model configs
        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
        )

        # Update engine_config based on model_config
        ModelFactory.update_engine_config_from_model_config(
            engine_config=engine_config,
            model_config=model_config,
        )

        # Create model using ModelFactory
        self.gpt_model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=None,
            merge_lora=False,
        )

        # Load the model
        self.gpt_model.load()
        self.compute_dtype = self.gpt_model.weight.dtype
        model = self.gpt_model.py_model
        self.model_config = model.config

        # Init device with new API
        init_device(
            parallelism_config=engine_config.parallelism_config,
            model_config=model_config,
            eplb_config=model_config.eplb_config,
            fmha_config=engine_config.fmha_config,
            device_resource_config=engine_config.device_resource_config,
            moe_config=engine_config.moe_config,
            sp_config=engine_config.sp_config,
            misc_config=engine_config.misc_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            hw_kernel_config=engine_config.hw_kernel_config,
            concurrency_config=engine_config.concurrency_config,
            ffn_disaggregate_config=engine_config.parallelism_config.ffn_disaggregate_config,
            runtime_config=engine_config.runtime_config,
        )
        self.device = "cuda:0"

        # Init kv cache
        self._init_kv_cache()
        model.kv_cache = self.kv_cache

        return model

    def _set_configs(self, model_path: str):
        """Set configuration structures instead of environment variables."""
        # Create PyEnvConfigs to hold all configurations
        self.py_env_configs = PyEnvConfigs()

        # Get model info from HuggingFace
        model_path, model_type = get_model_info_from_hf(model_path, None)
        self.py_env_configs.model_args.model_type = model_type
        self.py_env_configs.model_args.ckpt_path = model_path
        self.py_env_configs.model_args.max_seq_len = 4096
        self.py_env_configs.profiling_debug_logging_config.hack_layer_num = 1
        self.py_env_configs.kv_cache_config.seq_size_per_block = self.tokens_per_block
        if not self.py_env_configs.model_args.tokenizer_path:
            self.py_env_configs.model_args.tokenizer_path = model_path

        # Set ModelSpecificConfig.load_python_model = True
        self.py_env_configs.model_specific_config.load_python_model = True
        self.py_env_configs.device_resource_config.not_use_default_stream = True
        # Set DeviceResourceConfig.device_reserve_memory_bytes
        if self.py_env_configs.device_resource_config.device_reserve_memory_bytes == 0:
            self.py_env_configs.device_resource_config.device_reserve_memory_bytes = (
                -536870912
            )

    def _init_kv_cache(self):
        """Initialize KV cache, similar to auto_model.py"""
        max_total_tokens = 4096
        tokens_per_block = self.tokens_per_block

        self.kv_cache = KVCache()
        self.layer_num = self.model_config.gpt_init_params.layer_num
        self.model_config.gpt_init_params.seq_size_per_block = tokens_per_block
        self.kv_head_num = self.model_config.gpt_init_params.head_num_kv
        self.size_per_head = self.model_config.gpt_init_params.size_per_head

        block_nums = math.ceil(max_total_tokens / tokens_per_block) + 1

        kv_shape = [
            self.layer_num,
            block_nums,
            2,
            self.kv_head_num,
            tokens_per_block,
            self.size_per_head,
        ]

        # Use compute_dtype for KV cache
        kv_cache_total = torch.zeros(
            kv_shape, dtype=self.compute_dtype, device=self.device
        )
        k_cache_base = kv_cache_total
        v_cache_base = torch.empty(
            self.layer_num,
            0,
            self.kv_head_num,
            tokens_per_block,
            self.size_per_head,
            device=self.device,
        )
        self.kv_cache.k_cache_base = k_cache_base
        self.kv_cache.v_cache_base = v_cache_base

    def build_inputs(self, batch_size: int, max_seq_len: int, seq_size_per_block: int):
        """Build inputs in Python, similar to CudaGraphPrefill.py"""
        from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs

        num_tokens_per_bs = 1  # decode mode: 1 token per batch
        max_num_token = batch_size * num_tokens_per_bs

        inputs = PyModelInputs()
        attention_inputs = PyAttentionInputs()

        # input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch.full(
            (max_num_token,), 10, dtype=torch.int32, device="cuda"
        )

        # prefix_lengths [batch_size, int32] (for attention `prepare`)
        attention_inputs.prefix_lengths = torch.empty(0)

        # input_lengths [batch_size, int32] (decode only)
        attention_inputs.input_lengths = torch.ones(batch_size, dtype=torch.int32)

        # sequence_lengths [batch_size, int32] (decode only), with pin_memory
        attention_inputs.sequence_lengths = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()

        # kv_cache_block_id_device [batch_size, block_num]
        block_num = (max_seq_len + seq_size_per_block - 1) // seq_size_per_block
        attention_inputs.kv_cache_block_id_device = torch.zeros(
            (batch_size, block_num), dtype=torch.int32, device="cuda"
        )
        attention_inputs.kv_cache_block_id_host = torch.zeros(
            (batch_size, block_num), dtype=torch.int32, device="cpu"
        )

        # padding_offset
        attention_inputs.padding_offset = torch.zeros(
            max_seq_len, dtype=torch.int32, device="cuda"
        )

        # Set attention parameters
        attention_inputs.is_prefill = False
        attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))

        # cu_seqlens
        cu_len = batch_size + 1
        cu_seqlens = torch.zeros(cu_len, dtype=torch.int32, device="cpu").pin_memory()

        attention_inputs.cu_seqlens = cu_seqlens

        inputs.attention_inputs = attention_inputs
        return inputs

    def _test_single(self, batch_size: int):
        inputs = self.build_inputs(
            batch_size,
            self.max_seq_len,
            self.tokens_per_block,
        )
        inputs2 = self.build_inputs(
            batch_size,
            self.max_seq_len,
            self.tokens_per_block,
        )

        outputs1 = self.op.forward(inputs)
        outputs2 = self.normal_model.forward(inputs2)

        current_real_graph_size = self.op.getCurrentRealGraphSize()
        logging.info(
            f"current_real_graph_size: {current_real_graph_size}, batch_size: {batch_size}"
        )

        # With continuous capture from 1 to max_batch_size, real graph size should equal batch_size
        assert (
            current_real_graph_size >= batch_size
        ), f"Expected real graph size {batch_size}, got {current_real_graph_size}"

        logging.info(f"outputs1.hidden_states: {outputs1.hidden_states[0]}")
        logging.info(f"outputs2.hidden_states: {outputs2.hidden_states[0]}")

        outputs2.hidden_states = outputs2.hidden_states.type(
            outputs1.hidden_states.dtype
        )
        torch.testing.assert_close(
            outputs1.hidden_states[:batch_size], outputs2.hidden_states
        )

    def test_batch_decode(self):
        batch_range = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            19,
            27,
            48,
            64,
            80,
            87,
            96,
            102,
            112,
            125,
            128,
        ]

        for bs in batch_range:
            self._test_single(bs)
            logging.info(f"success for batch size: {bs}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
