import logging
import math
import os
import sys
import unittest
from pathlib import Path


from rtp_llm.models.base_model import BaseModel
import torch

# Add rtp_llm root path
rtp_opensouce_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.append(str(rtp_opensouce_path))

import rtp_llm.models
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.cpp.devices.cuda_impl.tests.libtest_cuda_graph_decode_ops import (
    CudaGraphDecodePaddingOp,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import KVCache, get_device, get_typemeta, init_device


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
        """Build model using ModelFactory, similar to rtp_auto_model.py"""
        model_path = "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"

        # Set environment variables
        self._set_env()

        # Set config
        StaticConfig.model_config.checkpoint_path = model_path
        StaticConfig.update_from_env()

        # Create model config and load model
        factory_model_config = ModelFactory.create_normal_model_config()
        self.gpt_model = ModelFactory.creat_standalone_py_model_from_huggingface(
            model_path_or_name=model_path,
            revision=None,
            model_config=factory_model_config,
        )

        self.compute_dtype = self.gpt_model.compute_dtype
        model = self.gpt_model.py_model
        self.model_config = model.config

        # Init device
        init_device(self.gpt_model.config)
        self.device = get_device().get_device_type().name.lower()

        # Init kv cache
        self._init_kv_cache(factory_model_config)
        model.kv_cache = self.kv_cache

        return model

    def _set_env(self):
        """Set environment variables for standalone model loading"""
        os.environ["LOAD_PYTHON_MODEL"] = "1"

        if os.getenv("ACT_TYPE") is None:
            os.environ["ACT_TYPE"] = "FP16"
        if os.getenv("DEVICE_RESERVE_MEMORY_BYTES") is None:
            os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(-536870912)

    def _init_kv_cache(self, factory_model_config):
        """Initialize KV cache, similar to rtp_auto_model.py"""
        max_total_tokens = 4096
        tokens_per_block = 64

        self.kv_cache = KVCache()
        self.layer_num = self.model_config.gpt_init_params.layer_num
        self.model_config.gpt_init_params.seq_size_per_block = tokens_per_block
        self.kv_head_num = self.model_config.gpt_init_params.head_num_kv
        self.size_per_head = self.model_config.gpt_init_params.size_per_head

        block_nums = math.ceil(max_total_tokens / tokens_per_block) + 1
        self.tokens_per_block = tokens_per_block

        kv_shape = [
            self.layer_num,
            block_nums,
            2,
            self.kv_head_num,
            tokens_per_block,
            self.size_per_head,
        ]

        kv_cache_dtype = self._get_kv_cache_dtype(factory_model_config)
        kv_cache_total = torch.zeros(kv_shape, dtype=kv_cache_dtype, device=self.device)
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

    def _get_kv_cache_dtype(self, factory_model_config) -> torch.dtype:
        """Get KV cache dtype from config"""
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
