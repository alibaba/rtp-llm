"""Base class for TRT attention tests with shared utilities"""

import math
from typing import List

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    compute_pytorch_prefill_reference,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta, init_device


class TRTAttnTestBase(BaseAttentionTest):
    """Base test class for TRT attention operations

    Provides shared utilities for:
    - Device initialization
    - Config creation
    - Attention inputs creation (padded/non-padded)
    - QKV tensor creation
    - KV cache creation
    """

    def setUp(self):
        """Set up test fixtures and initialize device"""
        super().setUp()

        # Initialize device for TRT operations
        try:
            py_env_configs = PyEnvConfigs()
            py_env_configs.model_args.max_seq_len = 512

            if py_env_configs.device_resource_config.device_reserve_memory_bytes == 0:
                py_env_configs.device_resource_config.device_reserve_memory_bytes = (
                    -2 * 1024 * 1024 * 1024
                )

            engine_config = EngineConfig.create(py_env_configs)
            model_config = ModelConfig()
            model_config.max_seq_len = 512

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
            print("Device initialized successfully", flush=True)
        except Exception as e:
            print(f"Warning: Failed to initialize device: {e}", flush=True)
            import traceback

            traceback.print_exc()

    def _create_config(
        self,
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
        data_type: str = "fp16",
    ):
        """Helper to create attention config"""
        attn_configs = AttentionConfigs()
        attn_configs.head_num = head_num
        attn_configs.kv_head_num = head_num_kv
        attn_configs.size_per_head = size_per_head
        attn_configs.tokens_per_block = seq_size_per_block
        attn_configs.use_mla = False

        dtype_map = {
            "fp16": torch.float16,
            "fp8": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
        }
        attn_configs.dtype = dtype_map.get(data_type, torch.float16)

        return attn_configs

    def _create_prefill_attention_inputs(
        self,
        batch_size: int,
        input_lengths: List[int],
        seq_size_per_block: int,
        prefix_lengths: List[int] = None,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill (non-padded mode)"""
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True

        if prefix_lengths is None:
            attn_inputs.prefix_lengths = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            prefix_lens = [0] * batch_size
        else:
            attn_inputs.prefix_lengths = torch.tensor(
                prefix_lengths, dtype=torch.int32, device=self.device
            )
            prefix_lens = prefix_lengths

        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=self.device
        )

        cu_seqlens = [0]
        for seq_len in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        cu_kv_seqlens = [0]
        for i in range(batch_size):
            total_kv = prefix_lens[i] + input_lengths[i]
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + total_kv)
        attn_inputs.cu_kv_seqlens = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device=self.device
        )

        attn_inputs.total_tokens = cu_seqlens[-1]
        attn_inputs.context_total_kv_length = cu_kv_seqlens[-1]

        total_kvs = [prefix_lens[i] + input_lengths[i] for i in range(batch_size)]
        max_blocks_per_seq = math.ceil(max(total_kvs) / seq_size_per_block)
        kv_cache_block_id = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32
        )

        block_offset = 0
        for i, total_kv in enumerate(total_kvs):
            num_blocks = math.ceil(total_kv / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)

        return attn_inputs

    def _create_prefill_attention_inputs_padded(
        self,
        batch_size: int,
        input_lengths: List[int],
        max_seq_len: int,
        seq_size_per_block: int,
        prefix_lengths: List[int] = None,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill with padded mode (CUDA graph)"""
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True

        if prefix_lengths is None:
            attn_inputs.prefix_lengths = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            prefix_lens = [0] * batch_size
        else:
            attn_inputs.prefix_lengths = torch.tensor(
                prefix_lengths, dtype=torch.int32, device=self.device
            )
            prefix_lens = prefix_lengths

        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=self.device
        )

        cu_seqlens = [0]
        for seq_len in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        cu_kv_seqlens = [0]
        for i in range(batch_size):
            total_kv = prefix_lens[i] + input_lengths[i]
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + total_kv)
        attn_inputs.cu_kv_seqlens = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device=self.device
        )

        attn_inputs.total_tokens = cu_seqlens[-1]
        attn_inputs.context_total_kv_length = cu_kv_seqlens[-1]

        max_blocks_per_seq = math.ceil(max_seq_len / seq_size_per_block)
        kv_cache_block_id = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32
        )

        block_offset = 0
        for i in range(batch_size):
            num_blocks = max_blocks_per_seq
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)

        return attn_inputs

    def _create_qkv_tensor(
        self,
        total_tokens: int,
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Create QKV input tensor"""
        qkv_dim = (head_num + 2 * head_num_kv) * size_per_head
        qkv = torch.randn(
            total_tokens,
            qkv_dim,
            dtype=dtype,
            device=self.device,
        )
        return qkv

    def run_correctness_test(
        self,
        attn_op,
        op_name: str,
        batch_size: int,
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        seq_size_per_block: int,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        prefix_lengths: List[int] = None,
        use_padded: bool = False,
    ):
        """Run correctness test for an attention operator

        Args:
            attn_op: The attention operator instance
            op_name: Name of the operator for logging
            batch_size: Number of sequences
            input_lengths: New token counts to process
            head_num: Number of query heads
            head_num_kv: Number of KV heads
            size_per_head: Dimension per head
            seq_size_per_block: Block size for KV cache
            attn_configs: Attention configuration
            attn_inputs: Pre-created attention inputs
            prefix_lengths: Already cached KV lengths (optional)
            use_padded: Whether using padded mode
        """
        total_tokens = sum(input_lengths)

        qkv = self._create_qkv_tensor(
            total_tokens,
            head_num,
            head_num_kv,
            size_per_head,
            dtype=attn_configs.dtype,
        )

        attn_inputs.dtype = get_typemeta(qkv)

        is_supported = attn_op.support(attn_inputs)
        print(f"{op_name} support check: {is_supported}", flush=True)

        if not is_supported:
            print(
                f"WARNING: {op_name} does not support this configuration, skipping test",
                flush=True,
            )
            return

        params = attn_op.prepare(attn_inputs)
        self.assertIsNotNone(params, f"{op_name} prepare() returned None")

        if prefix_lengths:
            total_kv_lengths = [
                prefix_lengths[i] + input_lengths[i] for i in range(batch_size)
            ]
        else:
            total_kv_lengths = input_lengths

        if use_padded:
            max_total_kv = max(total_kv_lengths)
            max_blocks_per_seq = math.ceil(max_total_kv / seq_size_per_block)
            total_blocks = max_blocks_per_seq * batch_size
        else:
            total_blocks = sum(
                [math.ceil(kv_len / seq_size_per_block) for kv_len in total_kv_lengths]
            )

        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            seq_size_per_block,
            head_num_kv,
            size_per_head,
            dtype=torch.float16,
        )

        try:
            output = attn_op.forward(qkv, kv_cache, params)

            expected_output_dim = head_num * size_per_head
            self.assertEqual(
                output.shape,
                (total_tokens, expected_output_dim),
                f"{op_name} output shape mismatch",
            )

            self.assertFalse(
                torch.isnan(output).any(), f"{op_name} output contains NaN"
            )
            self.assertFalse(
                torch.isinf(output).any(), f"{op_name} output contains Inf"
            )

            print(
                f"{op_name} forward pass successful ✓ Output shape: {output.shape}",
                flush=True,
            )

            print(f"Computing PyTorch reference for {op_name}...", flush=True)
            ref_output = compute_pytorch_prefill_reference(
                qkv.clone(),
                input_lengths,
                head_num,
                head_num_kv,
                size_per_head,
            )

            from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
                compare_tensors,
            )

            print(f"Comparing {op_name} output with PyTorch reference...", flush=True)
            compare_tensors(
                output,
                ref_output,
                rtol=1e-2,
                atol=1e-2,
                name=f"{op_name} output (batch={batch_size}, input_lengths={input_lengths})",
            )

            print(f"✓ {op_name} correctness check passed!", flush=True)

        except Exception as e:
            self.fail(f"{op_name} forward pass failed: {e}")
