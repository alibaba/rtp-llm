"""Base class for TRT attention tests with shared utilities"""

import math
from typing import List, Tuple

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
    set_seed,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    compute_pytorch_prefill_reference,
    print_attn_inputs_detail,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta, init_exec_ctx
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance


class TRTLLMFMHAv2TestBase(BaseAttentionTest):
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
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = (
                64
            )

            engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)
            model_config = ModelConfig()
            model_config.max_seq_len = 512

            pc = engine_config.parallelism_config
            init_exec_ctx(
                device_id=pc.world_rank % pc.local_world_size,
                trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
                enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
                mla_ops_type=int(model_config.mla_ops_type),
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
        attn_configs.kernel_tokens_per_block = seq_size_per_block
        attn_configs.use_mla = False
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE

        dtype_map = {
            "fp16": torch.float16,
            "fp8": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
        }
        attn_configs.dtype = dtype_map.get(data_type, torch.float16)

        return attn_configs

    def _create_prefill_attention_inputs_base(
        self,
        batch_size: int,
        input_lengths: List[int],
        prefix_lengths: List[int] = None,
    ) -> Tuple[PyAttentionInputs, List[int]]:
        """Create base PyAttentionInputs with common fields

        Returns:
            tuple: (attn_inputs, prefix_lens) where prefix_lens is the processed prefix lengths list
        """
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
        attn_inputs.cu_seqlens_device = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        cu_kv_seqlens = [0]
        for i in range(batch_size):
            total_kv = prefix_lens[i] + input_lengths[i]
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + total_kv)
        attn_inputs.cu_kv_seqlens_device = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device=self.device
        )

        max_seq_len = max(input_lengths)
        padding_offsets = []
        cumulative_padding = 0
        for seq_len in input_lengths:
            padding_offsets.extend([cumulative_padding] * seq_len)
            cumulative_padding += max_seq_len - seq_len
        attn_inputs.padding_offset = torch.tensor(
            padding_offsets, dtype=torch.int32, device=self.device
        )

        attn_inputs.total_tokens = cu_seqlens[-1]
        attn_inputs.context_total_kv_length = cu_kv_seqlens[-1]

        return attn_inputs, prefix_lens

    def _create_prefill_attention_inputs(
        self,
        batch_size: int,
        input_lengths: List[int],
        seq_size_per_block: int,
        prefix_lengths: List[int] = None,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill (non-padded mode)"""
        print(
            f"\n[_create_prefill_attention_inputs] batch_size={batch_size}, input_lengths={input_lengths}, "
            f"seq_size_per_block={seq_size_per_block}, prefix_lengths={prefix_lengths}",
            flush=True,
        )

        attn_inputs, prefix_lens = self._create_prefill_attention_inputs_base(
            batch_size, input_lengths, prefix_lengths
        )

        # Non-padded mode: kv_cache_block_id must include ALL KV (prefix + new)
        # This is because:
        # 1. _write_kv_cache will write all KV (simulating FusedRopeKVCache behavior)
        # 2. TRT attention needs to read all KV from these blocks
        total_kvs = [prefix_lens[i] + input_lengths[i] for i in range(batch_size)]
        max_blocks_per_seq = math.ceil(max(total_kvs) / seq_size_per_block)
        kv_cache_block_id = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32
        )

        block_offset = 0
        for i, total_kv in enumerate(total_kvs):
            # Allocate blocks for all KV (prefix + new)
            num_blocks = math.ceil(total_kv / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        # Use a non-identity physical-page mapping so paged correctness tests
        # fail if the kernel ignores block_tables and reads contiguous pages.
        for i, total_kv in enumerate(total_kvs):
            num_blocks = math.ceil(total_kv / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = (
                block_offset - 1 - kv_cache_block_id[i, :num_blocks]
            )

        print(
            f"[Non-padded] total_kvs={total_kvs}, max_blocks_per_seq={max_blocks_per_seq}, total_blocks={block_offset}",
            flush=True,
        )

        attn_inputs.kv_cache_block_id = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        return attn_inputs

    def _create_prefill_attention_inputs_padded(
        self,
        batch_size: int,
        input_lengths: List[int],
        max_seq_len: int,
        seq_size_per_block: int,
        prefix_lengths: List[int] = None,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill with padded mode (CUDA graph)

        CRITICAL: In padded mode:
        - ONLY QKV tensor needs padding: [batch_size * max_seq_len, hidden_dim]
        - ALL length fields remain ACTUAL lengths (not padded):
          * input_lengths: actual sequence lengths [64, 128, 256, 512]
          * cu_seqlens: cumulative sums of actual lengths [0, 64, 192, 448, 960]
          * total_tokens: sum of actual lengths (960, not 2048)
        - KV cache blocks: allocated uniformly (max_blocks_per_seq for all sequences)
        """
        print(
            f"\n[_create_prefill_attention_inputs_padded] batch_size={batch_size}, input_lengths={input_lengths}, "
            f"max_seq_len={max_seq_len}, seq_size_per_block={seq_size_per_block}, prefix_lengths={prefix_lengths}",
            flush=True,
        )

        attn_inputs, prefix_lens = self._create_prefill_attention_inputs_base(
            batch_size, input_lengths, prefix_lengths
        )

        # Padded mode: kv_cache_block_id must include ALL KV (prefix + new)
        # In padded mode, use max_seq_len (which is max of input_lengths) for allocation
        total_kvs = [prefix_lens[i] + input_lengths[i] for i in range(batch_size)]
        max_blocks_per_seq = math.ceil(max(total_kvs) / seq_size_per_block)

        kv_cache_block_id = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32
        )

        block_offset = 0
        for i, total_kv in enumerate(total_kvs):
            # Allocate blocks for all KV (prefix + new)
            num_blocks = math.ceil(total_kv / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        print(
            f"[Padded] total_kvs={total_kvs}, max_blocks_per_seq={max_blocks_per_seq}, total_blocks={block_offset}",
            flush=True,
        )

        attn_inputs.kv_cache_block_id = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.is_s_padded = True
        return attn_inputs

    def _create_qkv_tensor(
        self,
        total_tokens: int,
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        set_seed(256)
        """Create QKV input tensor"""
        qkv_dim = (head_num + 2 * head_num_kv) * size_per_head
        qkv = torch.randn(
            total_tokens,
            qkv_dim,
            dtype=dtype,
            device=self.device,
        )
        print(
            f"Created QKV tensor: shape={qkv.shape}, total_tokens={total_tokens}, qkv_dim={qkv_dim}",
            flush=True,
        )
        return qkv

    ## 1. kv_cache: KVCache object with kv_cache_base [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
    ## 2. qkv_full: [total_tokens, qkv_dim] where qkv_dim = (head_num + 2 * num_kv_heads) * head_dim
    def _write_kv_cache(
        self,
        qkv_full: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        total_kv_lengths: List[int],
        kv_cache,  # KVCache object
    ):
        import math

        prefix_lengths = attn_inputs.prefix_lengths
        total_tokens = sum(total_kv_lengths)

        # Get the actual tensor from KVCache object
        kv_cache_tensor = (
            kv_cache.kv_cache_base
        )  # [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        num_kv_heads = kv_cache_tensor.shape[2]
        head_dim = kv_cache_tensor.shape[4]
        seq_size_per_block = kv_cache_tensor.shape[3]

        # Calculate head_num (number of Q heads)
        # qkv_full: [total_tokens, qkv_dim] where qkv_dim = (head_num + 2*num_kv_heads) * head_dim
        qkv_dim = qkv_full.shape[1]
        head_num = qkv_dim // head_dim - 2 * num_kv_heads

        self.assertEqual(
            qkv_full.shape[0],
            total_tokens,
            f"qkv_full.shape[0]:{qkv_full.shape[0]} != total_tokens:{total_tokens}",
        )

        batch_size = len(prefix_lengths)
        offset = 0
        block_ids = attn_inputs.kv_cache_kernel_block_id
        for i in range(batch_size):
            # Write ALL KV (prefix + new tokens), not just prefix
            seq_len = total_kv_lengths[i]
            # cache_tensor: [seq_len, qkv_dim] where qkv_dim = (head_num + 2*num_kv_heads) * head_dim
            cache_tensor = qkv_full[offset : offset + seq_len]
            offset += total_kv_lengths[i]

            # Reshape to [seq_len, head_num + 2*num_kv_heads, head_dim] for easy slicing
            cache_tensor = cache_tensor.view(
                seq_len, head_num + 2 * num_kv_heads, head_dim
            )

            print(
                f"write cache for batch {i}, seq_len={seq_len}, cache_tensor shape={cache_tensor.shape}"
            )
            allocate_blocks = math.ceil((total_kv_lengths[i] / seq_size_per_block))
            print(
                f"  Allocating {allocate_blocks} blocks for {total_kv_lengths[i]} tokens",
                flush=True,
            )

            ## Extract K and V from cache_tensor
            ## [seq_len, num_kv_heads, head_dim]
            k_tensor = cache_tensor[:, head_num : head_num + num_kv_heads, :]
            v_tensor = cache_tensor[
                :, head_num + num_kv_heads : head_num + 2 * num_kv_heads, :
            ]

            # Transpose to [num_kv_heads, seq_len, head_dim] for block-wise writing
            k_tensor_t = k_tensor.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]
            v_tensor_t = v_tensor.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]

            # Write block by block following the reference implementation
            # kv_cache_tensor shape: [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
            for b in range(allocate_blocks):
                block_start = b * seq_size_per_block
                block_end = min((b + 1) * seq_size_per_block, seq_len)
                block_len = block_end - block_start
                physical_block_id = int(block_ids[i, b].item())

                print(
                    f"  [DEBUG] Logical block {b} -> physical block {physical_block_id}: "
                    f"writing tokens [{block_start}:{block_end}), block_len={block_len}",
                    flush=True,
                )

                # Write K: block_cache[b, 0, :, 0:block_len, :] = k_tensor_t[:, block_start:block_end, :]
                kv_cache_tensor[physical_block_id, 0, :, 0:block_len, :] = k_tensor_t[
                    :, block_start:block_end, :
                ]
                # Write V: block_cache[b, 1, :, 0:block_len, :] = v_tensor_t[:, block_start:block_end, :]
                kv_cache_tensor[physical_block_id, 1, :, 0:block_len, :] = v_tensor_t[
                    :, block_start:block_end, :
                ]

            print(f"  write cache for batch {i} successfully\n", flush=True)

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
        rtol: float = 5e-3,
        atol: float = 5e-3,
        max_mismatch_rate: float = 0.0,
        use_packed_kv_cache: bool = False,
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
            rtol: Relative tolerance for the reference comparison
            atol: Absolute tolerance for the reference comparison
            max_mismatch_rate: Maximum fraction of elements allowed outside tolerance
            use_packed_kv_cache: Whether to expose KV cache as a packed 2D buffer
        """
        # For paged attention with prefix: create full QKV (prefix + new) for writing cache
        # For non-prefix cases: create QKV only with input_lengths
        if prefix_lengths is not None:
            total_tokens = sum(
                input_lengths[i] + prefix_lengths[i] for i in range(batch_size)
            )
            print(
                f"Creating QKV with prefix + new: {total_tokens} tokens "
                f"(prefix: {sum(prefix_lengths)}, new: {sum(input_lengths)})",
                flush=True,
            )
        else:
            total_tokens = sum(input_lengths)

        # CRITICAL: In padded mode, ONLY QKV needs padding; all length fields stay actual
        if use_padded:
            max_seq_len = max(input_lengths)
            qkv_size = batch_size * max_seq_len
            print(
                f"[Padded Mode] Creating QKV with padded size: {qkv_size} (batch_size={batch_size}, max_seq_len={max_seq_len})",
                flush=True,
            )
            print(
                f"[Padded Mode] Note: attn_inputs.total_tokens remains {total_tokens} (actual, not padded)",
                flush=True,
            )

            # Create padded QKV tensor with fixed layout: [seq0 + pad | seq1 + pad | ...]
            qkv_dim = (head_num + 2 * head_num_kv) * size_per_head
            qkv = torch.zeros(
                qkv_size, qkv_dim, dtype=attn_configs.dtype, device=self.device
            )

            # Fill in actual data for each sequence, leaving the rest as padding (zeros)
            for i, seq_len in enumerate(input_lengths):
                seq_start = i * max_seq_len
                seq_data = torch.randn(
                    seq_len, qkv_dim, dtype=attn_configs.dtype, device=self.device
                )
                qkv[seq_start : seq_start + seq_len] = seq_data

            print(
                f"[Padded Mode] QKV filled: {sum(input_lengths)} actual tokens out of {qkv_size} padded slots",
                flush=True,
            )
        else:
            qkv_size = total_tokens
            qkv = self._create_qkv_tensor(
                qkv_size,
                head_num,
                head_num_kv,
                size_per_head,
                dtype=attn_configs.dtype,
            )

        attn_inputs.dtype = get_typemeta(qkv)

        is_supported = attn_op.support(attn_configs, attn_inputs)
        print(f"{op_name} support check: {is_supported}", flush=True)

        if not is_supported:
            self.fail(f"{op_name} does not support this configuration")

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

        cache_dtype = (
            torch.float8_e4m3fn
            if attn_configs.kv_cache_dtype == KvCacheDataType.FP8
            else attn_configs.dtype
        )
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            seq_size_per_block,
            head_num_kv,
            size_per_head,
            dtype=cache_dtype,
        )

        try:
            # For paged attention tests: manually write ALL KV (prefix + new) to cache
            # This simulates the behavior of FusedRopeKVCache which writes KV to cache
            if prefix_lengths is not None:
                # Write ALL KV (prefix + new) to cache
                self._write_kv_cache(qkv, attn_inputs, total_kv_lengths, kv_cache)
                if use_packed_kv_cache:
                    kv_cache.kv_cache_base = kv_cache.kv_cache_base.flatten(1)

                # For paged FlashInfer TRT-LLM FMHA v2, extract Q from new tokens only.
                # Need to skip prefix part in QKV
                qkv_new_only = []
                offset = 0
                for i in range(batch_size):
                    total_len = input_lengths[i] + prefix_lengths[i]
                    prefix_len = prefix_lengths[i]
                    new_len = input_lengths[i]
                    # Skip prefix, take only new tokens
                    qkv_new_only.append(qkv[offset + prefix_len : offset + total_len])
                    offset += total_len
                qkv_new_only = torch.cat(
                    qkv_new_only, dim=0
                )  # [sum(input_lengths), qkv_dim]

                # Extract Q and transpose to [local_head_num, token_num, size_per_head]
                qkv_reshaped = qkv_new_only.view(
                    sum(input_lengths), head_num + 2 * head_num_kv, size_per_head
                )
                q = qkv_reshaped[
                    :, :head_num, :
                ]  # [sum(input_lengths), head_num, size_per_head]

                attn_input = (
                    q.contiguous()
                )  # [sum(input_lengths), local_head_num, size_per_head]
            else:
                # For non-paged or normal TRT: pass full QKV
                attn_input = qkv

            params = attn_op.prepare(attn_inputs)
            print_attn_inputs_detail(attn_inputs, qkv)
            self.assertIsNotNone(params, f"{op_name} prepare() returned None")

            # Forward pass: TRT attention reads KV from cache
            output = attn_op.forward(attn_input, kv_cache, params)
            expected_output_dim = head_num * size_per_head
            self.assertEqual(output.dtype, qkv.dtype)

            # Check output shape
            if use_padded:
                max_seq_len = max(input_lengths)
                expected_tokens = batch_size * max_seq_len
                print(
                    f"[Padded Mode] Expecting output shape: [{expected_tokens}, {expected_output_dim}]",
                    flush=True,
                )
            else:
                expected_tokens = sum(input_lengths)

            self.assertEqual(
                output.shape,
                (expected_tokens, expected_output_dim),
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

            # Extract actual data from padded tensors for comparison
            if use_padded:
                max_seq_len = max(input_lengths)
                # Extract non-padded QKV for reference computation
                qkv_for_ref = []
                for i, seq_len in enumerate(input_lengths):
                    seq_start = i * max_seq_len
                    qkv_for_ref.append(qkv[seq_start : seq_start + seq_len])
                qkv_for_ref = torch.cat(qkv_for_ref, dim=0)

                # Extract non-padded output for comparison
                output_for_compare = []
                for i, seq_len in enumerate(input_lengths):
                    seq_start = i * max_seq_len
                    output_for_compare.append(output[seq_start : seq_start + seq_len])
                output_for_compare = torch.cat(output_for_compare, dim=0)

                print(
                    f"[Padded Mode] Extracted {qkv_for_ref.shape[0]} actual tokens for reference",
                    flush=True,
                )
            else:
                qkv_for_ref = qkv
                output_for_compare = output

            if attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
                qkv_for_ref = qkv_for_ref.to(torch.float8_e4m3fn).to(attn_configs.dtype)

            print(f"Computing PyTorch reference for {op_name}...", flush=True)

            # For paged attention with prefix: compute reference with full QKV (prefix + new)
            # Then extract only new tokens from reference output
            if prefix_lengths is not None:
                ref_lengths = [
                    input_lengths[i] + prefix_lengths[i] for i in range(batch_size)
                ]
                print(
                    f"Computing reference with full lengths: {ref_lengths}", flush=True
                )
                ref_output_full = compute_pytorch_prefill_reference(
                    qkv_for_ref.clone(),
                    ref_lengths,
                    head_num,
                    head_num_kv,
                    size_per_head,
                    is_causal=attn_configs.is_causal,
                )
                # Extract only new tokens (input_lengths) from reference output
                # For each request: skip prefix part, only keep the new tokens part
                ref_output = []
                offset = 0
                for i in range(batch_size):
                    prefix_len = prefix_lengths[i]
                    new_len = input_lengths[i]
                    total_len = prefix_len + new_len

                    # Skip prefix, take only new tokens: [offset+prefix_len : offset+prefix_len+new_len]
                    ref_output.append(
                        ref_output_full[
                            offset + prefix_len : offset + prefix_len + new_len
                        ]
                    )
                    offset += total_len

                ref_output = torch.cat(ref_output, dim=0)
                print(
                    f"Extracted new tokens from reference: {ref_output.shape} (should be {sum(input_lengths)})",
                    flush=True,
                )
            else:
                ref_output = compute_pytorch_prefill_reference(
                    qkv_for_ref.clone(),
                    input_lengths,
                    head_num,
                    head_num_kv,
                    size_per_head,
                    is_causal=attn_configs.is_causal,
                )

            print(f"Comparing {op_name} output with PyTorch reference...", flush=True)
            print(
                f"output_for_compare.shape: {output_for_compare.shape}, ref_output.shape: {ref_output.shape}",
                flush=True,
            )

            if max_mismatch_rate > 0:
                assert_close_with_mismatch_tolerance(
                    output_for_compare,
                    ref_output,
                    rtol=rtol,
                    atol=atol,
                    max_mismatched_elements=int(max_mismatch_rate * ref_output.numel()),
                )
            else:
                compare_tensors(
                    output_for_compare,
                    ref_output,
                    rtol=rtol,
                    atol=atol,
                    name=f"{op_name} output (batch={batch_size}, input_lengths={input_lengths}{', padded' if use_padded else ''})",
                )

            print(f"✓ {op_name} correctness check passed!", flush=True)

        except Exception as e:
            self.fail(f"{op_name} forward pass failed: {e}")
