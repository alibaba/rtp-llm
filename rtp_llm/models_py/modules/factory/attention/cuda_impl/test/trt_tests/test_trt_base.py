"""Base class for TRT attention tests with shared utilities"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

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
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta, init_device


def compute_attention_per_head(
    q_tensor: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
    k_tensor: torch.Tensor,  # [total_kv_len, num_kv_heads, head_dim]
    v_tensor: torch.Tensor,  # [total_kv_len, num_kv_heads, head_dim]
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """
    手动计算每个head的attention结果

    公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    对于GQA: 每 group_size 个Q heads共享一个KV head
    """
    num_tokens = q_tensor.shape[0]
    total_kv_len = k_tensor.shape[0]
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / (head_dim**0.5)

    attention_outputs = []

    for token_idx in range(num_tokens):
        token_outputs = []

        for q_head_idx in range(num_q_heads):
            # GQA mapping
            kv_head_idx = q_head_idx // group_size

            # 获取Q, K, V
            q = q_tensor[token_idx, q_head_idx, :]  # [head_dim]
            k = k_tensor[:, kv_head_idx, :]  # [total_kv_len, head_dim]
            v = v_tensor[:, kv_head_idx, :]  # [total_kv_len, head_dim]

            # 计算 attention scores
            scores = torch.matmul(k, q) * scale

            # Apply causal mask
            causal_mask = torch.zeros_like(scores, dtype=torch.bool)
            causal_mask[token_idx + 1 :] = True
            scores = scores.masked_fill(causal_mask, float("-inf"))

            # Softmax and weighted sum
            attn_weights = F.softmax(scores, dim=0)
            output = torch.matmul(attn_weights, v)

            token_outputs.append(output)

        token_outputs = torch.stack(token_outputs, dim=0)  # [num_q_heads, head_dim]
        attention_outputs.append(token_outputs)

    attention_outputs = torch.stack(
        attention_outputs, dim=0
    )  # [num_tokens, num_q_heads, head_dim]
    return attention_outputs


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
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = (
                64
            )
            py_env_configs.device_resource_config.host_reserve_memory_bytes = 0

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

        print(
            f"[Non-padded] total_kvs={total_kvs}, max_blocks_per_seq={max_blocks_per_seq}, total_blocks={block_offset}",
            flush=True,
        )

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

        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
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

    ## 1. kv_cache: KVCache object with k_cache_base [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
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
            kv_cache.k_cache_base
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
        block_offset = 0
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

            # Print ORIGINAL K/V tensors (按 token x head 组织) - ALL DIMENSIONS
            print(
                f"\n  [ORIGINAL K/V] Before writing to cache (first 2 tokens, all KV heads, ALL dims):",
                flush=True,
            )
            for token_idx in range(min(seq_len, 2)):
                print(f"    Token {token_idx}:", flush=True)
                for kv_head_idx in range(num_kv_heads):
                    k_vals = k_tensor[token_idx, kv_head_idx, :]  # ALL dims
                    v_vals = v_tensor[token_idx, kv_head_idx, :]  # ALL dims
                    print(
                        f"      KV_Head {kv_head_idx}: K={k_vals.tolist()}, V={v_vals.tolist()}",
                        flush=True,
                    )

            # Transpose to [num_kv_heads, seq_len, head_dim] for block-wise writing
            k_tensor_t = k_tensor.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]
            v_tensor_t = v_tensor.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]

            # Write block by block following the reference implementation
            # kv_cache_tensor shape: [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
            for b in range(allocate_blocks):
                block_start = b * seq_size_per_block
                block_end = min((b + 1) * seq_size_per_block, seq_len)
                block_len = block_end - block_start

                print(
                    f"  [DEBUG] Block {b}: writing tokens [{block_start}:{block_end}), block_len={block_len}",
                    flush=True,
                )

                # Write K: block_cache[b, 0, :, 0:block_len, :] = k_tensor_t[:, block_start:block_end, :]
                kv_cache_tensor[block_offset + b, 0, :, 0:block_len, :] = k_tensor_t[
                    :, block_start:block_end, :
                ]
                # Write V: block_cache[b, 1, :, 0:block_len, :] = v_tensor_t[:, block_start:block_end, :]
                kv_cache_tensor[block_offset + b, 1, :, 0:block_len, :] = v_tensor_t[
                    :, block_start:block_end, :
                ]

            # Read back from cache and print (按 token x head 组织)
            print(
                f"\n  [READ BACK FROM CACHE] After writing (first 2 tokens, all KV heads):",
                flush=True,
            )
            for token_idx in range(min(seq_len, 2)):
                # Calculate which block and position within block
                block_idx = token_idx // seq_size_per_block
                pos_in_block = token_idx % seq_size_per_block

                print(
                    f"    Token {token_idx} (Block {block_idx}, Pos {pos_in_block}):",
                    flush=True,
                )
                for kv_head_idx in range(num_kv_heads):
                    # Read from: [block_offset + block_idx, K/V, kv_head_idx, pos_in_block, :]
                    k_cached = kv_cache_tensor[
                        block_offset + block_idx, 0, kv_head_idx, pos_in_block, :4
                    ]
                    v_cached = kv_cache_tensor[
                        block_offset + block_idx, 1, kv_head_idx, pos_in_block, :4
                    ]
                    print(
                        f"      KV_Head {kv_head_idx}: K={k_cached.tolist()}, V={v_cached.tolist()}",
                        flush=True,
                    )

            # Verify correctness by comparing
            print(f"\n  [VERIFICATION] Comparing original vs cached:", flush=True)
            for token_idx in range(min(seq_len, 2)):
                block_idx = token_idx // seq_size_per_block
                pos_in_block = token_idx % seq_size_per_block

                for kv_head_idx in range(num_kv_heads):
                    k_orig = k_tensor[token_idx, kv_head_idx, :]
                    v_orig = v_tensor[token_idx, kv_head_idx, :]
                    k_cached = kv_cache_tensor[
                        block_offset + block_idx, 0, kv_head_idx, pos_in_block, :
                    ]
                    v_cached = kv_cache_tensor[
                        block_offset + block_idx, 1, kv_head_idx, pos_in_block, :
                    ]

                    k_match = torch.allclose(k_orig, k_cached, rtol=1e-5)
                    v_match = torch.allclose(v_orig, v_cached, rtol=1e-5)
                    status = "✓" if (k_match and v_match) else "✗"
                    print(
                        f"    T{token_idx} KV_H{kv_head_idx}: K_match={k_match}, V_match={v_match} {status}",
                        flush=True,
                    )

            # 手动计算这个batch的attention结果
            print(f"\n  [MANUAL ATTENTION] for batch {i}:", flush=True)
            prefix_len = prefix_lengths[i].item()
            new_len = seq_len - prefix_len

            # 提取这个batch的Q (只要新tokens)
            # cache_tensor已经是 [seq_len, head_num + 2*num_kv_heads, head_dim]
            q_new = cache_tensor[
                prefix_len:, :head_num, :
            ]  # [new_len, head_num, head_dim]

            # 提取这个batch的全部KV (prefix + new)
            k_full_batch = cache_tensor[
                :, head_num : head_num + num_kv_heads, :
            ]  # [seq_len, num_kv_heads, head_dim]
            v_full_batch = cache_tensor[
                :, head_num + num_kv_heads : head_num + 2 * num_kv_heads, :
            ]

            # 调用手动计算
            manual_output_batch = compute_attention_per_head(
                q_tensor=q_new,
                k_tensor=k_full_batch,
                v_tensor=v_full_batch,
                num_q_heads=head_num,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )

            # 打印手动计算结果 (前2个新tokens)
            print(
                f"    Manual results (first {min(new_len, 2)} new tokens):", flush=True
            )
            for token_idx in range(min(new_len, 2)):
                print(
                    f"      Token {token_idx} (global offset={offset - seq_len + prefix_len + token_idx}):",
                    flush=True,
                )
                for h in range(head_num):
                    out = manual_output_batch[token_idx, h, :4]
                    print(f"        Head {h}: {out.tolist()}", flush=True)

            block_offset += allocate_blocks
            print(f"  write cache for batch {i} successfully\n", flush=True)

        # Make same KV head's different tokens have identical K/V
        # For each KV head: Token0's K/V = Token1's K/V = ...
        # This simplifies attention: weights become uniform, output = average of V values
        print(
            f"\n[DEBUG] Making each KV head's tokens have identical K/V for testing...",
            flush=True,
        )
        num_kv_heads = kv_cache_tensor.shape[2]
        seq_size_per_block = kv_cache_tensor.shape[3]

        # for kv_head_idx in range(num_kv_heads):
        #     # For this KV head, make all tokens have same K (copy from token 0)
        #     for token_idx in range(1, seq_size_per_block):
        #         kv_cache_tensor[:, 0, kv_head_idx, token_idx, :] = kv_cache_tensor[:, 0, kv_head_idx, 0, :]

        #     # For this KV head, make all tokens have same V (copy from token 0)
        #     for token_idx in range(1, seq_size_per_block):
        #         kv_cache_tensor[:, 1, kv_head_idx, token_idx, :] = kv_cache_tensor[:, 1, kv_head_idx, 0, :]

        print(f"  Each KV head's tokens now have identical K and V values", flush=True)
        print(f"  (But different KV heads still have different values)", flush=True)

        # Swap token0 and token1 for KV_Head 0 to test if kernel accesses wrong token
        print(f"\n[DEBUG] Swapping token0 and token1 for KV_Head0...", flush=True)
        # Save token0's data
        # k_head0_token0 = kv_cache_tensor[:, 0, 0, 0, :].clone()
        # v_head0_token0 = kv_cache_tensor[:, 1, 0, 0, :].clone()

        # # token0 = token1
        # kv_cache_tensor[:, 0, 0, 0, :] = kv_cache_tensor[:, 0, 0, 1, :]
        # kv_cache_tensor[:, 1, 0, 0, :] = kv_cache_tensor[:, 1, 0, 1, :]

        # # token1 = old token0

        # kv_cache_tensor[:, 1, 0, 0, :] = 1.0
        # kv_cache_tensor[:, 0, 0, 1, :] = 1.0
        # kv_cache_tensor[:, 1, 0, 1, :] = 1.0
        # Manual attention calculation for debugging
        print(f"\n[MANUAL ATTENTION CALCULATION] Q_Head0 with KV_Head0", flush=True)
        print(f"KV cache data after modifications:", flush=True)
        print(
            f"  KV_Head0, Token0: K={kv_cache_tensor[0, 0, 0, 0, :4].tolist()}, V={kv_cache_tensor[0, 1, 0, 0, :4].tolist()}",
            flush=True,
        )
        print(
            f"  KV_Head0, Token1: K={kv_cache_tensor[0, 0, 0, 1, :4].tolist()}, V={kv_cache_tensor[0, 1, 0, 1, :4].tolist()}",
            flush=True,
        )
        if num_kv_heads > 1:
            print(
                f"  KV_Head1, Token0: K={kv_cache_tensor[0, 0, 1, 0, :4].tolist()}, V={kv_cache_tensor[0, 1, 1, 0, :4].tolist()}",
                flush=True,
            )

        # Extract Q tensor once (avoid duplicate views)
        # qkv_full: [total_tokens, (head_num + 2*num_kv_heads) * head_dim]
        # Q is the first head_num * head_dim elements
        # Layout: Q (first head_num) | K (next num_kv_heads) | V (last num_kv_heads)
        q_tensor = qkv_full.view(total_tokens, head_num + 2 * num_kv_heads, head_dim)[
            :, :head_num, :
        ]

        # Optional: modify Q for testing (currently commented out)
        # q_t1_h0 = q_tensor[0, 0, :]  # [head_dim]
        # q_t1_h0[:] = 0.77

        # Diagnose Q tensor reading bug: calculate attention for ALL Q heads with KV_Head0
        # to see which Q head TRT is actually using
        if total_tokens >= 2:
            print(
                f"\n[DIAGNOSIS] Calculate attention for ALL Q heads @ KV_Head0",
                flush=True,
            )
            print(
                f"  This helps identify which Q head TRT kernel is actually reading",
                flush=True,
            )

            # Get K and V from cache for KV_Head 0
            k_kv_head0 = kv_cache_tensor[
                0, 0, 0, :2, :
            ]  # [2, head_dim] (token0, token1)
            v_kv_head0 = kv_cache_tensor[0, 1, 0, :2, :]  # [2, head_dim]
            scale = 1.0 / (head_dim**0.5)

            print(f"\n  KV_Head0 data:", flush=True)
            print(f"    K[token0][:4] = {k_kv_head0[0, :4].tolist()}", flush=True)
            print(f"    K[token1][:4] = {k_kv_head0[1, :4].tolist()}", flush=True)
            print(f"    V[token0][:4] = {v_kv_head0[0, :4].tolist()}", flush=True)
            print(f"    V[token1][:4] = {v_kv_head0[1, :4].tolist()}", flush=True)

            # Calculate attention for Token1 with each Q head
            for q_head_idx in range(min(head_num, 4)):
                q_t1 = q_tensor[1, q_head_idx, :]  # [head_dim]
                attn_scores = (
                    torch.matmul(q_t1.unsqueeze(0), k_kv_head0.transpose(0, 1)) * scale
                )
                attn_weights = torch.softmax(attn_scores, dim=-1)
                expected_output = torch.matmul(attn_weights, v_kv_head0)[0]

                print(f"\n  Token1 Q_Head{q_head_idx} @ KV_Head0:", flush=True)
                print(f"    Q[:4] = {q_t1[:4].tolist()}", flush=True)
                print(f"    Attn weights = {attn_weights[0].tolist()}", flush=True)
                print(
                    f"    Expected output[:8] = {expected_output[:8].tolist()}",
                    flush=True,
                )

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

        is_supported = attn_op.support(attn_inputs)
        print(f"{op_name} support check: {is_supported}", flush=True)

        # if not is_supported:
        #     self.fail(f"{op_name} does not support this configuration")

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
            # For paged attention tests: manually write ALL KV (prefix + new) to cache
            # This simulates the behavior of FusedRopeKVCache which writes KV to cache
            if prefix_lengths is not None:
                # Write ALL KV (prefix + new) to cache
                self._write_kv_cache(qkv, attn_inputs, total_kv_lengths, kv_cache)

                # For PAGED_TRT_V2: extract Q from ONLY new tokens part and transpose
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

                # Print Q tensor before transpose (按 token x head 组织)
                print(
                    f"\n[Q TENSOR] Before transpose (first 2 tokens, first 4 Q heads):",
                    flush=True,
                )
                for token_idx in range(min(sum(input_lengths), 2)):
                    print(f"  Token {token_idx}:", flush=True)
                    for q_head_idx in range(min(head_num, 4)):
                        q_vals = q[token_idx, q_head_idx, :4]  # First 4 dims
                        print(f"    Q_Head {q_head_idx}: {q_vals.tolist()}", flush=True)

                # FIX: TRT kernel expects Q in [token, head, dim] layout, NOT [head, token, dim]
                # Keep Q as [token, head, dim] without transpose
                print(
                    f"\n[FIX] Keeping Q in [token, head, dim] layout as expected by TRT kernel",
                    flush=True,
                )
                q_transposed = (
                    q.contiguous()
                )  # [sum(input_lengths), local_head_num, size_per_head]

                # Print Q tensor final layout (no transpose, keep as [token, head, dim])
                print(
                    f"\n[Q TENSOR] Final layout [token, head, dim] (first 2 tokens, first 4 heads):",
                    flush=True,
                )
                for token_idx in range(min(sum(input_lengths), 2)):
                    print(f"  Token {token_idx}:", flush=True)
                    for q_head_idx in range(min(head_num, 4)):
                        q_vals = q_transposed[token_idx, q_head_idx, :4]  # First 4 dims
                        print(f"    Q_Head {q_head_idx}: {q_vals.tolist()}", flush=True)

                attn_input = q_transposed
            else:
                # For non-paged or normal TRT: pass full QKV
                attn_input = qkv

            params = attn_op.prepare(attn_inputs)
            print_attn_inputs_detail(attn_inputs, qkv)
            self.assertIsNotNone(params, f"{op_name} prepare() returned None")

            # MANUAL ATTENTION CALCULATION for debugging GQA KV head indexing
            # if prefix_lengths is not None and sum(input_lengths) >= 2 and head_num_kv >= 2:
            #     print(f"\n[MANUAL ATTENTION] Token1 Q_Head0 with KV_Head0 (with causal mask):", flush=True)

            #     # Get KV cache tensor (contains both K and V)
            #     # Shape: [total_blocks, 2, num_kv_heads, seq_size_per_block, size_per_head]
            #     # Second dim: 0=K, 1=V
            #     kv_cache_tensor = kv_cache.k_cache_base
            #     print(f"  kv_cache_tensor shape: {kv_cache_tensor.shape if kv_cache_tensor is not None else None}", flush=True)

            #     if kv_cache_tensor is None:
            #         print(f"  [WARNING] kv_cache_tensor is None, skipping manual attention calculation", flush=True)
            #     else:
            #         # Extract K and V from first block (which contains token 0 and token 1)
            #         # k_block: [num_kv_heads, seq_size_per_block, size_per_head]
            #         # v_block: [num_kv_heads, seq_size_per_block, size_per_head]
            #         k_block = kv_cache_tensor[0, 0, :, :, :]  # Block 0, K (dim=0)
            #         v_block = kv_cache_tensor[0, 1, :, :, :]  # Block 0, V (dim=1)

            #         # Get Q for token 1, head 0
            #         q_t1_h0 = q[1, 0, :]  # [size_per_head]
            #         print(f"  Q[token1, head0] shape: {q_t1_h0.shape}, values[:8]: {q_t1_h0[:8].tolist()}", flush=True)

            #         # Get K and V for KV head 0 and 1 (token 0 and 1)
            #         k_kv_head0 = k_block[0, :2, :]  # [2, size_per_head] (token 0 and 1)
            #         v_kv_head0 = v_block[0, :2, :]  # [2, size_per_head]
            #         v_kv_head1 = v_block[1, :2, :]  # [2, size_per_head]

            #         print(f"\n  KV_Head 0 data:", flush=True)
            #         print(f"    K[token0] values[:8]: {k_kv_head0[0, :8].tolist()}", flush=True)
            #         print(f"    K[token1] values[:8]: {k_kv_head0[1, :8].tolist()}", flush=True)
            #         print(f"    V[token0] values[:8]: {v_kv_head0[0, :8].tolist()}", flush=True)
            #         print(f"    V[token1] values[:8]: {v_kv_head0[1, :8].tolist()}", flush=True)

            #         print(f"\n  KV_Head 1 data (for comparison):", flush=True)
            #         print(f"    V[token0] values[:8]: {v_kv_head1[0, :8].tolist()}", flush=True)
            #         print(f"    V[token1] values[:8]: {v_kv_head1[1, :8].tolist()}", flush=True)

            #         # Calculate attention scores: Q @ K^T
            #         scale = 1.0 / math.sqrt(size_per_head)
            #         attn_scores = torch.matmul(q_t1_h0.unsqueeze(0), k_kv_head0.transpose(0, 1))  # [1, 2]
            #         attn_scores = attn_scores * scale
            #         print(f"\n  Attention scores (before causal mask & softmax): {attn_scores[0].tolist()}", flush=True)

            #         # Apply causal mask: Token 1 can see Token 0 and Token 1
            #         causal_mask = torch.tensor([0.0, 0.0], dtype=attn_scores.dtype, device=attn_scores.device)
            #         attn_scores_masked = attn_scores + causal_mask
            #         print(f"  Attention scores (after causal mask, no change): {attn_scores_masked[0].tolist()}", flush=True)

            #         # Apply softmax
            #         attn_weights = torch.softmax(attn_scores_masked, dim=-1)  # [1, 2]
            #         print(f"  Attention weights (after softmax): {attn_weights[0].tolist()}", flush=True)

            #         # Calculate output: weights @ V from KV_Head0
            #         attn_output_h0 = torch.matmul(attn_weights, v_kv_head0)  # [1, size_per_head]
            #         print(f"\n  Manual attention output for Q_Head0 (using KV_Head0): {attn_output_h0[0].tolist()}", flush=True)

            #         # ========== Now calculate for Q_Head2 (should use KV_Head1) ==========
            #         print(f"\n\n[MANUAL ATTENTION] Token1 Q_Head2 with KV_Head1 (with causal mask):", flush=True)

            #         # Get Q for token 1, head 2
            #         q_t1_h2 = q[1, 2, :]  # [size_per_head]
            #         print(f"  Q[token1, head2] shape: {q_t1_h2.shape}, values[:8]: {q_t1_h2[:8].tolist()}", flush=True)

            #         # Get K and V for KV head 1 (token 0 and 1)
            #         k_kv_head1 = k_block[1, :2, :]  # [2, size_per_head]

            #         print(f"\n  Using KV_Head 1 (correct for Q_Head2):", flush=True)
            #         print(f"    K[token0] values[:8]: {k_kv_head1[0, :8].tolist()}", flush=True)
            #         print(f"    K[token1] values[:8]: {k_kv_head1[1, :8].tolist()}", flush=True)
            #         print(f"    V[token0] values[:8]: {v_kv_head1[0, :8].tolist()}", flush=True)
            #         print(f"    V[token1] values[:8]: {v_kv_head1[1, :8].tolist()}", flush=True)

            #         # Calculate attention scores: Q @ K^T
            #         attn_scores_h2 = torch.matmul(q_t1_h2.unsqueeze(0), k_kv_head1.transpose(0, 1))  # [1, 2]
            #         attn_scores_h2 = attn_scores_h2 * scale
            #         print(f"\n  Attention scores (before causal mask & softmax): {attn_scores_h2[0].tolist()}", flush=True)

            #         # Apply causal mask
            #         attn_scores_h2_masked = attn_scores_h2 + causal_mask
            #         print(f"  Attention scores (after causal mask): {attn_scores_h2_masked[0].tolist()}", flush=True)

            #         # Apply softmax
            #         attn_weights_h2 = torch.softmax(attn_scores_h2_masked, dim=-1)  # [1, 2]
            #         print(f"  Attention weights (after softmax): {attn_weights_h2[0].tolist()}", flush=True)

            #         # Calculate output: weights @ V from KV_Head1
            #         attn_output_h2 = torch.matmul(attn_weights_h2, v_kv_head1)  # [1, size_per_head]
            #         print(f"\n  Manual attention output for Q_Head2 (using KV_Head1): {attn_output_h2[0].tolist()}", flush=True)

            #         print(f"\n  Now compare with TRT outputs...", flush=True)

            # Forward pass: TRT attention reads KV from cache
            output = attn_op.forward(attn_input, kv_cache, params)
            expected_output_dim = head_num * size_per_head

            # Print TRT actual output for comparison
            if prefix_lengths is not None and sum(input_lengths) >= 2:
                output_reshaped = output.view(
                    sum(input_lengths), head_num, size_per_head
                )
                trt_t1_h0 = output_reshaped[1, 0, :]  # Token 1, Q Head 0

                print(f"\n[TRT ACTUAL OUTPUT] Token1 Q_Head0:", flush=True)
                print(f"  Actual output[:4] = {trt_t1_h0[:4].tolist()}", flush=True)
                print(f"  Actual output (full) = {trt_t1_h0.tolist()}", flush=True)

                if head_num > 1:
                    trt_t1_h1 = output_reshaped[1, 1, :]  # Token 1, Q Head 1
                    print(f"\n[TRT ACTUAL OUTPUT] Token1 Q_Head1:", flush=True)
                    print(f"  Actual output[:4] = {trt_t1_h1[:4].tolist()}", flush=True)

            # Print TRT output for token1 head0 and head2 for comparison with manual calculation
            # if prefix_lengths is not None and sum(input_lengths) >= 2 and head_num_kv >= 2:
            #     # Output shape: [sum(input_lengths), head_num * size_per_head]
            #     # Reshape to [sum(input_lengths), head_num, size_per_head]
            #     output_reshaped = output.view(sum(input_lengths), head_num, size_per_head)

            #     trt_token1_head0 = output_reshaped[1, 0, :]  # Token 1, Q Head 0 (should use KV Head 0)
            #     trt_token1_head2 = output_reshaped[1, 2, :]  # Token 1, Q Head 2 (should use KV Head 1)

            #     print(f"\n[TRT OUTPUT] Token1 Q_Head0 (should use KV_Head0):", flush=True)
            #     print(f"  TRT output: {trt_token1_head0.tolist()}", flush=True)
            #     print(f"  Expected (from manual calc above): weighted sum of KV_Head0's V values", flush=True)

            #     print(f"\n[TRT OUTPUT] Token1 Q_Head2 (should use KV_Head1):", flush=True)
            #     print(f"  TRT output: {trt_token1_head2.tolist()}", flush=True)
            #     print(f"  Expected (from manual calc above): weighted sum of KV_Head1's V values", flush=True)

            #     print(f"\n  ========== DIAGNOSIS ==========", flush=True)
            #     print(f"  If Q_Head0 output matches manual calc using KV_Head0: ✓ CORRECT", flush=True)
            #     print(f"  If Q_Head0 output matches manual calc using KV_Head1: ✗ BUG (diagonal pattern)", flush=True)
            #     print(f"  If Q_Head2 output matches manual calc using KV_Head1: ✓ CORRECT", flush=True)
            #     print(f"  If Q_Head2 output matches manual calc using KV_Head0: ✗ BUG", flush=True)

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
                )

            print(f"Comparing {op_name} output with PyTorch reference...", flush=True)
            print(
                f"output_for_compare.shape: {output_for_compare.shape}, ref_output.shape: {ref_output.shape}",
                flush=True,
            )

            # Detailed per-token comparison for debugging
            output_reshaped = output_for_compare.view(
                sum(input_lengths), head_num, size_per_head
            )
            ref_reshaped = ref_output.view(sum(input_lengths), head_num, size_per_head)

            # Add analysis: Why does 1 token work but multi-token shows diagonal pattern?
            # Hypothesis: TRT kernel uses token_idx instead of kv_head_idx somewhere
            print(f"\n[ANALYSIS] Testing configuration:", flush=True)
            print(f"  head_num (Q heads): {head_num}", flush=True)
            print(f"  head_num_kv (KV heads): {head_num_kv}", flush=True)
            print(f"  GQA group_size: {head_num // head_num_kv}", flush=True)
            print(f"  num_tokens: {sum(input_lengths)}", flush=True)

            print(
                f"\n[DEBUG] Per-token per-head comparison (ALL tokens, ALL heads):",
                flush=True,
            )

            # Create a match matrix to see the pattern
            match_matrix = []
            for token_idx in range(sum(input_lengths)):
                token_matches = []
                for head_idx in range(head_num):
                    out_head = output_reshaped[token_idx, head_idx, :]
                    ref_head = ref_reshaped[token_idx, head_idx, :]
                    diff = (out_head - ref_head).abs().max().item()
                    is_match = diff < 0.01
                    token_matches.append("✓" if is_match else "✗")
                match_matrix.append(token_matches)

            print(f"\n  Match Matrix (rows=tokens, cols=heads):", flush=True)
            print(
                f"  {'':4s} " + " ".join([f"H{i:2d}" for i in range(head_num)]),
                flush=True,
            )
            for token_idx, matches in enumerate(match_matrix):
                print(f"  T{token_idx:2d}: " + "  ".join(matches), flush=True)

            # Show detailed values for first 4 tokens, first 4 heads
            print(
                f"\n[DEBUG] Detailed values (first 4 tokens, first 4 heads):",
                flush=True,
            )
            for token_idx in range(min(sum(input_lengths), 4)):
                print(f"\n  Token {token_idx}:", flush=True)
                for head_idx in range(min(head_num, 4)):
                    out_head = output_reshaped[token_idx, head_idx, :4]  # First 4 dims
                    ref_head = ref_reshaped[token_idx, head_idx, :4]
                    diff = (out_head - ref_head).abs().max().item()
                    match = "✓" if diff < 0.01 else "✗"
                    print(
                        f"    Head {head_idx}: out={out_head.tolist()}, ref={ref_head.tolist()}, diff={diff:.4f} {match}",
                        flush=True,
                    )

            print(f"\noutput_for_compare: {output_for_compare}")
            print(f"ref_output: {ref_output}")
            compare_tensors(
                output_for_compare,
                ref_output,
                rtol=5e-3,
                atol=5e-3,
                name=f"{op_name} output (batch={batch_size}, input_lengths={input_lengths}{', padded' if use_padded else ''})",
            )

            print(f"✓ {op_name} correctness check passed!", flush=True)

        except Exception as e:
            self.fail(f"{op_name} forward pass failed: {e}")
