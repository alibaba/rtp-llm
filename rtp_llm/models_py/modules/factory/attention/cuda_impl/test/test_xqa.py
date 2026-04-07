import logging
import unittest
from typing import List

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQOut,
    PyAttentionInputs,
    XQAAttnOp,
    XQAParams,
    XQASpecAttnOp,
    get_typemeta,
    init_device,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestXQAAttnOp(BaseAttentionTest):
    """Test suite for XQAAttnOp with correctness verification and support testing"""

    def _create_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for decode"""
        return self._create_attention_inputs_base(
            batch_size=batch_size,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )

    def _create_spec_attention_inputs(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        q_len_per_req: int,
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        final_sequence_lengths = [
            prefix_len + q_len_per_req for prefix_len in prefix_lengths
        ]

        attn_inputs.is_prefill = True
        attn_inputs.is_target_verify = True
        attn_inputs.sequence_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.input_lengths = torch.full(
            (batch_size,), q_len_per_req, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, final_sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        decode_cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * q_len_per_req,
            q_len_per_req,
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs.cu_seqlens = decode_cu_seqlens
        attn_inputs.decode_cu_seqlens_d = decode_cu_seqlens
        cu_kv_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        cu_kv_seqlens[1:] = torch.tensor(
            final_sequence_lengths, dtype=torch.int32, device=self.device
        ).cumsum(0)
        attn_inputs.cu_kv_seqlens = cu_kv_seqlens
        attn_inputs.total_tokens = batch_size * q_len_per_req
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=torch.bfloat16))
        return attn_inputs

    def _compute_flashinfer_spec_decode_reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        prefix_lengths: List[int],
        block_id_list: List[List[int]],
        seq_size_per_block: int,
    ) -> torch.Tensor:
        batch_size, q_len_per_req, _, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        num_heads = q.shape[2]
        group_size = num_heads // num_kv_heads
        scale = head_dim**-0.5
        ref_outputs = []

        for batch_idx in range(batch_size):
            block_ids = block_id_list[batch_idx]
            k_blocks = k_cache[block_ids]
            v_blocks = v_cache[block_ids]
            k_seq = (
                k_blocks.permute(1, 0, 2, 3)
                .reshape(num_kv_heads, -1, head_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            v_seq = (
                v_blocks.permute(1, 0, 2, 3)
                .reshape(num_kv_heads, -1, head_dim)
                .permute(1, 0, 2)
                .contiguous()
            )

            token_outputs = []
            for token_idx in range(q_len_per_req):
                seq_len = prefix_lengths[batch_idx] + token_idx + 1
                k_token = k_seq[:seq_len]
                v_token = v_seq[:seq_len]
                if group_size > 1:
                    k_token = k_token.repeat_interleave(group_size, dim=1)
                    v_token = v_token.repeat_interleave(group_size, dim=1)

                scores = (
                    torch.einsum(
                        "hd,thd->ht",
                        q[batch_idx, token_idx].float(),
                        k_token.float(),
                    )
                    * scale
                )
                attn = torch.softmax(scores, dim=-1)
                token_outputs.append(
                    torch.einsum("ht,thd->hd", attn, v_token.float()).to(q.dtype)
                )
            ref_outputs.append(torch.stack(token_outputs, dim=0))

        return torch.stack(ref_outputs, dim=0)

    def _test_decode_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
    ):
        """Test decode correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            data_type="bf16",
        )

        attn_inputs = self._create_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create XQAAttnOp instance
        attn_op = XQAAttnOp(config.attn_configs)

        # Test support function
        is_supported = attn_op.support(attn_inputs)
        logging.info(f"XQAAttnOp support check: {is_supported}")

        if not is_supported:
            logging.warning(
                f"XQAAttnOp does not support this configuration, skipping correctness test"
            )
            return

        # Prepare parameters
        params_base = attn_op.prepare(attn_inputs)
        params = XQAParams() if not isinstance(params_base, XQAParams) else params_base

        # Create query input [batch_size, head_num, head_dim]
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(batch_size, local_head_num, config.size_per_head)

        # Create KV cache
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=torch.float16,
        )

        # Forward pass through XQAAttnOp
        output = attn_op.forward(q, kv_cache, params)
        torch.cuda.synchronize()

        # XQA output shape: [batch_size, head_num * head_dim]
        # Need to reshape to [batch_size, head_num, head_dim] to match reference
        output = output.reshape(batch_size, local_head_num, config.size_per_head)

        # Generate block_id_list from attn_inputs for reference computation
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )

        # Compute reference outputs using flashinfer
        ref_output_stacked = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )

        # Compare outputs
        compare_tensors(
            output,
            ref_output_stacked,
            rtol=1e-2,
            atol=1e-2,
            name=f"XQA Decode output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def _test_spec_decode_correctness(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        q_len_per_req: int,
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
    ):
        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            data_type="bf16",
            kv_cache_dtype=KvCacheDataType.FP8,
        )
        config.attn_configs.need_rope_kv_cache = False
        final_sequence_lengths = [
            prefix_len + q_len_per_req for prefix_len in prefix_lengths
        ]
        attn_inputs = self._create_spec_attention_inputs(
            batch_size=batch_size,
            prefix_lengths=prefix_lengths,
            q_len_per_req=q_len_per_req,
            seq_size_per_block=config.seq_size_per_block,
        )

        attn_op = XQASpecAttnOp(config.attn_configs)
        is_supported = attn_op.support(attn_inputs)
        logging.info(f"XQASpecAttnOp support check: {is_supported}")

        if not is_supported:
            logging.warning(
                "XQASpecAttnOp does not support this configuration, skipping correctness test"
            )
            return

        params_base = attn_op.prepare(attn_inputs)
        params = XQAParams() if not isinstance(params_base, XQAParams) else params_base
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        total_tokens = batch_size * q_len_per_req
        q = self._create_query_tensor(
            total_tokens,
            local_head_num,
            config.size_per_head,
            dtype=config.attn_configs.dtype,
        )
        q_4d = q.reshape(
            batch_size, q_len_per_req, local_head_num, config.size_per_head
        )
        total_blocks = self._calculate_total_blocks(
            final_sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=torch.float8_e4m3fn,
        )
        block_id_list = self._generate_block_id_list(
            attn_inputs, final_sequence_lengths, config.seq_size_per_block
        )
        output = attn_op.forward(q_4d, kv_cache, params).reshape(
            batch_size, q_len_per_req, local_head_num, config.size_per_head
        )
        ref_output = self._compute_flashinfer_spec_decode_reference(
            q_4d,
            k_cache,
            v_cache,
            prefix_lengths,
            block_id_list,
            config.seq_size_per_block,
        )
        torch.cuda.synchronize()
        compare_tensors(
            output,
            ref_output,
            rtol=1e-1,
            atol=3e-1,
            name=(
                f"XQA Spec Decode output (batch={batch_size}, prefix_lens={prefix_lengths}, "
                f"q_len_per_req={q_len_per_req})"
            ),
        )

    def test_support(self):
        """Test XQAAttnOp support function comprehensively

        Based on CudaXqa.cc supportXqa function:
        - input_type: BF16 or FP16
        - output_type: BF16, FP16, or FP8_E4M3
        - kv_cache_type: BF16, FP16, or FP8_E4M3
        - group_size <= 16
        - head_dim: 64, 128, or 256
        - page_size: 16, 32, 64, or 128
        """
        logging.info("\n=== Testing XQAAttnOp support() functionality ===")

        # Test SUPPORTED configurations
        logging.info("\n--- Testing SUPPORTED configurations ---")

        supported_cases = [
            # (head_num, head_num_kv, size_per_head, seq_size_per_block, data_type, description)
            (
                32,
                8,
                128,
                64,
                "fp16",
                "Standard config: GQA, head_dim=128, page_size=64",
            ),
            (32, 8, 256, 64, "fp16", "Large head_dim: 256"),
            (32, 8, 64, 64, "fp16", "Small head_dim: 64"),
            (32, 32, 128, 64, "fp16", "MHA: group_size=1"),
            (32, 2, 128, 64, "fp16", "Large group_size: 16"),
            (32, 8, 128, 16, "fp16", "Small page_size: 16"),
            (32, 8, 128, 32, "fp16", "page_size: 32"),
            (32, 8, 128, 128, "fp16", "Large page_size: 128"),
            (64, 4, 128, 64, "fp16", "GQA-16: group_size=16"),
            (128, 8, 256, 128, "fp16", "Large head_num with head_dim=256"),
        ]

        supported_count = 0
        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in supported_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}, dtype={data_type}\n"
                f"    → Support: {is_supported} {'✓' if is_supported else '✗ UNEXPECTED'}"
            )

            if is_supported:
                supported_count += 1
            else:
                logging.warning(f"    ⚠️  Expected SUPPORTED but got NOT SUPPORTED")

        logging.info(f"\nSupported cases: {supported_count}/{len(supported_cases)}")

        # Test UNSUPPORTED configurations
        logging.info("\n--- Testing UNSUPPORTED configurations ---")

        unsupported_cases = [
            # (head_num, head_num_kv, size_per_head, seq_size_per_block, data_type, description)
            (32, 1, 128, 64, "fp16", "group_size=32 > 16: UNSUPPORTED"),
            (64, 1, 128, 64, "fp16", "group_size=64 > 16: UNSUPPORTED"),
            (32, 8, 96, 64, "fp16", "head_dim=96 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 192, 64, "fp16", "head_dim=192 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 512, 64, "fp16", "head_dim=512 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 128, 8, "fp16", "page_size=8 not in {16,32,64,128}: UNSUPPORTED"),
            (32, 8, 128, 48, "fp16", "page_size=48 not in {16,32,64,128}: UNSUPPORTED"),
            (
                32,
                8,
                128,
                256,
                "fp16",
                "page_size=256 not in {16,32,64,128}: UNSUPPORTED",
            ),
        ]

        unsupported_count = 0
        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in unsupported_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}, dtype={data_type}\n"
                f"    → Support: {is_supported} {'✗ UNEXPECTED' if is_supported else '✓'}"
            )

            if not is_supported:
                unsupported_count += 1
            else:
                logging.warning(f"    ⚠️  Expected UNSUPPORTED but got SUPPORTED")

        logging.info(
            f"\nUnsupported cases correctly rejected: {unsupported_count}/{len(unsupported_cases)}"
        )

        # Test boundary cases
        logging.info("\n--- Testing BOUNDARY cases ---")

        boundary_cases = [
            # Edge cases for group_size
            (
                32,
                2,
                128,
                64,
                "fp16",
                "group_size=16 (max allowed): SHOULD BE SUPPORTED",
            ),
            (
                34,
                2,
                128,
                64,
                "fp16",
                "group_size=17 (just over limit): SHOULD BE UNSUPPORTED",
            ),
            # Edge cases for head_dim
            (32, 8, 64, 64, "fp16", "head_dim=64 (min): SHOULD BE SUPPORTED"),
            (32, 8, 256, 64, "fp16", "head_dim=256 (max): SHOULD BE SUPPORTED"),
            # Edge cases for page_size
            (32, 8, 128, 16, "fp16", "page_size=16 (min): SHOULD BE SUPPORTED"),
            (32, 8, 128, 128, "fp16", "page_size=128 (max): SHOULD BE SUPPORTED"),
        ]

        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in boundary_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}\n"
                f"    → Support: {is_supported}"
            )

        logging.info("\n=== XQAAttnOp support() testing completed ===")

    def test_spec_support(self):
        """Test XQASpecAttnOp support is only enabled for target verify inputs"""
        logging.info("\n=== Testing XQASpecAttnOp support() functionality ===")
        config = self._create_config(
            data_type="bf16", kv_cache_dtype=KvCacheDataType.FP8
        )
        decode_attn_op = XQAAttnOp(config.attn_configs)
        spec_attn_op = XQASpecAttnOp(config.attn_configs)

        decode_attn_inputs = self._create_attention_inputs(
            batch_size=2,
            sequence_lengths=[128, 256],
            seq_size_per_block=config.seq_size_per_block,
        )
        self.assertTrue(decode_attn_op.support(decode_attn_inputs))
        self.assertFalse(spec_attn_op.support(decode_attn_inputs))

        spec_attn_inputs = self._create_spec_attention_inputs(
            batch_size=2,
            prefix_lengths=[124, 252],
            q_len_per_req=4,
            seq_size_per_block=config.seq_size_per_block,
        )
        config.attn_configs.need_rope_kv_cache = False
        self.assertFalse(decode_attn_op.support(spec_attn_inputs))
        self.assertTrue(spec_attn_op.support(spec_attn_inputs))

        non_fp8_config = self._create_config(data_type="bf16")
        self.assertFalse(
            XQASpecAttnOp(non_fp8_config.attn_configs).support(spec_attn_inputs)
        )

    def test_support_functionality(self):
        """Test XQAAttnOp support function with various configurations"""
        logging.info("\n=== Testing XQAAttnOp support functionality ===")

        test_cases = [
            # (head_num, head_num_kv, size_per_head, batch_size, seq_lens, description)
            (32, 8, 128, 1, [128], "Single batch, standard config"),
            (32, 8, 128, 4, [64, 128, 256, 512], "Multi-batch, varying lengths"),
            (32, 32, 128, 2, [100, 200], "MHA config"),
            (32, 4, 128, 2, [100, 200], "GQA-8 config"),
            (32, 8, 256, 2, [100, 200], "256 head dim"),
            (32, 8, 64, 2, [100, 200], "64 head dim"),
        ]

        for (
            head_num,
            head_num_kv,
            size_per_head,
            batch_size,
            seq_lens,
            desc,
        ) in test_cases:
            logging.info(f"\n--- Testing: {desc} ---")
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size, seq_lens, config.seq_size_per_block
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            logging.info(
                f"  Config: head_num={head_num}, head_num_kv={head_num_kv}, "
                f"size_per_head={size_per_head}"
            )
            logging.info(f"  Batch size: {batch_size}, seq_lens: {seq_lens}")
            logging.info(f"  Support result: {is_supported}")

    def test_single_batch_decode(self):
        """Test decode for a single batch"""
        logging.info("\n=== Testing single batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[128],
                size_per_head=head_dim,
            )

    def test_multi_batch_decode(self):
        """Test decode for multiple batches with varying sequence lengths"""
        logging.info("\n=== Testing multi-batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=4,
                sequence_lengths=[64, 128, 256, 512],
                size_per_head=head_dim,
            )

    def test_different_block_sizes(self):
        """Test with different block sizes"""
        logging.info("\n=== Testing different block sizes ===")
        for head_dim in [128, 256]:
            for block_size in [16, 32, 64, 128]:
                logging.info(
                    f"\n--- Testing head_dim={head_dim}, block_size={block_size} ---"
                )
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    size_per_head=head_dim,
                    seq_size_per_block=block_size,
                )

    def test_different_head_configurations(self):
        """Test with different head configurations (GQA)"""
        logging.info("\n=== Testing different head configurations ===")
        test_cases = [
            (32, 32, "MHA"),  # MHA: head_num == head_num_kv (group_size=1)
            (32, 8, "GQA"),  # GQA: head_num > head_num_kv (group_size=4)
            (32, 4, "GQA-4"),  # GQA with group_size=8
        ]

        for head_dim in [128, 256]:
            for head_num, head_num_kv, name in test_cases:
                logging.info(
                    f"\n--- Testing {name}: head_num={head_num}, head_num_kv={head_num_kv}, head_dim={head_dim} ---"
                )
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                )

    def test_edge_case_sequence_lengths(self):
        """Test edge cases with sequence lengths"""
        logging.info("\n=== Testing edge case sequence lengths ===")

        for head_dim in [128, 256]:
            # Sequence length exactly equal to block size
            logging.info(
                f"\n--- Testing seq_len == block_size, head_dim={head_dim} ---"
            )
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[64],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Sequence length slightly more than block size
            logging.info(f"\n--- Testing seq_len > block_size, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[65],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Very short sequences
            logging.info(f"\n--- Testing short sequences, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=2,
                sequence_lengths=[10, 20],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

    def test_spec_decode(self):
        """Test target verify decode with q_len_per_req > 1"""
        logging.info("\n=== Testing spec decode ===")
        self._test_spec_decode_correctness(
            batch_size=2,
            prefix_lengths=[124, 252],
            q_len_per_req=4,
            size_per_head=128,
            seq_size_per_block=64,
        )


if __name__ == "__main__":
    unittest.main()
