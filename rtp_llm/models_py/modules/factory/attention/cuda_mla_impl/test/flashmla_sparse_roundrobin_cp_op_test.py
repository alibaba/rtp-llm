"""
Unit tests for RoundRobinSparseMlaFp8CPOp (Context Parallel prefill for Sparse MLA FP8).

Tests the CP path with tp_size=1 (single rank): all_gather is identity,
so output should match non-CP SparseMlaFp8Op on the same inputs.

Usage:
    python -m pytest rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_sparse_roundrobin_cp_op_test.py -v
    python -m unittest rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_sparse_roundrobin_cp_op_test
"""

import math
from unittest import SkipTest, TestCase, main, skipIf
from unittest.mock import patch

import torch

from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)


def _check_cuda_flashmla():
    """Require CUDA >= 12.9 and flash_mla import."""
    try:
        if not torch.version.cuda:
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) < (12, 9):
            return False
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: F401

        return True
    except (ImportError, AttributeError, ValueError):
        return False


CUDA_FLASHMLA_OK = _check_cuda_flashmla()
SKIP_REASON = "Requires CUDA >= 12.9 and flash_mla"


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_block_table(
    batch_size: int, seq_len: int, page_size: int, device: torch.device
) -> torch.Tensor:
    num_blocks_per_seq = math.ceil(seq_len / page_size)
    block_table = torch.zeros(
        [batch_size, num_blocks_per_seq], dtype=torch.int32, device=torch.device("cpu")
    )
    bias = 0
    for i in range(batch_size):
        block_table[i, :] = torch.arange(
            bias,
            bias + num_blocks_per_seq,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias += num_blocks_per_seq
    return block_table.to(device)


@skipIf(not CUDA_FLASHMLA_OK, SKIP_REASON)
class RoundRobinSparseMlaFp8CPOpTest(TestCase):
    """Test RoundRobinSparseMlaFp8CPOp with single rank (tp_size=1)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)

    def setUp(self):
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.empty_cache()

    def tearDown(self):
        torch.cuda.empty_cache()

    def _build_common_params(
        self,
        total_q_len: int,
        chunk_lengths: list,
        prefix_len: int = 0,
    ):
        """Build common attn_inputs, mla_params, parallelism_config, and tensors.

        Parameters match the zigzag CP test (flashmla_sparse_cp_op_test.py):
        num_heads=64, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=512,
        page_size=64, top_k=128, fp8_bytes_per_token=656.
        """
        device = self.device
        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        batch_size = len(chunk_lengths)
        n_restore = sum(chunk_lengths)
        total_kv_len = prefix_len * batch_size + n_restore

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_cp_padding_lengths = torch.zeros(
            len(chunk_lengths), dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.arange(
            n_restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.ones(
            n_restore, dtype=torch.int32, device=device
        )
        # actual_input_lengths_cpu has one entry per batch request, matching chunk_lengths
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        seq_lengths = [prefix_len + cl for cl in chunk_lengths]
        attn_inputs.sequence_lengths = torch.tensor(
            seq_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [prefix_len] * batch_size, dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params

        max_seq_len = max(seq_lengths)
        block_table_host = _make_block_table(
            batch_size, max_seq_len, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0
        parallelism_config.prefill_cp_config.kv_cache_sharded = True

        q = (
            torch.randn(
                total_q_len, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        compressed_kv = (
            torch.randn(total_q_len, kv_lora_rank, dtype=torch.bfloat16, device=device)
            * 0.1
        )
        k_pe = (
            torch.randn(
                total_q_len, qk_rope_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        topk_indices = torch.randint(
            0,
            max(total_kv_len, 1),
            (total_q_len, 1, top_k),
            dtype=torch.int32,
            device=device,
        )
        batch_indice_parts = []
        for i, cl in enumerate(chunk_lengths):
            batch_indice_parts.append(
                torch.full((cl,), i, dtype=torch.int32, device=device)
            )

        batch_indice_d = batch_indice_d = torch.cat(batch_indice_parts)

        total_blocks = batch_size * block_table_host.shape[1]
        kv_cache_base = torch.empty(
            total_blocks,
            page_size,
            fp8_bytes_per_token,
            dtype=torch.uint8,
            device=device,
        )
        kv_cache = KVCache()
        kv_cache.kv_cache_base = kv_cache_base

        return dict(
            attn_inputs=attn_inputs,
            mla_params=mla_params,
            parallelism_config=parallelism_config,
            block_table_device=block_table_device,
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            topk_indices=topk_indices,
            batch_indice_d=batch_indice_d,
            kv_cache=kv_cache,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
            fp8_bytes_per_token=fp8_bytes_per_token,
            total_kv_len=total_kv_len,
        )

    def _make_roundrobin_op(self, params):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            RoundRobinSparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )

        op = RoundRobinSparseMlaFp8CPOp(
            num_heads=params["num_heads"],
            kv_lora_rank=params["kv_lora_rank"],
            qk_rope_head_dim=params["qk_rope_head_dim"],
            qk_nope_head_dim=params["qk_nope_head_dim"],
            page_size=params["page_size"],
            softmax_extra_scale=params["softmax_extra_scale"],
            top_k=params["top_k"],
            attn_inputs=params["attn_inputs"],
            parallelism_config=params["parallelism_config"],
        )
        op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        op.write_cache_store_impl = None
        op.attn_inputs = params["attn_inputs"]
        op.plan(params["mla_params"], params["block_table_device"])
        return op

    # ----------------------------------------------------------------
    # Forward shape / correctness tests
    # ----------------------------------------------------------------

    def test_roundrobin_forward_output_shape(self):
        """RoundRobin CP op forward returns correct shape [total_q_len, num_heads, kv_lora_rank]."""
        _set_seed(123)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertEqual(
            out.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

    def test_roundrobin_forward_match_non_cp(self):
        """
        With tp_size=1, RoundRobin CP all_gather is identity.
        Output should match non-CP SparseMlaFp8Op on the same inputs.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )

        _set_seed(42)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out_cp = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(
            out_cp.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

        # Non-CP reference
        kv_cache_base = params["kv_cache"].kv_cache_base
        kv_cache_flat = kv_cache_base.view(-1, 1, kv_cache_base.size(-1))
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        non_cp_op = SparseMlaFp8Op(
            num_heads=params["num_heads"],
            kv_lora_rank=params["kv_lora_rank"],
            qk_rope_head_dim=params["qk_rope_head_dim"],
            qk_nope_head_dim=params["qk_nope_head_dim"],
            page_size=params["page_size"],
            softmax_extra_scale=params["softmax_extra_scale"],
            top_k=params["top_k"],
        )
        non_cp_op.plan(params["mla_params"], params["block_table_device"])
        out_non_cp = non_cp_op.forward(
            params["q"], kv_cache_flat, params["topk_indices"], layer_id=0
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(out_cp, out_non_cp, atol=1e-2, rtol=1e-2),
            "RoundRobin CP output should match non-CP when tp_size=1 (all_gather identity)",
        )

    def test_roundrobin_forward_multi_chunk(self):
        """RoundRobin CP op works correctly with multiple batch requests [4,4]."""
        _set_seed(77)
        total_q_len = 8
        # For round-robin, chunk_lengths = [4, 4] means 2 batch requests, each with 4 tokens
        chunk_lengths = [4, 4]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertEqual(
            out.shape,
            (total_q_len, params["num_heads"], params["kv_lora_rank"]),
        )

    def test_roundrobin_no_topk_returns_none(self):
        """When topk is None, forward should return None (write-only path)."""
        _set_seed(99)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,  # topk=None
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        self.assertIsNone(out)

    # ----------------------------------------------------------------
    # Prefix cache forward test (mirrors zigzag test_cp_op_forward_with_prefix_cache)
    # ----------------------------------------------------------------

    def test_roundrobin_forward_with_prefix_cache(self):
        """
        With tp_size=1 and prefix_lengths > 0 (reuse cache), verify that:
        1. The CP op does not crash (buffer allocation uses full KV length)
        2. Output shape is correct
        3. Output matches non-CP reference on the same cache state

        This exercises the path where cu_kv_seqlens_global[-1] (prefix + new)
        is used for buffer sizing, and _cu_local_kv_seqlens / _kv_allgather_restore_indices
        correctly account for prefix tokens.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )

        _set_seed(42)
        device = self.device

        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        prefix_len = 64
        new_tokens = 8
        total_kv_len = prefix_len + new_tokens
        num_blocks_needed = math.ceil(total_kv_len / page_size)

        chunk_lengths = [new_tokens]
        n_restore = sum(chunk_lengths)

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_cp_padding_lengths = torch.zeros(
            len(chunk_lengths), dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.arange(
            n_restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.ones(
            n_restore, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            [new_tokens], dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [new_tokens], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [prefix_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params

        block_table_host = _make_block_table(
            1, total_kv_len, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0
        parallelism_config.prefill_cp_config.kv_cache_sharded = True

        q = (
            torch.randn(
                new_tokens, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        compressed_kv = (
            torch.randn(new_tokens, kv_lora_rank, dtype=torch.bfloat16, device=device)
            * 0.1
        )
        k_pe = (
            torch.randn(
                new_tokens, qk_rope_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        # topk indices span the full KV range [0, prefix_len + new_tokens)
        topk_indices = torch.randint(
            0, total_kv_len, (new_tokens, 1, top_k), dtype=torch.int32, device=device
        )
        batch_indice_d = torch.zeros(new_tokens, dtype=torch.int32, device=device)

        num_blocks = block_table_host.shape[1]
        # Pre-fill cache with random data to simulate prefix blocks already present
        kv_cache_base = (
            (
                torch.randn(
                    num_blocks,
                    page_size,
                    fp8_bytes_per_token,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.1
            )
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        kv_cache = KVCache()
        kv_cache.kv_cache_base = kv_cache_base

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            RoundRobinSparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )

        cp_op = RoundRobinSparseMlaFp8CPOp(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
            attn_inputs=attn_inputs,
            parallelism_config=parallelism_config,
        )
        cp_op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        cp_op.write_cache_store_impl = None
        cp_op.attn_inputs = attn_inputs

        cp_op.plan(mla_params, block_table_device)

        # Verify cu_kv_seqlens_global includes prefix
        self.assertEqual(
            int(cp_op.cu_kv_seqlens_global[-1].item()),
            total_kv_len,
            "cu_kv_seqlens_global should cover prefix + new tokens",
        )

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out_cp = cp_op.forward(
                q,
                compressed_kv,
                k_pe,
                topk_indices,
                batch_indice_d,
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(
            out_cp.shape,
            (new_tokens, num_heads, kv_lora_rank),
            "CP output shape should be [new_tokens, num_heads, kv_lora_rank]",
        )

        # Compare against non-CP reference on the same cache
        kv_cache_flat = kv_cache_base.view(-1, 1, kv_cache_base.size(-1))
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        non_cp_op = SparseMlaFp8Op(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )
        non_cp_op.plan(mla_params, block_table_device)
        out_non_cp = non_cp_op.forward(q, kv_cache_flat, topk_indices, layer_id=0)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(out_cp, out_non_cp, atol=1e-2, rtol=1e-2),
            "RoundRobin CP output with prefix cache should match non-CP output",
        )

    # ----------------------------------------------------------------
    # cu_kv_seqlens_global tests
    # ----------------------------------------------------------------

    def test_roundrobin_prefix_cache_plan_succeeds(self):
        """RoundRobin with prefix_lengths > 0 should succeed (no longer raises)."""
        _set_seed(55)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths, prefix_len=64)
        op = self._make_roundrobin_op(params)
        expected_total_kv = sum(chunk_lengths) + 64
        self.assertEqual(
            int(op.cu_kv_seqlens_global[-1].item()),
            expected_total_kv,
            "cu_kv_seqlens_global should include prefix_lengths",
        )

    def test_roundrobin_cu_kv_seqlens_global(self):
        """Verify cu_kv_seqlens_global is correctly computed (no prefix)."""
        _set_seed(33)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        self.assertEqual(
            int(op.cu_kv_seqlens_global[-1].item()),
            sum(chunk_lengths),
            "cu_kv_seqlens_global[-1] should equal total KV length",
        )

    # ----------------------------------------------------------------
    # KV cache write verification tests
    # ----------------------------------------------------------------

    def test_roundrobin_kv_cache_write_owned_slots(self):
        """Verify that forward(topk=None) writes data to owned slots in kv_cache.

        After the write-only forward pass, slots corresponding to owned tokens
        (slot_mapping != -1) should contain non-zero data.
        """
        _set_seed(200)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,  # topk=None → write-only
                params["batch_indice_d"],
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertIsNone(out)

        slot_mapping = params["mla_params"].slot_mapping
        page_size = params["page_size"]
        cache = kv_cache.kv_cache_base  # [num_blocks, page_size, fp8_bytes]

        owned_count = 0
        for i in range(slot_mapping.shape[0]):
            slot = int(slot_mapping[i].item())
            if slot == -1:
                continue
            block_idx = slot // page_size
            offset_in_block = slot % page_size
            row = cache[block_idx, offset_in_block]
            self.assertTrue(
                row.any(),
                f"Owned slot {slot} (token {i}) should have non-zero data after cache write",
            )
            owned_count += 1

        # With cp_size=1, all tokens are owned (no -1 in slot_mapping)
        self.assertGreater(owned_count, 0, "Should have at least one owned slot")

    def test_roundrobin_kv_cache_write_skips_negative_slots(self):
        """Verify that slots with slot_mapping == -1 are NOT written to.

        Manually set some slot_mapping entries to -1 to simulate non-owned tokens
        in a multi-rank sharded scenario, then verify the number of written cache
        rows matches exactly the number of non-(-1) slot_mapping entries.
        """
        _set_seed(201)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths)
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        # Mark even-indexed tokens as non-owned to simulate sharding.
        # Must also update the precomputed _local_mla_slot_mapping, which is
        # what _write_local_cache actually uses for cache writes.
        slot_mapping = params["mla_params"].slot_mapping
        for i in range(0, slot_mapping.shape[0], 2):
            slot_mapping[i] = -1
        original_slot_mapping = slot_mapping.clone()
        op._local_mla_slot_mapping = op._build_local_slot_mapping(
            slot_mapping, slot_mapping.shape[0], op.prefill_cp_size
        )

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,
                params["batch_indice_d"],
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()

        cache = kv_cache.kv_cache_base
        # Count non-zero rows in cache
        written_rows = 0
        for b in range(cache.shape[0]):
            for r in range(cache.shape[1]):
                if cache[b, r].any():
                    written_rows += 1

        expected_writes = int((original_slot_mapping != -1).sum().item())
        self.assertEqual(
            written_rows,
            expected_writes,
            f"Number of written cache rows ({written_rows}) should match "
            f"non-(-1) slot_mapping entries ({expected_writes})",
        )

    def test_roundrobin_kv_cache_write_with_prefix(self):
        """Verify cache write works correctly when prefix_len > 0.

        With prefix cache, only new tokens (not prefix) are written via forward.
        The slot_mapping from fill_params covers new tokens starting at prefix_length
        positions. Verify that the correct number of slots are written.
        """
        _set_seed(202)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        def _identity_all_gather(tensor, group=None):
            return tensor

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,
                params["batch_indice_d"],
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertIsNone(out)

        # Count written rows — should be exactly the number of new tokens
        # that have valid (non -1) slot_mapping entries
        slot_mapping = params["mla_params"].slot_mapping
        expected_writes = int((slot_mapping != -1).sum().item())

        cache = kv_cache.kv_cache_base
        written_rows = 0
        for b in range(cache.shape[0]):
            for r in range(cache.shape[1]):
                if cache[b, r].any():
                    written_rows += 1

        self.assertEqual(
            written_rows,
            expected_writes,
            f"With prefix_len={prefix_len}, written rows ({written_rows}) should match "
            f"valid slot_mapping entries ({expected_writes})",
        )
        self.assertGreater(written_rows, 0, "Should write at least some new tokens")

    # ----------------------------------------------------------------
    # plan() attribute verification tests
    # ----------------------------------------------------------------

    def test_roundrobin_plan_sharded_kv_seqlens_no_prefix(self):
        """Verify _cu_local_kv_seqlens and _total_local_kv with prefix_len=0.

        With cp_size=1, virtual_block_size = page_size * 1 = page_size.
        local capacity = ceil(kv_len / vbs) * page_size.
        Sharded metadata is always computed (used by indexer topk).
        """
        _set_seed(300)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(total_q_len, chunk_lengths, prefix_len=0)
        op = self._make_roundrobin_op(params)

        page_size = params["page_size"]
        kv_len = sum(chunk_lengths)
        vbs = page_size * 1
        n_vblocks = (kv_len + vbs - 1) // vbs
        expected_local_capacity = n_vblocks * page_size

        self.assertEqual(op._total_local_kv, expected_local_capacity)
        cu = op._cu_local_kv_seqlens.cpu().tolist()
        self.assertEqual(cu, [0, expected_local_capacity])

    def test_roundrobin_plan_sharded_kv_seqlens_with_prefix(self):
        """Verify _cu_local_kv_seqlens includes prefix tokens.

        With prefix_len=64, kv_len = 64 + 8 = 72.
        cp_size=1, vbs=64, n_vblocks = ceil(72/64) = 2, local_capacity = 128.
        """
        _set_seed(301)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        page_size = params["page_size"]
        kv_len = sum(chunk_lengths) + prefix_len
        vbs = page_size * 1
        n_vblocks = (kv_len + vbs - 1) // vbs
        expected_local_capacity = n_vblocks * page_size

        self.assertEqual(op._total_local_kv, expected_local_capacity)
        cu = op._cu_local_kv_seqlens.cpu().tolist()
        self.assertEqual(cu, [0, expected_local_capacity])

    def test_roundrobin_plan_restore_indices_identity_cp1(self):
        """With cp_size=1, restore indices should be identity mapping.

        For cp_size=1, every token is owned by rank 0. The all-gather is identity,
        so restore_indices[p] should equal p for all valid positions.
        Restore indices are always computed (used by indexer topk).
        """
        _set_seed(302)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        kv_len = sum(chunk_lengths) + prefix_len
        restore = op._kv_allgather_restore_indices.cpu()
        self.assertEqual(restore.shape[0], kv_len)
        expected = torch.arange(kv_len, dtype=torch.long, device=torch.device("cpu"))
        self.assertTrue(
            torch.equal(restore, expected),
            f"With cp_size=1, restore indices should be identity. Got {restore.tolist()}",
        )

    def test_roundrobin_plan_restore_indices_with_prefix(self):
        """Restore indices with prefix should cover prefix + new tokens.

        With prefix_len=64 and 8 new tokens, kv_len=72.
        restore_indices should have 72 entries.
        """
        _set_seed(303)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        kv_len = sum(chunk_lengths) + prefix_len
        restore = op._kv_allgather_restore_indices.cpu()
        self.assertEqual(
            restore.shape[0],
            kv_len,
            f"restore_indices should have {kv_len} entries (prefix={prefix_len} + new={sum(chunk_lengths)})",
        )

    def test_roundrobin_plan_global_fp8_metadata(self):
        """Verify _global_fp8_metadata is computed for the global total_q."""
        _set_seed(304)
        total_q_len = 8
        chunk_lengths = [8]
        # _global_fp8_metadata is only computed in the prefix-cache path
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        self.assertIsNotNone(op._global_fp8_metadata)
        self.assertIsNotNone(op._global_fp8_metadata.tile_scheduler_metadata)
        # num_splits may be None depending on get_mla_metadata scheduling decisions;
        # just verify the attribute exists.
        self.assertTrue(hasattr(op._global_fp8_metadata, "num_splits"))

    def test_roundrobin_plan_workspace_metadata_no_prefix(self):
        """Verify workspace metadata is computed only for non-prefix path."""
        _set_seed(305)
        total_q_len = 8
        chunk_lengths = [8]
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=0
        )
        op = self._make_roundrobin_op(params)

        n_restore = sum(chunk_lengths)
        self.assertEqual(op._ws_total_kv, n_restore)
        self.assertIsNotNone(op._ws_slot_mapping)
        self.assertEqual(op._ws_slot_mapping.shape[0], n_restore)
        self.assertIsNotNone(op._ws_block_table)

    def test_roundrobin_plan_workspace_none_with_prefix(self):
        """With prefix cache, workspace metadata should be None (not needed)."""
        _set_seed(306)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        self.assertIsNone(op._ws_slot_mapping)
        self.assertIsNone(op._ws_block_table)
        self.assertIsNone(op._ws_total_pages)

    def test_roundrobin_plan_local_slot_mappings(self):
        """Verify local slot mappings are computed for direct cache write."""
        _set_seed(307)
        total_q_len = 8
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            total_q_len, chunk_lengths, prefix_len=prefix_len
        )
        op = self._make_roundrobin_op(params)

        local_tokens = sum(chunk_lengths)
        self.assertEqual(op._local_mla_slot_mapping.shape[0], local_tokens)
        self.assertEqual(op._local_indexer_slot_mapping.shape[0], local_tokens)
        # Owned tokens should have non-negative slot values
        owned = op._local_mla_slot_mapping >= 0
        self.assertTrue(owned.any(), "At least some tokens should be owned by this rank")


if __name__ == "__main__":
    main()
