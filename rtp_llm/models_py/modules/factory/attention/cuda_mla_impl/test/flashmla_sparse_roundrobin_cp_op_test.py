"""
Unit tests for RoundRobinSparseMlaFp8CPOp (Context Parallel prefill for Sparse MLA FP8).

These tests simulate multi-rank round-robin CP with mocked all_gather outputs.

Usage:
    python -m pytest rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_sparse_roundrobin_cp_op_test.py -v
    python -m unittest rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_sparse_roundrobin_cp_op_test
"""

import math
from unittest import SkipTest, TestCase, main, skipIf
from unittest.mock import patch

import torch

from rtp_llm.ops import KvCacheDataType, compute_ops
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
    """Test RoundRobinSparseMlaFp8CPOp with mocked multi-rank all_gather."""

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
        tp_size: int = 2,
        tp_rank: int = 0,
    ):
        """Build common attn_inputs, mla_params, parallelism_config, and tensors.

        Parameters match the zigzag CP test (flashmla_sparse_cp_op_test.py):
        num_heads=64, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=512,
        page_size=64, top_k=128, fp8_bytes_per_token=656.
        """
        device = self.device
        assert tp_size > 1
        assert 0 <= tp_rank < tp_size
        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        actual_input_lengths = list(chunk_lengths)
        batch_size = len(actual_input_lengths)
        local_chunk_lengths = [
            (x + tp_size - 1) // tp_size for x in actual_input_lengths
        ]
        local_tokens = sum(local_chunk_lengths)
        total_kv_len = prefix_len * batch_size + sum(actual_input_lengths)

        local_offsets = []
        offset = 0
        for local_len in local_chunk_lengths:
            local_offsets.append(offset)
            offset += local_len

        restore = []
        padding_mask = []
        padding_lengths = []
        for req_idx, actual_len in enumerate(actual_input_lengths):
            local_len = local_chunk_lengths[req_idx]
            padded_len = local_len * tp_size
            padding_lengths.append(padded_len - actual_len)
            for pos in range(padded_len):
                rank = pos % tp_size
                local_idx = pos // tp_size
                restore.append(rank * local_tokens + local_offsets[req_idx] + local_idx)
                padding_mask.append(1 if pos < actual_len else 0)

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            local_chunk_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_cp_padding_lengths = torch.zeros(
            len(actual_input_lengths), dtype=torch.int32, device=device
        )
        cp_params.prefill_cp_padding_lengths[:] = torch.tensor(
            padding_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.tensor(
            restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.tensor(
            padding_mask, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            actual_input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            actual_input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        seq_lengths = [prefix_len + cl for cl in actual_input_lengths]
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
        parallelism_config.tp_rank = tp_rank
        parallelism_config.tp_size = tp_size
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0
        parallelism_config.prefill_cp_config.kv_cache_sharded = True

        q = (
            torch.randn(
                local_tokens,
                num_heads,
                qk_head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )
        compressed_kv = (
            torch.randn(local_tokens, kv_lora_rank, dtype=torch.bfloat16, device=device)
            * 0.1
        )
        k_pe = (
            torch.randn(
                local_tokens, qk_rope_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        topk_indices = torch.randint(
            0,
            max(total_kv_len, 1),
            (local_tokens, 1, top_k),
            dtype=torch.int32,
            device=device,
        )
        batch_indice_parts = []
        for i, cl in enumerate(local_chunk_lengths):
            batch_indice_parts.append(
                torch.full((cl,), i, dtype=torch.int32, device=device)
            )

        batch_indice_d = torch.cat(batch_indice_parts)

        total_blocks = batch_size * block_table_host.shape[1]
        kv_cache_base = (
            (
                torch.randn(
                    total_blocks,
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
            local_tokens=local_tokens,
            local_chunk_lengths=local_chunk_lengths,
            actual_input_lengths=actual_input_lengths,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def _make_all_gather_mock(self, handlers):
        call_idx = {"value": 0}

        def _mock_all_gather(tensor, group=None):
            idx = call_idx["value"]
            call_idx["value"] += 1
            handler = handlers[idx]
            return handler(tensor) if callable(handler) else handler

        return _mock_all_gather

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

    def _build_invalid_topk_for_local_rows(self, op, kv_lens, top_k):
        rows = []
        for req in op._local_req_ids.cpu().tolist():
            rows.append(torch.full((top_k,), kv_lens[req] + 1, dtype=torch.int32))
        return torch.stack(rows).to(self.device)

    # ----------------------------------------------------------------
    # Forward shape / correctness tests
    # ----------------------------------------------------------------

    def test_roundrobin_forward_output_shape(self):
        """RoundRobin CP op forward returns correct local-rank output shape."""
        _set_seed(123)
        params = self._build_common_params(8, [8], tp_size=2, tp_rank=0)
        peer = self._build_common_params(8, [8], tp_size=2, tp_rank=1)
        op = self._make_roundrobin_op(params)

        gather_mock = self._make_all_gather_mock(
            [
                torch.cat([params["compressed_kv"], peer["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer["k_pe"]], dim=0),
            ]
        )

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=gather_mock,
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
            (params["local_tokens"], params["num_heads"], params["kv_lora_rank"]),
        )

    def test_roundrobin_forward_depends_on_peer_rank_kv(self):
        """No-prefix output should depend on the gathered KV from peer ranks."""
        _set_seed(42)
        params = self._build_common_params(8, [8], tp_size=2, tp_rank=0)
        peer_a = self._build_common_params(8, [8], tp_size=2, tp_rank=1)
        peer_b = self._build_common_params(8, [8], tp_size=2, tp_rank=1)
        peer_b["compressed_kv"] = peer_b["compressed_kv"] * 3.0
        peer_b["k_pe"] = peer_b["k_pe"] * -2.0
        op = self._make_roundrobin_op(params)

        gather_a = self._make_all_gather_mock(
            [
                torch.cat([params["compressed_kv"], peer_a["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer_a["k_pe"]], dim=0),
            ]
        )
        gather_b = self._make_all_gather_mock(
            [
                torch.cat([params["compressed_kv"], peer_b["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer_b["k_pe"]], dim=0),
            ]
        )

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=gather_a,
        ):
            out_a = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=gather_b,
        ):
            out_b = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(out_a.shape, out_b.shape)
        self.assertFalse(
            torch.allclose(out_a, out_b, atol=1e-3, rtol=1e-3),
            "Changing peer-rank KV should change the gathered-workspace output",
        )

    def test_roundrobin_forward_multi_chunk(self):
        """RoundRobin CP op works correctly with multiple batch requests [4,4]."""
        _set_seed(77)
        params = self._build_common_params(4, [4, 4], tp_size=2, tp_rank=0)
        peer = self._build_common_params(4, [4, 4], tp_size=2, tp_rank=1)
        op = self._make_roundrobin_op(params)

        gather_mock = self._make_all_gather_mock(
            [
                torch.cat([params["compressed_kv"], peer["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer["k_pe"]], dim=0),
            ]
        )

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=gather_mock,
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
            (params["local_tokens"], params["num_heads"], params["kv_lora_rank"]),
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
        With tp_size=2 and prefix_lengths > 0 (reuse cache), verify that:
        1. The CP op plans successfully with full KV length (prefix + new)
        2. Prefix path now reuses workspace metadata for AG-KV attention
        3. Write-only forward remains valid in the prefix-cache path

        This exercises the path where cu_kv_seqlens_global[-1] (prefix + new)
        is used for buffer sizing, and _cu_local_kv_seqlens / _kv_allgather_restore_indices
        correctly account for prefix tokens.
        """
        _set_seed(42)
        prefix_len = 64
        params = self._build_common_params(
            4, [8], prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        cp_op = self._make_roundrobin_op(params)

        # Verify cu_kv_seqlens_global includes prefix
        self.assertEqual(
            int(cp_op.cu_kv_seqlens_global[-1].item()),
            prefix_len + sum(params["actual_input_lengths"]),
            "cu_kv_seqlens_global should cover prefix + new tokens",
        )
        self.assertTrue(cp_op.has_prefix_cache)
        self.assertIsNotNone(cp_op._ws_block_table)
        self.assertIsNotNone(cp_op._ws_slot_mapping)
        self.assertIsNotNone(cp_op._local_kv_pack_dst_rows)
        self.assertIsNotNone(cp_op._local_kv_pack_src_slots)

        def _repeat_gather(tensor, group=None):
            return tensor.repeat(2, *([1] * (tensor.dim() - 1)))

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_repeat_gather,
        ):
            out_cp = cp_op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                None,
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertIsNone(out_cp)

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

        After the write-only forward pass, slots corresponding to this rank's owned
        local tokens should contain non-zero data.
        """
        _set_seed(200)
        params = self._build_common_params(4, [8], tp_size=2, tp_rank=0)
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

        slot_mapping = op._local_mla_slot_mapping
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

        self.assertGreater(owned_count, 0, "Should have at least one owned slot")

    def test_roundrobin_kv_cache_write_skips_negative_slots(self):
        """Verify that slots with slot_mapping == -1 are NOT written to.

        Manually set some slot_mapping entries to -1 to simulate non-owned tokens
        in a multi-rank sharded scenario, then verify the number of written cache
        rows matches exactly the number of non-(-1) slot_mapping entries.
        """
        _set_seed(201)
        params = self._build_common_params(4, [8], tp_size=2, tp_rank=0)
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        # Mark even-indexed tokens as non-owned to simulate sharding.
        slot_mapping = op._local_mla_slot_mapping.clone()
        for i in range(0, slot_mapping.shape[0], 2):
            slot_mapping[i] = -1
        original_slot_mapping = slot_mapping.clone()
        op._local_mla_slot_mapping = slot_mapping

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
        prefix_len = 64
        params = self._build_common_params(
            4, [8], prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        def _repeat_gather(tensor, group=None):
            return tensor.repeat(2, *([1] * (tensor.dim() - 1)))

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_repeat_gather,
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
        slot_mapping = op._local_mla_slot_mapping
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

        With cp_size=2, virtual_block_size = page_size * 2.
        local capacity = ceil(kv_len / vbs) * page_size.
        Sharded metadata is always computed (used by indexer topk).
        """
        _set_seed(300)
        chunk_lengths = [8]
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=0, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        page_size = params["page_size"]
        kv_len = sum(chunk_lengths)
        vbs = page_size * params["tp_size"]
        n_vblocks = (kv_len + vbs - 1) // vbs
        expected_local_capacity = n_vblocks * page_size

        self.assertEqual(op._total_local_kv, expected_local_capacity)
        cu = op._cu_local_kv_seqlens.cpu().tolist()
        self.assertEqual(cu, [0, expected_local_capacity])

    def test_roundrobin_plan_sharded_kv_seqlens_with_prefix(self):
        """Verify _cu_local_kv_seqlens includes prefix tokens.

        With prefix_len=64, kv_len = 64 + 8 = 72.
        cp_size=2, vbs=128, n_vblocks = ceil(72/128) = 1, local_capacity = 64.
        """
        _set_seed(301)
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        page_size = params["page_size"]
        kv_len = sum(chunk_lengths) + prefix_len
        vbs = page_size * params["tp_size"]
        n_vblocks = (kv_len + vbs - 1) // vbs
        expected_local_capacity = n_vblocks * page_size

        self.assertEqual(op._total_local_kv, expected_local_capacity)
        cu = op._cu_local_kv_seqlens.cpu().tolist()
        self.assertEqual(cu, [0, expected_local_capacity])

    def test_roundrobin_plan_restore_indices_rr_cp2(self):
        """With cp_size=2, restore indices should map rank-major gathered KV back to global order."""
        _set_seed(302)
        chunk_lengths = [8]
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=0, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        restore = op._kv_allgather_restore_indices.cpu().tolist()
        expected = [0, 64, 1, 65, 2, 66, 3, 67]
        self.assertEqual(
            restore,
            expected,
            f"Unexpected cp_size=2 restore indices: {restore}",
        )

    def test_roundrobin_plan_restore_indices_with_prefix(self):
        """Restore indices with prefix should cover prefix + new tokens.

        With prefix_len=64 and 8 new tokens, kv_len=72.
        restore_indices should have 72 entries.
        """
        _set_seed(303)
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        kv_len = sum(chunk_lengths) + prefix_len
        restore = op._kv_allgather_restore_indices.cpu()
        self.assertEqual(
            restore.shape[0],
            kv_len,
            f"restore_indices should have {kv_len} entries (prefix={prefix_len} + new={sum(chunk_lengths)})",
        )
        self.assertEqual(restore[0].item(), 0)
        self.assertEqual(restore[1].item(), op._total_local_kv)

    def test_roundrobin_plan_prefix_workspace_metadata(self):
        """Verify prefix path still prepares workspace metadata for AG-KV attention."""
        _set_seed(304)
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertIsNotNone(op._ws_slot_mapping)
        self.assertIsNotNone(op._ws_block_table)
        self.assertIsNotNone(op._local_kv_pack_dst_rows)
        self.assertIsNotNone(op._local_kv_pack_src_slots)

    def test_roundrobin_plan_workspace_metadata_no_prefix(self):
        """Verify no-prefix path prepares workspace metadata with the expected sizes."""
        _set_seed(305)
        chunk_lengths = [8]
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=0, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertIsNotNone(op._ws_slot_mapping)
        self.assertIsNotNone(op._ws_block_table)
        self.assertIsNotNone(op._ws_total_pages)

    def test_roundrobin_plan_workspace_present_with_prefix(self):
        """With prefix cache, workspace metadata should also exist for AG-KV reuse."""
        _set_seed(306)
        chunk_lengths = [8]
        prefix_len = 64
        params = self._build_common_params(
            4, chunk_lengths, prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertIsNotNone(op._ws_slot_mapping)
        self.assertIsNotNone(op._ws_block_table)
        self.assertIsNotNone(op._ws_total_pages)

    def test_roundrobin_workspace_buffer_reused_across_forwards(self):
        """No-prefix path should reuse the same FP8 workspace buffer across forwards."""
        _set_seed(3061)
        params = self._build_common_params(4, [8], prefix_len=0, tp_size=2, tp_rank=0)
        peer = self._build_common_params(4, [8], prefix_len=0, tp_size=2, tp_rank=1)
        op = self._make_roundrobin_op(params)

        gather_mock = self._make_all_gather_mock(
            [
                torch.cat([params["compressed_kv"], peer["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer["k_pe"]], dim=0),
                torch.cat([params["compressed_kv"], peer["compressed_kv"]], dim=0),
                torch.cat([params["k_pe"], peer["k_pe"]], dim=0),
            ]
        )

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=gather_mock,
        ):
            out0 = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
            ws_ptr0 = op._ws_fp8.data_ptr()
            out1 = op.forward(
                params["q"],
                params["compressed_kv"],
                params["k_pe"],
                params["topk_indices"],
                params["batch_indice_d"],
                params["kv_cache"],
                layer_id=0,
            )
            ws_ptr1 = op._ws_fp8.data_ptr()

        torch.cuda.synchronize()
        self.assertEqual(ws_ptr0, ws_ptr1, "Workspace buffer should be reused")
        self.assertTrue(
            torch.allclose(out0, out1, atol=1e-2, rtol=1e-2),
            "Buffer reuse should not change workspace-path outputs",
        )

    def test_roundrobin_plan_local_slot_mappings(self):
        """Verify local slot mappings are computed for direct cache write."""
        _set_seed(307)
        prefix_len = 64
        params = self._build_common_params(
            4, [8], prefix_len=prefix_len, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        local_tokens = params["local_tokens"]
        self.assertEqual(op._local_mla_slot_mapping.shape[0], local_tokens)
        self.assertEqual(op._local_indexer_slot_mapping.shape[0], local_tokens)
        # Owned tokens should have non-negative slot values
        owned = op._local_mla_slot_mapping >= 0
        self.assertTrue(
            owned.any(), "At least some tokens should be owned by this rank"
        )

    # ----------------------------------------------------------------
    # Mixed prefix: AG-KV vs AG-Q comparison tests
    # ----------------------------------------------------------------

    def _build_params_mixed_prefix(
        self,
        actual_input_lengths: list,
        prefix_lengths_list: list,
        tp_size: int = 2,
        tp_rank: int = 0,
    ):
        """Build params with per-request prefix lengths and CP-sharded fill_params.

        Unlike _build_common_params which uses a uniform prefix_len for all
        requests and calls fill_params without CP args, this helper:
        1. Supports per-request prefix_lengths
        2. Calls fill_params with cp_rank/cp_size/kv_cache_sharded so that
           slot_mapping uses the sharded round-robin formula
        """
        device = self.device
        assert tp_size > 1
        assert 0 <= tp_rank < tp_size
        batch_size = len(actual_input_lengths)
        assert len(prefix_lengths_list) == batch_size

        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        local_chunk_lengths = [
            (x + tp_size - 1) // tp_size for x in actual_input_lengths
        ]
        local_tokens = sum(local_chunk_lengths)

        local_offsets = []
        offset = 0
        for local_len in local_chunk_lengths:
            local_offsets.append(offset)
            offset += local_len

        restore = []
        padding_mask = []
        padding_lengths = []
        for req_idx, actual_len in enumerate(actual_input_lengths):
            local_len = local_chunk_lengths[req_idx]
            padded_len = local_len * tp_size
            padding_lengths.append(padded_len - actual_len)
            for pos in range(padded_len):
                rank = pos % tp_size
                local_idx = pos // tp_size
                restore.append(rank * local_tokens + local_offsets[req_idx] + local_idx)
                padding_mask.append(1 if pos < actual_len else 0)

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            local_chunk_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_cp_padding_lengths = torch.tensor(
            padding_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.tensor(
            restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.tensor(
            padding_mask, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            actual_input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            actual_input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        seq_lengths = [
            prefix_lengths_list[i] + actual_input_lengths[i] for i in range(batch_size)
        ]
        attn_inputs.sequence_lengths = torch.tensor(
            seq_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths_list, dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params

        max_seq_len = max(seq_lengths)
        vbs = page_size * tp_size
        max_vblocks = max(math.ceil(sl / vbs) for sl in seq_lengths)
        block_table_host = torch.zeros(
            batch_size, max_vblocks, dtype=torch.int32, device=torch.device("cpu")
        )
        block_idx = 0
        for i in range(batch_size):
            n_vb = math.ceil(seq_lengths[i] / vbs)
            for j in range(n_vb):
                block_table_host[i, j] = block_idx
                block_idx += 1
        total_blocks = block_idx

        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(
            attn_inputs,
            page_size,
            cp_rank=tp_rank,
            cp_size=tp_size,
            kv_cache_sharded=True,
        )

        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = tp_rank
        parallelism_config.tp_size = tp_size
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0
        parallelism_config.prefill_cp_config.kv_cache_sharded = True

        q = (
            torch.randn(
                local_tokens,
                num_heads,
                qk_head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )
        compressed_kv = (
            torch.randn(
                local_tokens,
                kv_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )
        k_pe = (
            torch.randn(
                local_tokens,
                qk_rope_head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )

        total_kv_len = sum(seq_lengths)
        topk_indices = torch.randint(
            0,
            max(total_kv_len, 1),
            (local_tokens, 1, top_k),
            dtype=torch.int32,
            device=device,
        )
        batch_indice_parts = []
        for i, cl in enumerate(local_chunk_lengths):
            batch_indice_parts.append(
                torch.full((cl,), i, dtype=torch.int32, device=device)
            )
        batch_indice_d = torch.cat(batch_indice_parts)

        kv_cache_base = (
            (
                torch.randn(
                    total_blocks,
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
            local_tokens=local_tokens,
            local_chunk_lengths=local_chunk_lengths,
            actual_input_lengths=actual_input_lengths,
            prefix_lengths_list=prefix_lengths_list,
            tp_size=tp_size,
            tp_rank=tp_rank,
            total_blocks=total_blocks,
        )

    def test_agkv_forward_mixed_prefix_shape(self):
        """AG-KV forward with mixed prefix [128, 0] runs without crash and correct shape."""
        _set_seed(500)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [128, 0]

        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertTrue(op.has_prefix_cache)
        self.assertFalse(op._use_prefix_q_path)

        def _repeat_gather(tensor, group=None):
            return tensor.repeat(2, *([1] * (tensor.dim() - 1)))

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_repeat_gather,
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
            (params["local_tokens"], params["num_heads"], params["kv_lora_rank"]),
        )

    def test_agkv_forward_mixed_prefix_all_zero_prefix(self):
        """AG-KV forward with all prefix_len=0 should also work (pure new-token batch)."""
        _set_seed(501)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [0, 0]

        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertFalse(op.has_prefix_cache)

        def _repeat_gather(tensor, group=None):
            return tensor.repeat(2, *([1] * (tensor.dim() - 1)))

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_repeat_gather,
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
            (params["local_tokens"], params["num_heads"], params["kv_lora_rank"]),
        )

    def test_agkv_workspace_new_token_data_integrity(self):
        """Data written to sharded cache is correctly recovered in AG-KV workspace.

        For each new token owned by this rank:
        1. concat_and_cache_mla writes FP8 data to sharded cache at slot S
        2. _build_workspace_from_sharded_cache reads from slot S into local_rows
        3. After all_gather + restore + materialize, workspace should contain
           the same FP8 bytes at the corresponding workspace slot

        We verify byte-for-byte match between cache[src_slot] and workspace[ws_slot]
        for all new tokens owned by this rank.
        """
        _set_seed(600)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [128, 0]
        tp_size = 2
        tp_rank = 0

        params = self._build_params_mixed_prefix(
            actual_input_lengths,
            prefix_lengths_list,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        op = self._make_roundrobin_op(params)

        kv_cache = params["kv_cache"]
        kv_cache.kv_cache_base.zero_()

        from rtp_llm.ops import compute_ops

        compute_ops.concat_and_cache_mla(
            params["compressed_kv"],
            params["k_pe"],
            kv_cache.kv_cache_base,
            op._local_mla_slot_mapping,
            op.kv_cache_write_op.kv_cache_type,
            torch.ones(1, dtype=torch.float32, device=self.device),
        )
        torch.cuda.synchronize()

        def _repeat_gather(tensor, group=None):
            return tensor.repeat(tp_size, *([1] * (tensor.dim() - 1)))

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_repeat_gather,
        ):
            workspace_cache = op._build_workspace_from_sharded_cache(kv_cache)
        torch.cuda.synchronize()

        page_size = params["page_size"]
        cache_rows = kv_cache.kv_cache_base.view(-1, kv_cache.kv_cache_base.size(-1))
        ws_rows = workspace_cache.view(-1, workspace_cache.size(-1))

        src_slots = op._local_kv_pack_src_slots.cpu()
        dst_rows = op._local_kv_pack_dst_rows.cpu()
        ws_slot_mapping = op._ws_slot_mapping.cpu()
        restore_indices = op._kv_allgather_restore_indices.cpu()

        total_kv = ws_slot_mapping.shape[0]
        mismatches = 0
        checked = 0
        for global_kv_idx in range(total_kv):
            restore_idx = restore_indices[global_kv_idx].item()
            if restore_idx >= op._total_local_kv:
                continue

            found = (dst_rows == restore_idx).nonzero(as_tuple=True)[0]
            if found.numel() == 0:
                continue

            pack_idx = found[0].item()
            src_slot = src_slots[pack_idx].item()
            ws_slot = ws_slot_mapping[global_kv_idx].item()

            cache_row = cache_rows[src_slot]
            ws_row = ws_rows[ws_slot]

            if not torch.equal(cache_row, ws_row):
                mismatches += 1
            checked += 1

        self.assertGreater(checked, 0, "Should have checked at least one owned token")
        self.assertEqual(
            mismatches,
            0,
            f"{mismatches}/{checked} workspace rows do not match cache data "
            f"(new tokens owned by rank {tp_rank})",
        )

    def test_agkv_plan_mixed_prefix_attributes(self):
        """Verify plan() attributes for mixed prefix batch [128, 0]."""
        _set_seed(700)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [128, 0]

        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        self.assertTrue(op.has_prefix_cache, "Batch has at least one prefix > 0")

        kv_lens = [
            prefix_lengths_list[i] + actual_input_lengths[i]
            for i in range(len(actual_input_lengths))
        ]
        expected_total_kv = sum(kv_lens)
        self.assertEqual(
            int(op.cu_kv_seqlens_global[-1].item()),
            expected_total_kv,
        )

        ws_slot_mapping = op._ws_slot_mapping
        self.assertEqual(ws_slot_mapping.shape[0], expected_total_kv)

        self.assertEqual(
            op._kv_allgather_restore_indices.shape[0],
            expected_total_kv,
        )

        page_size = params["page_size"]
        vbs = page_size * params["tp_size"]
        expected_local_kv = sum(math.ceil(kl / vbs) * page_size for kl in kv_lens)
        self.assertEqual(op._total_local_kv, expected_local_kv)

        self.assertIsNotNone(op._local_kv_pack_src_slots)
        self.assertIsNotNone(op._local_kv_pack_dst_rows)
        self.assertGreater(op._local_kv_pack_src_slots.numel(), 0)

    def test_agkv_topk_conversion_mixed_prefix(self):
        """_convert_topk_indices_to_workspace produces valid workspace indices
        for a mixed-prefix batch."""
        _set_seed(800)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [128, 0]

        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)

        kv_lens = [
            prefix_lengths_list[i] + actual_input_lengths[i]
            for i in range(len(actual_input_lengths))
        ]

        valid_tokens = op.total_local_ids.numel()
        local_req_ids = op._local_req_ids.cpu()

        topk_list = []
        for i in range(valid_tokens):
            req = local_req_ids[i].item()
            max_pos = kv_lens[req]
            row = torch.randint(0, max_pos, (params["top_k"],), dtype=torch.int32)
            topk_list.append(row)
        topk = torch.stack(topk_list).to(self.device)

        ws_topk = op._convert_topk_indices_to_workspace(topk, op._ws_block_table)
        torch.cuda.synchronize()

        self.assertEqual(ws_topk.shape[0], valid_tokens)
        self.assertEqual(ws_topk.shape[-1], params["top_k"])

        ws_topk_cpu = ws_topk.cpu()
        page_size = params["page_size"]
        ws_total_slots = op._ws_total_pages * page_size

        valid_mask = topk.cpu() >= 0
        ws_vals = ws_topk_cpu.view(valid_tokens, -1)
        for i in range(valid_tokens):
            for k in range(params["top_k"]):
                if valid_mask[i, k]:
                    slot = ws_vals[i, k].item()
                    self.assertGreaterEqual(
                        slot, 0, f"Token {i} topk {k}: negative slot"
                    )
                    self.assertLess(
                        slot,
                        ws_total_slots,
                        f"Token {i} topk {k}: slot {slot} >= total {ws_total_slots}",
                    )

    def test_roundrobin_workspace_masks_negative_topk(self):
        """Workspace topk conversion should preserve -1 (inactive) indices."""
        _set_seed(810)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [0, 0]
        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)
        valid_tokens = op.total_local_ids.numel()
        topk = torch.full(
            (valid_tokens, params["top_k"]), -1, dtype=torch.int32, device=self.device
        )

        ws_topk = op._convert_topk_indices_to_workspace(topk, op._ws_block_table)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(ws_topk, torch.full_like(ws_topk, -1)),
            "Negative (inactive) topk should remain -1 in workspace path",
        )

    def test_roundrobin_sharded_cache_masks_oob_topk(self):
        """Sharded-cache topk filtering should also mask request-local OOB positions."""
        _set_seed(811)
        actual_input_lengths = [8, 16]
        prefix_lengths_list = [128, 0]
        params = self._build_params_mixed_prefix(
            actual_input_lengths, prefix_lengths_list, tp_size=2, tp_rank=0
        )
        op = self._make_roundrobin_op(params)
        topk = self._build_invalid_topk_for_local_rows(
            op,
            [p + a for p, a in zip(prefix_lengths_list, actual_input_lengths)],
            params["top_k"],
        )

        sharded_topk = op._filter_topk_to_sharded_cache(topk)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(sharded_topk, torch.full_like(sharded_topk, -1)),
            "Out-of-range request-local topk should be masked to -1 in sharded path",
        )


if __name__ == "__main__":
    main()
