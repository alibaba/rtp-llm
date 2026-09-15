"""
Unit test for IndexerOp._get_topk_ragged_cp.

Tests the CP path for computing TopK indices in prefill with context parallel:
single rank with one chunk so that generate_q_indices yields valid indices
for index_select on local q_fp8.
"""

import functools
import math
import multiprocessing as mp
import os
from unittest import SkipTest, TestCase, main, skipIf
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)
from rtp_llm.test.utils.port_util import PortManager


def _check_cuda_deep_gemm():
    try:
        if not torch.cuda.is_available():
            return False
        import deep_gemm  # noqa: F401

        return True
    except ImportError:
        return False


CUDA_DEEPGEMM_OK = _check_cuda_deep_gemm()
SKIP_REASON = "CUDA and deep_gemm required for IndexerOp._get_topk_ragged_cp"


def _expected_ue8m0_dequant(values: torch.Tensor) -> torch.Tensor:
    values_fp32 = values.float()
    scales = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(
                torch.clamp(values_fp32.abs().amax(dim=-1, keepdim=True), min=1e-4)
                / 448.0
            )
        ),
    )
    return (values_fp32 / scales).to(torch.float8_e4m3fn).float() * scales


def _tp2_indexer_worker(rank: int, nccl_port: int) -> None:
    import deep_gemm
    import torch.distributed as dist

    from rtp_llm.models_py.kernels.cuda.fast_topk import (
        fast_topk_transform_ragged_fused,
    )

    initialized = False
    try:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        parallelism = ParallelismConfig()
        parallelism.world_rank = rank
        parallelism.world_size = 2
        parallelism.local_rank = rank
        parallelism.tp_rank = rank
        parallelism.tp_size = 2
        parallelism.dp_rank = 0
        parallelism.dp_size = 1

        original_init_process_group = dist.init_process_group

        @functools.wraps(original_init_process_group)
        def _init_process_group(*args, **kwargs):
            options = dict(kwargs)
            backend = options.get("backend") or (args[0] if args else None)
            if backend == "nccl" and "device_id" not in options:
                options["device_id"] = device
                try:
                    return original_init_process_group(*args, **options)
                except TypeError:
                    return original_init_process_group(*args, **kwargs)
            return original_init_process_group(*args, **kwargs)

        base_port = nccl_port + 11
        with patch.object(dist, "init_process_group", _init_process_group):
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.init_symm_mem_communicator",
                lambda *_args, **_kwargs: None,
                create=True,
            ):
                init_distributed_environment(
                    parallelism,
                    NcclCommConfig(
                        nccl_ip="127.0.0.1",
                        tp_nccl_port=base_port - 2,
                        dp_tp_nccl_port=base_port - 10,
                        ffn_tp_nccl_port=base_port - 5,
                    ),
                    nccl_init_port=base_port - 11,
                    backend="nccl",
                    timeout=120,
                )
        initialized = True

        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        page_size = 64
        prefix_length = 2049
        input_tokens = 8
        total_kv_tokens = prefix_length + input_tokens

        op = IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=0,
            blocksize=page_size,
            block_size=index_head_dim,
        )

        generator = torch.Generator(device="cpu").manual_seed(20260904)
        prefix_keys = torch.randn(
            prefix_length, index_head_dim, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        input_keys = torch.randn(
            input_tokens, index_head_dim, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        global_queries = torch.randn(
            input_tokens,
            index_n_heads,
            index_head_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        prefix_keys = prefix_keys.to(device)
        input_keys = input_keys.to(device)
        global_queries = global_queries.to(device)

        cache_stride = index_head_dim + 4
        page_count = math.ceil(total_kv_tokens / page_size)
        cache = LayerKVCache(
            torch.empty(
                page_count,
                page_size * cache_stride,
                dtype=torch.uint8,
                device=device,
            ),
            page_size,
        )
        op.quant_k_only(
            prefix_keys,
            cache,
            torch.arange(prefix_length, dtype=torch.int64, device=device),
        )

        shard_ids = torch.arange(rank, input_tokens, 2, device=device)
        local_keys = input_keys[shard_ids].contiguous()
        selected_global_ids = shard_ids
        total_local_ids = torch.tensor([2, 0, 3, 1], dtype=torch.long, device=device)
        local_queries = torch.empty(
            4,
            index_n_heads,
            index_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        local_queries[total_local_ids] = global_queries[selected_global_ids]

        gathered_order = torch.tensor(
            [0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.long, device=device
        )
        restore_indices = torch.argsort(gathered_order)
        slot_mapping = torch.arange(
            prefix_length,
            total_kv_tokens,
            dtype=torch.int64,
            device=device,
        )
        q_fp8, q_scale = op.quant_q_k_cp(
            local_queries,
            local_keys,
            cache,
            slot_mapping,
            restore_indices,
        )

        block_table = torch.arange(
            page_count, dtype=torch.int32, device=device
        ).unsqueeze(0)
        cu_kv_seqlens = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32, device=device
        )
        attention_inputs = PyAttentionInputs()
        attention_inputs.kv_cache_kernel_block_id_device = block_table

        row_starts = torch.zeros(4, dtype=torch.int32, device=device)
        row_ends = (prefix_length + selected_global_ids + 1).to(torch.int32)
        lengths = row_ends.clone()
        topk_offsets = torch.zeros(4, dtype=torch.int32, device=device)

        class FmhaParams:
            pass

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            q_scale,
            cache,
            FmhaParams(),
            attention_inputs,
            total_local_ids,
            cu_kv_seqlens,
            total_kv_tokens,
            row_starts,
            row_ends,
            lengths,
            topk_offsets,
        )

        gathered_k = torch.empty(
            total_kv_tokens,
            index_head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        gathered_scale_bytes = torch.empty(
            total_kv_tokens, 4, dtype=torch.uint8, device=device
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            op._indexer_cache_view(cache),
            gathered_k,
            gathered_scale_bytes,
            block_table,
            cu_kv_seqlens,
        )
        gathered_dequant = gathered_k.float() * gathered_scale_bytes.view(torch.float32)
        logical_keys = torch.cat([prefix_keys, input_keys], dim=0)
        torch.testing.assert_close(
            gathered_dequant,
            _expected_ue8m0_dequant(logical_keys),
            rtol=0,
            atol=0,
        )

        reference_logits = deep_gemm.fp8_mqa_logits(
            q_fp8[total_local_ids].contiguous(),
            (gathered_k, gathered_scale_bytes.view(torch.float32)),
            q_scale[total_local_ids].squeeze(-1).contiguous(),
            row_starts,
            row_ends,
            clean_logits=False,
        )
        expected_topk = fast_topk_transform_ragged_fused(
            score=reference_logits,
            lengths=lengths,
            topk_indices_offset=topk_offsets,
            topk=index_topk,
            row_starts=row_starts,
        )
        torch.testing.assert_close(
            torch.sort(topk_result, dim=-1).values,
            torch.sort(expected_topk, dim=-1).values,
            rtol=0,
            atol=0,
        )
        torch.cuda.synchronize(device)
    finally:
        if initialized:
            destroy_distributed_environment()


class GetTopkRaggedCPTest(TestCase):
    """Test IndexerOp._get_topk_ragged_cp with single rank and one chunk."""

    def setUp(self):
        if not CUDA_DEEPGEMM_OK:
            raise SkipTest(SKIP_REASON)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)

    def test_get_topk_ragged_cp_shape_and_no_crash(self):
        """
        Single CP rank, one chunk of 8 tokens.
        generate_q_indices([8]) -> q0_idx=[0,1,2,3], q1_idx=[4,5,6,7];
        q_fp8 has 8 rows, so index_select is valid.
        """
        # deep_gemm fp8_mqa_logits requires seq_len_alignment % block_q == 0 with block_q = 128/num_heads; use 32 heads so block_q=4.
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        total_tokens = 8
        chunk_lengths = [8]

        op = IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            cos_sin_cache=None,
            blocksize=64,
            block_size=block_size,
        )

        device = self.device
        total_local_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
        total_global_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
        cu_kv_seqlens_global = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )
        q_fp8 = torch.randn(
            total_tokens,
            index_n_heads,
            index_head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            total_tokens, index_n_heads, 1, dtype=torch.float32, device=device
        )

        num_blocks = 1
        page_size = 64
        cache_stride = index_head_dim + (index_head_dim // block_size) * 4
        kv_cache = LayerKVCache(
            torch.empty(
                num_blocks,
                page_size * cache_stride,
                dtype=torch.uint8,
                device=device,
            ),
            page_size,
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_kernel_block_id = torch.tensor(
            [[0]], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[0]], dtype=torch.int32, device=device
        )
        attn_inputs.cu_kv_seqlens_device = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )
        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        attn_inputs.context_parallel_info = cp_params

        ks = torch.arange(total_tokens, dtype=torch.int32, device=device)
        ke = torch.arange(1, total_tokens + 1, dtype=torch.int32, device=device)
        expanded_seq_lens = torch.ones(total_tokens, dtype=torch.int32, device=device)
        topk_indices_offset = torch.zeros(
            total_tokens, dtype=torch.int32, device=device
        )

        class FmhaParams:
            pass

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = expanded_seq_lens
        fmha_params.topk_indices_offset = topk_indices_offset

        q0_idx_list, _q1_idx_list = generate_q_indices(chunk_lengths)
        n0 = len(q0_idx_list)

        # Precompute indexed params (simulates what create_params does)
        precomputed_ks = fmha_params.ks[total_global_ids]
        precomputed_ke = fmha_params.ke[total_global_ids]
        precomputed_lengths = fmha_params.expanded_seq_lens[total_global_ids]
        precomputed_topk_off = fmha_params.topk_indices_offset[total_global_ids]

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            total_local_ids,
            cu_kv_seqlens_global,
            total_tokens,
            precomputed_ks,
            precomputed_ke,
            precomputed_lengths,
            precomputed_topk_off,
        )
        topk0 = topk_result[:n0]
        topk1 = topk_result[n0:]

        self.assertIsInstance(topk0, torch.Tensor)
        self.assertIsInstance(topk1, torch.Tensor)
        self.assertEqual(topk0.dtype, torch.int32)
        self.assertEqual(topk1.dtype, torch.int32)
        self.assertEqual(topk0.device, q_fp8.device)
        self.assertEqual(topk1.device, q_fp8.device)
        self.assertEqual(topk0.dim(), 2)
        self.assertEqual(topk1.dim(), 2)
        self.assertEqual(topk0.shape[1], index_topk)
        self.assertEqual(topk1.shape[1], index_topk)
        self.assertEqual(
            topk0.shape[0], 4, "q0 has 4 rows from generate_q_indices([8])"
        )
        self.assertEqual(
            topk1.shape[0], 4, "q1 has 4 rows from generate_q_indices([8])"
        )

    def test_get_topk_ragged_cp_with_prefix_cache(self):
        """
        Single CP rank, one chunk of 8 tokens with prefix_length=64 (page-aligned).
        Total KV = prefix(64) + input(8) = 72 tokens.
        q_fp8 has 8 rows (input only), topk computed over full 72-token KV range.
        ks/ke shifted by prefix_length so each query row sees prefix + its own context.
        """
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        input_tokens = 8
        prefix_length = 64
        total_kv_tokens = input_tokens + prefix_length  # 72
        chunk_lengths = [8]

        op = IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            cos_sin_cache=None,
            blocksize=64,
            block_size=block_size,
        )

        device = self.device
        total_local_ids = torch.arange(input_tokens, device=device, dtype=torch.long)
        total_global_ids = torch.arange(input_tokens, device=device, dtype=torch.long)
        # cu_kv_seqlens_global includes prefix: [0, prefix + input]
        cu_kv_seqlens_global = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32, device=device
        )

        q_fp8 = torch.randn(
            input_tokens,
            index_n_heads,
            index_head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            input_tokens, index_n_heads, 1, dtype=torch.float32, device=device
        )

        # Allocate enough blocks for total_kv_tokens
        import math

        page_size = 64
        num_blocks = math.ceil(total_kv_tokens / page_size)
        cache_stride = index_head_dim + (index_head_dim // block_size) * 4
        kv_cache = LayerKVCache(
            torch.empty(
                num_blocks,
                page_size * cache_stride,
                dtype=torch.uint8,
                device=device,
            ),
            page_size,
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_kernel_block_id_device = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        attn_inputs.cu_kv_seqlens_device = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32, device=device
        )
        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        attn_inputs.context_parallel_info = cp_params

        # ks/ke shifted by prefix_length: each query token i sees KV[prefix+i : prefix+i+1]
        ks = (
            torch.arange(input_tokens, dtype=torch.int32, device=device) + prefix_length
        )
        ke = (
            torch.arange(1, input_tokens + 1, dtype=torch.int32, device=device)
            + prefix_length
        )
        expanded_seq_lens = torch.ones(input_tokens, dtype=torch.int32, device=device)
        topk_indices_offset = torch.zeros(
            input_tokens, dtype=torch.int32, device=device
        )

        class FmhaParams:
            pass

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = expanded_seq_lens
        fmha_params.topk_indices_offset = topk_indices_offset

        q0_idx_list, _q1_idx_list = generate_q_indices(chunk_lengths)
        n0 = len(q0_idx_list)

        # Precompute indexed params (simulates what create_params does)
        precomputed_ks = fmha_params.ks[total_global_ids]
        precomputed_ke = fmha_params.ke[total_global_ids]
        precomputed_lengths = fmha_params.expanded_seq_lens[total_global_ids]
        precomputed_topk_off = fmha_params.topk_indices_offset[total_global_ids]

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            total_local_ids,
            cu_kv_seqlens_global,
            total_kv_tokens,
            precomputed_ks,
            precomputed_ke,
            precomputed_lengths,
            precomputed_topk_off,
        )
        topk0 = topk_result[:n0]
        topk1 = topk_result[n0:]

        self.assertIsInstance(topk0, torch.Tensor)
        self.assertIsInstance(topk1, torch.Tensor)
        self.assertEqual(topk0.dtype, torch.int32)
        self.assertEqual(topk1.dtype, torch.int32)
        self.assertEqual(topk0.device, q_fp8.device)
        self.assertEqual(topk1.device, q_fp8.device)
        self.assertEqual(topk0.dim(), 2)
        self.assertEqual(topk1.dim(), 2)
        self.assertEqual(topk0.shape[1], index_topk)
        self.assertEqual(topk1.shape[1], index_topk)
        self.assertEqual(
            topk0.shape[0], 4, "q0 has 4 rows from generate_q_indices([8])"
        )
        self.assertEqual(
            topk1.shape[0], 4, "q1 has 4 rows from generate_q_indices([8])"
        )

    @skipIf(torch.cuda.device_count() < 2, "need 2 CUDA devices")
    def test_tp2_restore_and_topk_match_reference(self):
        context = mp.get_context("spawn")
        port_manager = PortManager()
        ports, locks = port_manager.get_consecutive_ports(1)
        processes = [
            context.Process(target=_tp2_indexer_worker, args=(rank, ports[0]))
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=300)
                self.assertEqual(process.exitcode, 0, process.name)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
