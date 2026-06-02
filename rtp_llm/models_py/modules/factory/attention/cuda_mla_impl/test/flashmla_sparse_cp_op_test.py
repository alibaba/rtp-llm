"""
Unit tests for SparseMlaFp8CPOp (Context Parallel prefill for Sparse MLA FP8).

- tp_size=1: mock all_gather, compare to non-CP.
- tp_size=2: two processes, real all_gather; merge outputs vs single-GPU non-CP (needs >=2 GPUs).

Usage:
    python -m pytest rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_sparse_cp_op_test.py -v
    python -m unittest rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_sparse_cp_op_test
"""

import functools
import logging
import math
import multiprocessing as mp
import os
import threading
from datetime import timedelta
from typing import List, Optional, Tuple
from unittest import SkipTest, TestCase, main, skipIf
from unittest.mock import patch

import numpy as np
import torch

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.ops import KvCacheDataType, NcclCommConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)
from rtp_llm.test.utils.port_util import PortManager


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


def _tp2_worker(rank: int, nccl_port: int, result_queue: mp.Queue) -> None:
    """Minimal tp_size=2 CP forward; pushes (rank, local_ids, out) and rank0 pushes ref."""
    try:
        import torch.distributed as dist

        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
            generate_q_indices,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        n_gpu = torch.cuda.device_count()
        local_rank = rank % n_gpu
        torch.cuda.set_device(local_rank)
        pc = ParallelismConfig()
        base = nccl_port + 11
        pc.world_rank = rank
        pc.world_size = 2
        pc.local_rank = local_rank
        pc.tp_rank = rank
        pc.tp_size = 2
        pc.dp_rank = 0
        pc.dp_size = 1
        pc.prefill_cp_config = PrefillCPConfig()
        pc.prefill_cp_config.method = CPRotateMethod.DISABLED
        pc.prefill_cp_config.comm_buffer_size = 0
        dev = torch.device(f"cuda:{local_rank}")
        torch.set_default_device(dev)
        torch.set_default_dtype(torch.bfloat16)

        _orig_init_pg = dist.init_process_group

        @functools.wraps(_orig_init_pg)
        def _init_pg(*args, **kwargs):
            kw = dict(kwargs)
            backend = kw.get("backend") or (args[0] if args else None)
            if backend == "nccl" and "device_id" not in kw:
                kw["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
                try:
                    return _orig_init_pg(*args, **kw)
                except TypeError:
                    return _orig_init_pg(*args, **kwargs)
            return _orig_init_pg(*args, **kwargs)

        _barrier_to = timedelta(minutes=3)

        def _barrier() -> None:
            torch.cuda.synchronize()
            try:
                dist.barrier(timeout=_barrier_to)
            except TypeError:
                dist.barrier()

        with patch.object(dist, "init_process_group", _init_pg):
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.init_symm_mem_communicator",
                lambda *_a, **_kw: None,
            ):
                init_distributed_environment(
                    pc,
                    NcclCommConfig(
                        nccl_ip="127.0.0.1",
                        tp_nccl_port=base - 2,
                        dp_tp_nccl_port=base - 10,
                        ffn_tp_nccl_port=base - 5,
                    ),
                    nccl_init_port=base - 11,
                    backend="nccl",
                    timeout=120,
                )

        _set_seed(42)
        H, R, dh, page, topk, T = 64, 512, 64, 64, 128, 8
        qk = 512 + 64
        chunks = [2, 2]
        tok = sum(chunks)
        # Same global tensors on every rank (per-device CUDA RNG can diverge).
        cp = PyContextParallelParams()
        cp.prefill_cp_chunk_lengths = torch.tensor(
            chunks, dtype=torch.int32, device=dev
        )
        cp.prefill_qkv_restore_indice = torch.arange(T, dtype=torch.long, device=dev)
        cp.prefill_qkv_padding_mask = torch.ones(T, dtype=torch.int32, device=dev)
        cpu = torch.device("cpu")
        cp.prefill_actual_input_lengths_cpu = torch.tensor(
            [T], dtype=torch.int32, device=cpu
        )
        ai = PyAttentionInputs()
        ai.is_prefill = True
        ai.input_lengths = torch.tensor([T], dtype=torch.int32, device=cpu)
        ai.sequence_lengths = torch.tensor([T], dtype=torch.int32, device=cpu)
        ai.prefix_lengths = torch.tensor([0], dtype=torch.int32, device=cpu)
        ai.context_parallel_info = cp
        block_table_host = _make_block_table(1, T, page, cpu)
        block_table_device = block_table_host.to(dev)
        ai.kv_cache_block_id_host = block_table_host
        ai.kv_cache_block_id_device = block_table_device
        ai.kv_cache_kernel_block_id_host = block_table_host
        ai.kv_cache_kernel_block_id_device = block_table_device
        bt = block_table_device
        mla = rtp_llm_ops.SparseMlaParams()
        mla.fill_params(ai, page)

        g_q = (torch.randn(T, H, qk, dtype=torch.bfloat16, device=cpu) * 0.1).to(dev)
        g_ckv = (torch.randn(T, R, dtype=torch.bfloat16, device=cpu) * 0.1).to(dev)
        g_kpe = (torch.randn(T, dh, dtype=torch.bfloat16, device=cpu) * 0.1).to(dev)
        g_topk = torch.randint(0, T, (T, 1, topk), dtype=torch.int32, device=cpu).to(
            dev
        )
        bid = torch.zeros(T, dtype=torch.int32, device=dev)
        sl = slice(rank * tok, (rank + 1) * tok)
        ck = g_ckv[sl].contiguous()
        kp = g_kpe[sl].contiguous()
        nb = bt.shape[1]
        kvb = torch.empty(nb, page, 656, dtype=torch.uint8, device=dev)
        kvc = LayerKVCache()
        kvc.kv_cache_base = kvb
        op = SparseMlaFp8CPOp(
            num_heads=H,
            kv_lora_rank=R,
            qk_rope_head_dim=dh,
            qk_nope_head_dim=512,
            page_size=page,
            softmax_extra_scale=1.0,
            top_k=topk,
            parallelism_config=pc,
        )
        op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        op.write_cache_store_impl = None
        op.plan(mla, bt, attn_inputs=ai)
        q0_idx_list, q1_idx_list = generate_q_indices(chunks)
        q0_t = torch.tensor(q0_idx_list, device=dev, dtype=torch.long)
        q1_t = torch.tensor(q1_idx_list, device=dev, dtype=torch.long)
        t0 = torch.index_select(g_topk, 0, q0_t).contiguous()
        t1 = torch.index_select(g_topk, 0, q1_t).contiguous()
        out = op.forward(g_q, ck, kp, torch.cat([t0, t1], dim=0), bid, kvc, layer_id=0)
        torch.cuda.synchronize()
        lids = op.total_local_ids
        result_queue.put((rank, lids.cpu().numpy(), out.cpu().float().numpy()))
        _barrier()
        if rank == 0:
            d0 = torch.device("cuda:0")
            refb = torch.empty_like(kvb, device=d0)
            refl = LayerKVCache()
            refl.kv_cache_base = refb
            wo = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
            torch.cuda.set_device(d0)
            wo.forward(g_ckv, g_kpe, refl, mla)
            kvf = refb.view(-1, 1, refb.size(-1))
            if kvf.ndim == 3:
                kvf = kvf.unsqueeze(-2)
            nco = SparseMlaFp8Op(
                num_heads=H,
                kv_lora_rank=R,
                qk_rope_head_dim=dh,
                qk_nope_head_dim=512,
                page_size=page,
                softmax_extra_scale=1.0,
                top_k=topk,
            )
            nco.plan(mla, block_table_host.to(d0))
            oref = nco.forward(g_q, kvf, g_topk, layer_id=0)
            torch.cuda.synchronize()
            result_queue.put(("ref", oref.cpu().float().numpy()))
        _barrier()
        destroy_distributed_environment()
    except Exception:
        logging.exception("tp2 rank %s", rank)
        result_queue.put(("err", str(rank)))
        raise


@skipIf(not CUDA_FLASHMLA_OK, SKIP_REASON)
class SparseMlaFp8CPOpTest(TestCase):
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

    def test_cp_op_forward_shape_and_match_non_cp(self):
        """
        With tp_size=1, CP path all_gather is identity.
        Run both CP op and non-CP op on same inputs and check output shape and equality.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
            generate_q_indices,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        _set_seed(42)
        device = self.device

        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        total_q_len = 8
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        chunk_lengths = [4, 4]
        prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        n_restore = sum(chunk_lengths)
        prefill_qkv_restore_indice = torch.arange(
            n_restore, dtype=torch.long, device=device
        )
        prefill_qkv_padding_mask = torch.ones(
            n_restore, dtype=torch.int32, device=device
        )
        prefill_actual_input_lengths_cpu = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = prefill_cp_chunk_lengths
        cp_params.prefill_qkv_restore_indice = prefill_qkv_restore_indice
        cp_params.prefill_qkv_padding_mask = prefill_qkv_padding_mask
        cp_params.prefill_actual_input_lengths_cpu = prefill_actual_input_lengths_cpu

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [0], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params

        block_table_host = _make_block_table(
            1, n_restore, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device
        attn_inputs.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs.kv_cache_kernel_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0

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
            0, n_restore, (total_q_len, 1, top_k), dtype=torch.int32, device=device
        )
        batch_indice_d = torch.zeros(total_q_len, dtype=torch.int32, device=device)

        num_blocks = block_table_host.shape[1]
        fp8_bytes_per_token = 656
        kv_cache_base = torch.empty(
            num_blocks, page_size, fp8_bytes_per_token, dtype=torch.uint8, device=device
        )
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_base

        cp_op = SparseMlaFp8CPOp(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
            parallelism_config=parallelism_config,
        )
        cp_op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        cp_op.write_cache_store_impl = None

        cp_op.plan(mla_params, block_table_device, attn_inputs=attn_inputs)

        # With tp_size=1, all_gather is identity; mock it to avoid requiring distributed init
        def _identity_all_gather(tensor, group=None):
            return tensor

        q0_idx_list, q1_idx_list = generate_q_indices(chunk_lengths)
        q0_idx_t = torch.tensor(q0_idx_list, device=device, dtype=torch.long)
        q1_idx_t = torch.tensor(q1_idx_list, device=device, dtype=torch.long)
        topk0 = torch.index_select(topk_indices, 0, q0_idx_t).contiguous()
        topk1 = torch.index_select(topk_indices, 0, q1_idx_t).contiguous()
        # CP forward expects single topk tensor aligned with total_local_ids (q0 then q1)
        topk_cat = torch.cat([topk0, topk1], dim=0)
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out_cp = cp_op.forward(
                q,
                compressed_kv,
                k_pe,
                topk_cat,
                batch_indice_d,
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(out_cp.shape, (total_q_len, num_heads, kv_lora_rank))

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
            "CP output should match non-CP when tp_size=1 (all_gather identity)",
        )

    def test_cp_op_forward_output_shape(self):
        """CP op forward returns correct shape [total_q_len, num_heads, kv_lora_rank]."""
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
            generate_q_indices,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        _set_seed(123)
        device = self.device
        # Use same FP8 layout as test_cp_op_forward_shape_and_match_non_cp (V32: 576 d, 656 bytes)
        total_q_len = 6
        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        top_k = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # generate_kv_indices requires each chunk length to be even
        chunk_lengths = [4, 2]
        n_restore = total_q_len

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        cp_params.prefill_qkv_restore_indice = torch.arange(
            n_restore, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.ones(
            n_restore, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [n_restore], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [0], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params
        block_table_host = _make_block_table(
            1, n_restore, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device
        attn_inputs.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs.kv_cache_kernel_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

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
            0, n_restore, (total_q_len, 1, top_k), dtype=torch.int32, device=device
        )
        batch_indice_d = torch.zeros(total_q_len, dtype=torch.int32, device=device)
        num_blocks = block_table_host.shape[1]
        fp8_bytes_per_token = 656
        kv_cache_base = torch.empty(
            num_blocks, page_size, fp8_bytes_per_token, dtype=torch.uint8, device=device
        )
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_base

        cp_op = SparseMlaFp8CPOp(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=1.0,
            top_k=top_k,
            parallelism_config=parallelism_config,
        )
        cp_op.kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)
        cp_op.write_cache_store_impl = None
        cp_op.plan(mla_params, block_table_device, attn_inputs=attn_inputs)

        def _identity_all_gather(tensor, group=None):
            return tensor

        q0_idx_list, q1_idx_list = generate_q_indices(chunk_lengths)
        q0_idx_t = torch.tensor(q0_idx_list, device=device, dtype=torch.long)
        q1_idx_t = torch.tensor(q1_idx_list, device=device, dtype=torch.long)
        topk0 = torch.index_select(topk_indices, 0, q0_idx_t).contiguous()
        topk1 = torch.index_select(topk_indices, 0, q1_idx_t).contiguous()
        topk_cat = torch.cat([topk0, topk1], dim=0)
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_identity_all_gather,
        ):
            out = cp_op.forward(
                q,
                compressed_kv,
                k_pe,
                topk_cat,
                batch_indice_d,
                kv_cache,
                layer_id=0,
            )
        torch.cuda.synchronize()
        self.assertEqual(out.shape, (total_q_len, num_heads, kv_lora_rank))

    @skipIf(torch.cuda.device_count() < 2, "need 2 CUDA devices")
    def test_cp_tp2_matches_non_cp(self):
        """tp_size=2 real all_gather; merged CP vs single-GPU SparseMlaFp8Op."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        port_mgr = PortManager()
        ports, locks = port_mgr.get_consecutive_ports(1)
        qu: mp.Queue = mp.Queue()
        procs: List[mp.Process] = []
        drained: List[object] = []
        drain_exc: List[BaseException] = []

        def _drain_queue() -> None:
            try:
                for _ in range(3):
                    drained.append(qu.get(timeout=400))
            except BaseException as e:
                drain_exc.append(e)

        drain_t = threading.Thread(target=_drain_queue, daemon=True)
        try:
            drain_t.start()
            for r in range(2):
                procs.append(
                    mp.Process(
                        target=_tp2_worker,
                        args=(r, ports[0], qu),
                        daemon=False,
                    )
                )
                procs[-1].start()
            for p in procs:
                p.join(timeout=300)
                self.assertEqual(p.exitcode, 0, getattr(p, "name", p))
            drain_t.join(timeout=420)
        finally:
            for lk in locks:
                lk.__exit__(None, None, None)
        if drain_exc:
            raise drain_exc[0]
        self.assertEqual(
            len(drained),
            3,
            f"expected 3 queue messages, got {len(drained)} (avoid parent/child queue deadlock)",
        )
        by_rank: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        ref: Optional[np.ndarray] = None
        for it in drained:
            if it[0] == "err":
                self.fail(str(it[1]))
            if it[0] == "ref":
                ref = it[1]
            else:
                by_rank[int(it[0])] = (it[1], it[2])
        self.assertEqual(len(by_rank), 2)
        self.assertIsNotNone(ref)
        assert ref is not None
        t, h, rdim = ref.shape
        mg = np.zeros((t, h, rdim), dtype=np.float32)
        for ri in (0, 1):
            lids, full = by_rank[ri]
            ix = lids.astype(np.int64)
            mg[ix] = full[ix]
        # Split CP attention vs one-shot non-CP: FP8 + different kernel tiling (max ~0.11 seen).
        diff = float(np.max(np.abs(mg - ref)))
        self.assertTrue(
            np.allclose(mg, ref, atol=0.15, rtol=0.05),
            f"merged CP vs ref mismatch, max_abs_diff={diff}",
        )

    def test_cp_tp2_prefix_cache_matches_non_cp(self):
        """
        Single-GPU mock of tp_size=2 with prefix_length=64.
        all_gather is mocked as repeat(2, ...) to simulate 2-rank concatenation.
        Total KV = prefix(64) + input(8) = 72 tokens.
        Each CP rank holds 4 local tokens (chunks=[2,2]), all_gather doubles to 8.
        Compare CP output (rank 0) against non-CP reference on full 72 tokens.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
            generate_q_indices,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig

        _set_seed(42)
        device = self.device

        num_heads = 64
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 512
        page_size = 64
        softmax_extra_scale = 1.0
        top_k = 128
        prefix_length = 64
        input_tokens = 8
        total_kv_len = prefix_length + input_tokens  # 72
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        # tp_size=2, rank=0 holds first half of input tokens
        tp_size = 2
        tp_rank = 0
        chunk_lengths = [2, 2]  # per-rank local chunks
        tok = sum(chunk_lengths)  # 4 local tokens per rank

        # --- Generate full KV data for all 72 tokens ---
        all_compressed_kv = (
            torch.randn(total_kv_len, kv_lora_rank, dtype=torch.bfloat16, device=device)
            * 0.1
        )
        all_k_pe = (
            torch.randn(
                total_kv_len, qk_rope_head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        prefix_ckv = all_compressed_kv[:prefix_length]
        prefix_k_pe = all_k_pe[:prefix_length]
        input_ckv = all_compressed_kv[prefix_length:]
        input_k_pe = all_k_pe[prefix_length:]

        # Rank 0's local chunk of input KV
        local_ckv = input_ckv[:tok].contiguous()
        local_k_pe = input_k_pe[:tok].contiguous()

        q = (
            torch.randn(
                input_tokens,
                num_heads,
                qk_head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )
        batch_indice_d = torch.zeros(input_tokens, dtype=torch.int32, device=device)

        block_table_host = _make_block_table(
            1, total_kv_len, page_size, torch.device("cpu")
        )
        block_table_device = block_table_host.to(device)
        num_blocks = block_table_host.shape[1]

        # --- CP attn_inputs: input_lengths = local tokens, sequence_lengths = full ---
        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        # restore_indice maps all-gathered (tp_size * tok) positions back to global order.
        # With identity restore (tokens already in order), it's just arange(tp_size * tok).
        cp_params.prefill_qkv_restore_indice = torch.arange(
            tp_size * tok, dtype=torch.long, device=device
        )
        cp_params.prefill_qkv_padding_mask = torch.ones(
            tp_size * tok, dtype=torch.int32, device=device
        )
        cp_params.prefill_actual_input_lengths_cpu = torch.tensor(
            [input_tokens], dtype=torch.int32, device=torch.device("cpu")
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [input_tokens], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [prefix_length], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.context_parallel_info = cp_params
        attn_inputs.kv_cache_block_id_host = block_table_host
        attn_inputs.kv_cache_block_id_device = block_table_device
        attn_inputs.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs.kv_cache_kernel_block_id_device = block_table_device

        mla_params = rtp_llm_ops.SparseMlaParams()
        mla_params.fill_params(attn_inputs, page_size)

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_rank = tp_rank
        parallelism_config.tp_size = tp_size
        parallelism_config.prefill_cp_config = PrefillCPConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallelism_config.prefill_cp_config.comm_buffer_size = 0

        kv_cache_write_op = MlaKVCacheWriteOp(kv_cache_dtype=KvCacheDataType.FP8)

        # ============================================================
        # Non-CP reference: write all 72 tokens, run non-CP forward
        # ============================================================
        # Write all KV tokens using full-sequence params
        attn_inputs_write = PyAttentionInputs()
        attn_inputs_write.is_prefill = True
        attn_inputs_write.input_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_write.sequence_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_write.prefix_lengths = torch.tensor(
            [0], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_write.kv_cache_block_id_host = block_table_host
        attn_inputs_write.kv_cache_block_id_device = block_table_device
        attn_inputs_write.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs_write.kv_cache_kernel_block_id_device = block_table_device

        mla_params_write = rtp_llm_ops.SparseMlaParams()
        mla_params_write.fill_params(attn_inputs_write, page_size)

        kv_cache_ref = LayerKVCache()
        kv_cache_ref.kv_cache_base = torch.empty(
            num_blocks, page_size, fp8_bytes_per_token, dtype=torch.uint8, device=device
        )
        kv_cache_write_op.forward(
            all_compressed_kv, all_k_pe, kv_cache_ref, mla_params_write
        )
        torch.cuda.synchronize()

        # plan() needs mla_params matching q token count (input_tokens)
        attn_inputs_ref = PyAttentionInputs()
        attn_inputs_ref.is_prefill = True
        attn_inputs_ref.input_lengths = torch.tensor(
            [input_tokens], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_ref.sequence_lengths = torch.tensor(
            [total_kv_len], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_ref.prefix_lengths = torch.tensor(
            [prefix_length], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_ref.kv_cache_block_id_host = block_table_host
        attn_inputs_ref.kv_cache_block_id_device = block_table_device
        attn_inputs_ref.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs_ref.kv_cache_kernel_block_id_device = block_table_device

        mla_params_ref = rtp_llm_ops.SparseMlaParams()
        mla_params_ref.fill_params(attn_inputs_ref, page_size)

        topk_indices = torch.randint(
            0, total_kv_len, (input_tokens, 1, top_k), dtype=torch.int32, device=device
        )

        non_cp_op = SparseMlaFp8Op(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )
        non_cp_op.plan(mla_params_ref, block_table_device)

        kv_cache_ref_flat = kv_cache_ref.kv_cache_base.view(
            -1, 1, kv_cache_ref.kv_cache_base.size(-1)
        )
        if kv_cache_ref_flat.ndim == 3:
            kv_cache_ref_flat = kv_cache_ref_flat.unsqueeze(-2)
        out_non_cp = non_cp_op.forward(q, kv_cache_ref_flat, topk_indices, layer_id=0)
        torch.cuda.synchronize()

        # ============================================================
        # CP path (rank 0): pre-fill prefix, mock all_gather as repeat
        # ============================================================
        kv_cache_cp = LayerKVCache()
        kv_cache_cp.kv_cache_base = torch.empty(
            num_blocks, page_size, fp8_bytes_per_token, dtype=torch.uint8, device=device
        )

        # Pre-fill prefix tokens
        attn_inputs_prefix = PyAttentionInputs()
        attn_inputs_prefix.is_prefill = True
        attn_inputs_prefix.input_lengths = torch.tensor(
            [prefix_length], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_prefix.sequence_lengths = torch.tensor(
            [prefix_length], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_prefix.prefix_lengths = torch.tensor(
            [0], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs_prefix.kv_cache_block_id_host = block_table_host
        attn_inputs_prefix.kv_cache_block_id_device = block_table_device
        attn_inputs_prefix.kv_cache_kernel_block_id_host = block_table_host
        attn_inputs_prefix.kv_cache_kernel_block_id_device = block_table_device

        mla_params_prefix = rtp_llm_ops.SparseMlaParams()
        mla_params_prefix.fill_params(attn_inputs_prefix, page_size)
        kv_cache_write_op.forward(
            prefix_ckv, prefix_k_pe, kv_cache_cp, mla_params_prefix
        )
        torch.cuda.synchronize()

        cp_op = SparseMlaFp8CPOp(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
            parallelism_config=parallelism_config,
        )
        cp_op.kv_cache_write_op = kv_cache_write_op
        cp_op.write_cache_store_impl = None
        cp_op.attn_inputs = attn_inputs
        cp_op.plan(mla_params, block_table_device, attn_inputs)

        # Mock all_gather: return the full input KV (all 8 tokens) as if gathered
        # from 2 ranks (rank0 has first 4, rank1 has last 4).
        _all_gather_returns = iter([input_ckv.contiguous(), input_k_pe.contiguous()])

        def _mock_all_gather_tp2(tensor, group=None):
            return next(_all_gather_returns)

        q0_idx_list, q1_idx_list = generate_q_indices(chunk_lengths)
        q0_idx_t = torch.tensor(q0_idx_list, device=device, dtype=torch.long)
        q1_idx_t = torch.tensor(q1_idx_list, device=device, dtype=torch.long)
        topk0 = torch.index_select(topk_indices, 0, q0_idx_t).contiguous()
        topk1 = torch.index_select(topk_indices, 0, q1_idx_t).contiguous()
        topk_cat = torch.cat([topk0, topk1], dim=0)
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl.all_gather",
            side_effect=_mock_all_gather_tp2,
        ):
            out_cp = cp_op.forward(
                q,
                local_ckv,
                local_k_pe,
                topk_cat,
                batch_indice_d,
                kv_cache_cp,
                layer_id=0,
            )
        torch.cuda.synchronize()

        self.assertEqual(out_cp.shape, (input_tokens, num_heads, kv_lora_rank))
        # CP rank 0 only fills total_local_ids positions; compare those against non-CP ref.
        local_ids = cp_op.total_local_ids
        self.assertTrue(
            torch.allclose(
                out_cp[local_ids], out_non_cp[local_ids], atol=1e-2, rtol=1e-2
            ),
            "CP tp_size=2 (mocked) with prefix cache should match non-CP reference at rank-0 positions",
        )


if __name__ == "__main__":
    main()
