"""
Unit tests for ZigZagSparseMlaFp8CPOp (Context Parallel prefill for Sparse MLA FP8).

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
        op.attn_inputs = ai
        op.plan(mla, bt, ai)
        t0 = torch.index_select(g_topk, 0, op.q0_idx).contiguous()
        t1 = torch.index_select(g_topk, 0, op.q1_idx).contiguous()
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
    """Test ZigZagSparseMlaFp8CPOp with single rank (tp_size=1)."""

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
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            ZigZagSparseMlaFp8CPOp,
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

        cp_op = ZigZagSparseMlaFp8CPOp(
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
        cp_op.attn_inputs = attn_inputs

        cp_op.plan(mla_params, block_table_device, attn_inputs)

        # With tp_size=1, all_gather is identity; mock it to avoid requiring distributed init
        def _identity_all_gather(tensor, group=None):
            return tensor

        topk0 = torch.index_select(topk_indices, 0, cp_op.q0_idx).contiguous()
        topk1 = torch.index_select(topk_indices, 0, cp_op.q1_idx).contiguous()
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
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            ZigZagSparseMlaFp8CPOp,
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

        cp_op = ZigZagSparseMlaFp8CPOp(
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
        cp_op.attn_inputs = attn_inputs
        cp_op.plan(mla_params, block_table_device, attn_inputs)

        def _identity_all_gather(tensor, group=None):
            return tensor

        topk0 = torch.index_select(topk_indices, 0, cp_op.q0_idx).contiguous()
        topk1 = torch.index_select(topk_indices, 0, cp_op.q1_idx).contiguous()
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

    def test_cp_op_forward_with_prefix_cache(self):
        """
        With tp_size=1 and prefix_lengths > 0 (reuse cache), verify that:
        1. The CP op does not crash (buffer allocation uses full KV length)
        2. Output shape is correct
        3. Output matches non-CP reference on the same cache state

        This exercises the fix where cu_kv_seqlens_global[-1] (prefix + new)
        is used for buffer sizing instead of kv_restore_unpad_indices.shape[0]
        (new tokens only).
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            ZigZagSparseMlaFp8CPOp,
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
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        fp8_bytes_per_token = 656

        prefix_len = 64
        new_tokens = 8
        total_kv_len = prefix_len + new_tokens
        num_blocks_needed = math.ceil(total_kv_len / page_size)

        chunk_lengths = [new_tokens]
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
            [new_tokens], dtype=torch.int32, device=torch.device("cpu")
        )

        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = prefill_cp_chunk_lengths
        cp_params.prefill_qkv_restore_indice = prefill_qkv_restore_indice
        cp_params.prefill_qkv_padding_mask = prefill_qkv_padding_mask
        cp_params.prefill_actual_input_lengths_cpu = prefill_actual_input_lengths_cpu

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
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_base

        cp_op = ZigZagSparseMlaFp8CPOp(
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
            "CP output with prefix cache should match non-CP output",
        )


if __name__ == "__main__":
    main()
