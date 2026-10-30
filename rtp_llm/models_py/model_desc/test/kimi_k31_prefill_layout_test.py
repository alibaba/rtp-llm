"""Real TP transport for non-Query-CP sparse Prefill (including PDFUSION)."""

import unittest
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
)
from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.test.utils.port_util import PortManager


def _prefill_layout_worker(rank, port):
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_size = parallel.tp_size = parallel.local_world_size = 2
    parallel.world_rank = parallel.local_rank = rank
    parallel.role_type = RoleType.PDFUSION
    parallel.prefill_cp_config.kv_cache_sharded = True
    init_distributed_environment(
        parallel,
        NcclCommConfig(nccl_ip="127.0.0.1"),
        port,
        timeout=60,
    )
    try:
        op = PageRRSparseMlaOp(
            num_heads=4,
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            page_size=128,
            softmax_extra_scale=1.0,
            top_k=8,
            parallelism_config=parallel,
            workspace_owner=SimpleNamespace(
                prefill_query_cp=False, sfa_metadata_free=False
            ),
        )
        full = (
            torch.arange(8 * 8 * 16, dtype=torch.float32, device=f"cuda:{rank}")
            .reshape(8, 8, 16)
            .bfloat16()
        )
        head_shard = full[:, rank * 4 : (rank + 1) * 4].contiguous()
        row_shard = op._prefill(head_shard)
        torch.testing.assert_close(
            row_shard, full[rank * 4 : (rank + 1) * 4], rtol=0, atol=0
        )
        restored = op.communicator.restore_queries(row_shard)
        torch.testing.assert_close(restored, head_shard, rtol=0, atol=0)
        noncontiguous = full[:, rank * 4 : (rank + 1) * 4]
        torch.testing.assert_close(
            op._prefill(noncontiguous), row_shard, rtol=0, atol=0
        )
        # Attention V-up may change the output width; inverse transport must
        # depend on the actual output rather than the query's feature width.
        restored_values = op.communicator.restore_queries(
            row_shard[..., :8].contiguous()
        )
        torch.testing.assert_close(restored_values, head_shard[..., :8], rtol=0, atol=0)
        empty = op._prefill(head_shard[:0])
        assert empty.shape == (0, 8, 16)
        assert op.communicator.restore_queries(empty).shape == (0, 4, 16)
        query_cp = PageRRSparseMlaOp(
            num_heads=8,
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            page_size=128,
            softmax_extra_scale=1.0,
            top_k=8,
            parallelism_config=parallel,
            workspace_owner=SimpleNamespace(
                prefill_query_cp=True,
                sfa_metadata_free=False,
                sfa_use_fp8=False,
                mega_a2a=False,
            ),
        )
        query_cp.plan(SimpleNamespace(is_prefill=False), None)
        partial = torch.full(
            (2, 8, 8), 2 + rank * 4, dtype=torch.bfloat16, device="cuda"
        )
        lse = torch.full((2, 8), 0.0 if rank == 0 else 1.5849625, device="cuda")
        lengths = torch.tensor(
            [3, 3 if rank == 0 else 0], dtype=torch.int32, device="cuda"
        )
        merged = query_cp.communicator.combine(partial, lse, lengths, output_dim=8)
        expected = torch.tensor([5.0, 2.0], dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            merged, expected[None, :, None].expand(8, 2, 8), rtol=0, atol=0.03125
        )
        # Fix values at the sparse-attention boundary; run the real output gate
        # and replicated projection path, including its final request-row layout.
        attention = object.__new__(KimiK31DSA)
        torch.nn.Module.__init__(attention)
        attention.prologue = "test"
        attention.prefill_query_cp = True
        attention.tp_size, attention.tp_rank = 2, rank
        attention._fp8_output_gate = False
        attention.output = torch.nn.Identity()
        values = merged.transpose(0, 1).reshape(2, 64).contiguous()
        attention._decode_prologue = lambda *_: (values, torch.zeros_like(values))
        actual = attention(
            values,
            {"full": SimpleNamespace(workspace_owner=None)},
            None,
            attention_inputs=SimpleNamespace(valid_token_mask=None),
            metadata=SimpleNamespace(query_layout=None, outer_layout=None),
        )
        torch.testing.assert_close(
            actual, (expected[rank] / 2).expand(1, 64), rtol=0, atol=0.03125
        )
        torch.cuda.synchronize()
    finally:
        destroy_distributed_environment()


class KimiK31PrefillLayoutTest(unittest.TestCase):
    def test_sparse_prefill_head_to_query_and_inverse_transport(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "requires two CI GPUs")
        ports, locks = PortManager().get_consecutive_ports(1)
        try:
            mp.spawn(_prefill_layout_worker, args=(ports[0],), nprocs=2, join=True)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
