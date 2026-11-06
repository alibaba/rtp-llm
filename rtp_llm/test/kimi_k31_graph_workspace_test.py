"""Production planner/capacity checks; meta tensors, no model/GPU evidence."""
import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import DsaWorkspaceOwner
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import DsaKVExecution
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import MlaPageRRCacheGather


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self, stream):
        pass


def run():
    results = []
    for fp8 in (False, True):
        for shards in (8, 4):
            owner = DsaWorkspaceOwner(device="meta", indexer_use_fp8=fp8,
                heads=96, hidden_dim=7168, q_rank=1536, tp_size=shards)
            owner.graph_max_requests = 64
            owner.graph_max_queries = 4
            owner.graph_cache_groups = 2
            owner.indexer_score = NS(
                supports_direct_pages=lambda page_size, device: False,
                prepare_paged_metadata=lambda *a, **kw:
                    torch.empty((148, 4), dtype=torch.int32, device="meta"),
            )
            records = []
            pointers = {}
            for queries, batches in ((1, (64, 32, 16, 8)),
                                     (4, range(64, 1, -2)),
                                     (1, (64, 8, 64))):
                for batch in batches:
                    p = NS(workspace_owner=owner, is_prefill=False, is_cuda_graph=True,
                        positions_d=torch.empty(batch * queries, device="meta", dtype=torch.int32),
                        decode=NS(local_causal_lens=torch.empty(batch, queries, device="meta", dtype=torch.int32)),
                        block_table=torch.empty(batch, (2097152 // (1024 * shards)) * 8,
                                                device="meta", dtype=torch.int32),
                        page_size=1024, kernel_page_size=128,
                        shard_size=shards, shard_rank=0, forbid_realloc=False)
                    for name in ("sfa", "indexer"):
                        try:
                            e = DsaKVExecution(p, name)
                        except Exception:
                            print("FAIL", fp8, shards, batch, queries, name, {str(k): (v.resident_bytes, v.metadata_budget_bytes) for k,v in owner.phases.items()}, flush=True)
                            raise
                        assert e.required_bytes <= owner.budgets[name]
                        key = name
                        signature = {n: (id(t), t.numel()) for n, t in e.workspace.buffers.items()}
                        if key in pointers:
                            assert signature == pointers[key], (key, batch)
                        else:
                            pointers[key] = signature
                        records.append(dict(batch=batch, queries=queries, phase=name,
                                            blocks=len(e.blocks), required=e.required_bytes))
            assert len(owner.phases) == 2
            results.append(dict(fp8=fp8, shards=shards, records=records,
                phases={str(k):dict(scratch=v.resident_bytes, metadata=v.metadata_budget_bytes)
                        for k, v in owner.phases.items()}))



class DsaGraphWorkspaceBudgetTest(unittest.TestCase):
    def test_shared_prefill_budget_selects_blocks_for_each_phase(self):
        owner = DsaWorkspaceOwner(
            device="meta", indexer_use_fp8=True, heads=96, tp_size=8,
        )
        positions = torch.empty(8192, device="meta", dtype=torch.int32)
        params = NS(
            workspace_owner=owner, is_prefill=True, is_cuda_graph=False,
            positions_d=positions, query_positions=positions,
            block_table=torch.empty(1, 1024, device="meta", dtype=torch.int32),
            kv_lens_host=(1048576,), page_size=1024, kernel_page_size=128,
            shard_size=8, shard_rank=0,
            cache_adapter=MlaPageRRCacheGather(1024, 8, 0, 128),
        )
        execution_module = sys.modules[DsaKVExecution.__module__]
        # Exercise real planning and Page-RR descriptors on meta tensors.
        # Only the GPU metadata/event creation is replaced.
        with patch.dict(os.environ, {"RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB": "1"}), \
             patch.dict(sys.modules, {
                 "rtp_llm.models_py.triton_kernels.mla.page_rr": NS(_metadata=lambda *_: None)
             }), \
             patch.object(execution_module, "_prefill_block_layout", return_value=(None,) * 4), \
             patch.object(torch.cuda, "current_stream", return_value=NS(cuda_stream=1)), \
             patch.object(torch.cuda, "Event", Event):
            indexer = DsaKVExecution(params, "indexer")
            sfa = DsaKVExecution(params, "sfa")
        self.assertEqual(indexer.plan.page_capacity, 31)
        self.assertEqual(sfa.plan.page_capacity, 424)
        self.assertEqual(sum(block.width for block in indexer.blocks), 1048576)
        self.assertEqual(sum(block.width for block in sfa.blocks), 1048576)

    def test_all_buckets_share_bounded_scratch(self):
        with patch.object(torch.cuda, 'get_device_properties', return_value=NS(multi_processor_count=148)), \
             patch.object(torch.cuda, 'current_stream', return_value=NS(cuda_stream=1)), \
             patch.object(torch.cuda, 'Event', Event):
            run()


if __name__ == "__main__":
    unittest.main()
