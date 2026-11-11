"""Attention metadata must outlive its consumer, not its Python GC generation."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_ingress import (
    DsaKVIngress,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
    SparseMlaPageRRImpl,
)


class AttentionLifetimeTest(unittest.TestCase):
    def test_finished_prefill_releases_metadata_without_cyclic_gc(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(16):
                op = PageRRSparseMlaOp.__new__(PageRRSparseMlaOp)
                op.workspace_owner = SimpleNamespace(sfa_use_fp8=True)
                metadata = torch.empty(32, dtype=torch.int32)
                reference = weakref.ref(metadata)
                params = DsaPageRRParams.__new__(DsaPageRRParams)
                params.is_prefill = True
                params.producer_layout = SimpleNamespace(
                    local_rows=2,
                    physical_rows=2,
                    rankmajor_positions=metadata,
                    rankmajor_requests=metadata,
                )
                params.workspace_owner = SimpleNamespace(
                    device=torch.device("cpu"),
                    phase=Mock(),
                )
                params.shard_size = 1
                params.kernel_page_size = 2
                params.cache_adapter = SimpleNamespace(
                    slot_mapping=lambda positions, *args, **kwargs: positions
                )
                params.block_table = None
                params.is_warmup = False
                params.metadata = metadata
                with patch("torch.cuda.Event"), patch("torch.cuda.current_stream"):
                    params.ingress_plans = {
                        phase: DsaKVIngress(params, phase, 2)
                        for phase in ("indexer", "sfa")
                    }
                params_reference = weakref.ref(params)
                op.plan(params, None)
                del metadata, op, params
                self.assertIsNone(params_reference())
                self.assertIsNone(reference())
        finally:
            if enabled:
                gc.enable()

    def test_only_requested_phase_is_prepared_before_operator_plan(self):
        for phase in (None, "indexer", "sfa"):
            for prefill, graph in ((True, False), (False, False), (False, True)):
                with self.subTest(phase=phase, prefill=prefill, graph=graph):
                    events = []
                    params = SimpleNamespace(
                        is_prefill=prefill,
                        is_cuda_graph=graph,
                        block_table=object(),
                        prepare=Mock(),
                        prepare_execution=events.append,
                    )
                    impl = SparseMlaPageRRImpl.__new__(SparseMlaPageRRImpl)
                    impl.execution_phase = phase
                    impl.page_rr_params = params
                    impl.fmha_impl = SimpleNamespace(
                        plan=lambda *args: events.append("operator_plan")
                    )
                    inputs = object()
                    impl.prepare(inputs, forbid_realloc=graph)
                    expected = (
                        [phase]
                        if phase is not None and prefill
                        else ["sfa", "indexer"] if graph else ["indexer", "sfa"]
                    )
                    self.assertEqual(events, [*expected, "operator_plan"])
                    params.prepare.assert_called_once_with(inputs, forbid_realloc=graph)


if __name__ == "__main__":
    unittest.main()
