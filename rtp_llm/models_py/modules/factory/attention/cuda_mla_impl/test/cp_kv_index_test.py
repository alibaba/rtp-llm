"""Numerical/layout coverage for narrow V scatter and its conservative gates."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test import (
    glm53_prefill_q_selection_test as q_tests,
)
from rtp_llm.utils.model_weight import W


class IndexControlTest(unittest.TestCase):
    def test_layout_gates_and_projection_order(self):
        fixture = q_tests.QPreselectionControlTest()
        for name, rows, removed, identity, workspace, graph, enabled in (
            ("full", 257, False, False, True, False, True),
            ("padding", 257, True, False, True, False, False),
            ("small", 17, False, False, True, False, False),
            ("identity", 257, False, True, True, False, False),
            ("no_workspace", 257, False, False, False, False, False),
            ("graph", 257, False, False, True, True, False),
        ):
            with self.subTest(name=name):
                ids = torch.arange(rows)
                if not identity:
                    ids = torch.cat((ids[1::2], ids[::2]))
                if removed:
                    ids = ids[:-1]
                impl = fixture._wrapper(
                    ids, rows, identity=identity, workspace=workspace, graph=graph
                )
                impl.weights = [{W.mla_vc: torch.ones(2, 5, 2)}]
                impl.fmha_impl.scatter_reverse_indices = None
                selected = torch.arange(len(ids) * 10).reshape(-1, 2, 5).float()
                restored = torch.zeros(rows, 2, 5).index_copy_(0, ids, selected)
                expected = restored.sum(-1, keepdim=True).expand(-1, -1, 2)
                calls = []
                impl._apply_input_bmm = lambda q, *a, out=None: out

                def attention(*args, **kwargs):
                    self.assertEqual(kwargs["defer_output_scatter"], enabled)
                    return selected if enabled else restored

                def project(value, layer):
                    calls.append("project")
                    return value.sum(-1, keepdim=True).expand(-1, -1, 2).clone()

                def scatter(value, indices, total, *, out, reverse_indices):
                    calls.append("scatter")
                    self.assertEqual(value.shape[-1], 2)
                    self.assertEqual(out.shape, (rows, 2, 2))
                    return out.zero_().index_copy_(0, indices, value)

                impl.fmha_impl.forward = attention
                impl._apply_output_bmm = project
                with patch.dict(
                    os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": "1"}
                ), patch.object(cp, "triton_kv_scatter", side_effect=scatter):
                    actual = impl.forward(
                        torch.ones(rows, 2, 4),
                        torch.empty(rows, 5),
                        torch.empty(rows, 1),
                        object(),
                        0,
                        torch.ones(len(ids), 1),
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(
                    calls, ["project", "scatter"] if enabled else ["project"]
                )
                if workspace:
                    self.assertIsNone(impl._glm53_prefill_workspace._storage)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class IndexCudaTest(unittest.TestCase):
    def test_real_flashmla_projection_and_narrow_scatter_are_bit_exact(self):
        fixture = q_tests.QPreselectionCudaTest()
        fixture.setUp()
        for chunks, removed, strided in (
            ([129, 128], (), False),
            ([509, 516], (), False),
            ([2003, 2090], (), False),
            ([129, 128], (), True),
            ([129, 128], (100,), False),
            ([17], (), False),
        ):
            with self.subTest(chunks=chunks, removed=removed, strided=strided):
                data = fixture._fixture(chunks, removed, strided)
                old = fixture._impl(data, workspace=True)
                new = fixture._impl(data, workspace=True)
                if not new.fmha_impl.total_local_ids_is_identity:
                    new.fmha_impl.scatter_reverse_indices = (
                        cp.prepare_kv_scatter_indices(data.ids, data.rows)
                    )
                with patch.dict(os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": "0"}):
                    expected = fixture._run(old, data)
                original_scatter = cp.triton_kv_scatter
                widths = []

                def scatter(value, *args, **kwargs):
                    widths.append(value.size(-1))
                    return original_scatter(value, *args, **kwargs)

                with patch.dict(
                    os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": "1"}
                ), patch.object(cp, "triton_kv_scatter", side_effect=scatter):
                    actual = fixture._run(new, data)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if not new.fmha_impl.total_local_ids_is_identity:
                    self.assertEqual(widths, [512 if removed else 256])
                self.assertIsNone(new._glm53_prefill_workspace._storage)
                if removed:
                    self.assertEqual(
                        torch.count_nonzero(actual[list(removed)]).item(), 0
                    )


if __name__ == "__main__":
    unittest.main()
