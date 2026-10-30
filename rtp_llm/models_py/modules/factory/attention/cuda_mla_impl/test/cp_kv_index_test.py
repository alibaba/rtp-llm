"""Numerical/layout coverage for narrow V scatter and its conservative gates."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test import (
    glm53_prefill_q_selection_test as q_tests,
)
from rtp_llm.utils.model_weight import W


class HostPlanTest(unittest.TestCase):
    def test_host_prefix_is_owned_per_forward_and_input_is_unchanged(self):
        original = torch.tensor([999, 999], dtype=torch.int32)
        inp = SimpleNamespace(
            is_prefill=True,
            input_lengths=torch.tensor([2, 5], dtype=torch.int32),
            prefix_lengths=original,
        )
        for values in ([0, 12, 37], [0, 4, 10]):
            ptr = torch.tensor(values, dtype=torch.int32)
            params = SimpleNamespace(prefill_ragged_kv_len_indptr_h=ptr)
            expected = ptr[1:] - ptr[:-1] - inp.input_lengths
            planned = cp._cp_inputs_with_host_prefix(params, inp, use_host=True)
            self.assertIsNot(planned, inp)
            self.assertIs(inp.prefix_lengths, original)
            torch.testing.assert_close(planned.prefix_lengths, expected)
            ptr.fill_(123)
            torch.testing.assert_close(planned.prefix_lengths, expected)
            self.assertIs(
                cp._cp_inputs_with_host_prefix(params, inp, use_host=False), inp
            )
        bad = SimpleNamespace(
            prefill_ragged_kv_len_indptr_h=torch.zeros(1, dtype=torch.int32)
        )
        self.assertIs(cp._cp_inputs_with_host_prefix(bad, inp, use_host=True), inp)
        inp.is_prefill = False
        self.assertIs(cp._cp_inputs_with_host_prefix(params, inp, use_host=True), inp)

    def test_same_step_host_slots_and_fallback(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for owner in (64, 512):
                for rank in range(4):
                    for columns in (0, 2, 19):
                        for rows in (0, 1, 57):
                            with self.subTest(
                                device=device,
                                owner=owner,
                                rank=rank,
                                columns=columns,
                                rows=rows,
                            ):
                                pos = torch.arange(rows, dtype=torch.int32) * 33
                                batch = torch.arange(rows, dtype=torch.int32) % 3
                                table = (
                                    torch.arange(
                                        3 * columns, dtype=torch.int32
                                    ).reshape(3, columns)
                                    % 23
                                )
                                if columns:
                                    table[0, 0] = -1
                                params = SimpleNamespace(
                                    positions_d=pos.to(device),
                                    batch_indice_d=batch.to(device),
                                    positions_h=pos,
                                    batch_indice_h=batch,
                                )
                                inp = SimpleNamespace(
                                    kv_cache_kernel_block_id_host=table
                                )
                                args = (
                                    params,
                                    table.to(device),
                                    inp,
                                    64,
                                    4,
                                    rank,
                                    owner,
                                )
                                expected = cp._cp_sharded_slot_mapping(
                                    params.positions_d,
                                    args[1],
                                    params.batch_indice_d,
                                    64,
                                    4,
                                    rank,
                                    owner_tokens_per_block=owner,
                                )
                                actual, host = cp._plan_cp_sharded_slots(
                                    *args, use_host=True
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                                # Empty tables may use the generic fallback without a mirror.
                                if host is not None:
                                    torch.testing.assert_close(
                                        host, expected.cpu(), rtol=0, atol=0
                                    )
                                # Disabled host planning (including graphs) never returns a snapshot.
                                actual, host = cp._plan_cp_sharded_slots(
                                    *args, use_host=False
                                )
                                self.assertIsNone(host)
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                                # A mismatched host snapshot cannot be consumed.
                                params.positions_h = torch.empty(
                                    rows + 1, dtype=torch.int32
                                )
                                actual, host = cp._plan_cp_sharded_slots(
                                    *args, use_host=True
                                )
                                self.assertIsNone(host)
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    def test_host_gather_total_is_refreshed_and_graph_falls_back(self):
        op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
        op.block_table = torch.zeros(2, 1, dtype=torch.int32)
        op.use_cuda_graph = False
        with patch.dict(os.environ, {"RTP_LLM_CP_HOST_PLAN": "1"}):
            for lengths in ([0, 0, 0], [0, 17, 97], [0, 33, 33]):
                ptr = torch.tensor(lengths, dtype=torch.int32)
                op.mla_params = SimpleNamespace(
                    prefill_ragged_kv_len_indptr_h=ptr,
                    prefill_ragged_kv_len_indptr_d=ptr.clone(),
                )
                value = op._build_gather_workspace()
                if lengths[-1] == 0:
                    self.assertIsNone(value)
                else:
                    self.assertEqual(value.total_kv_len, lengths[-1])
                    torch.testing.assert_close(value.seq_lens, ptr[1:] - ptr[:-1])
            op.use_cuda_graph = True
            op.mla_params.prefill_ragged_kv_len_indptr_h = torch.tensor([0, 999, 999])
            self.assertEqual(op._build_gather_workspace().total_kv_len, 33)


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
            ([1557, 1557], (100, 500, 1558, 3000), False),
            ([2048, 2049], (0, 2048, 4096), True),
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
                    os.environ,
                    {
                        "RTP_LLM_CP_KV_INDEX_OPT": "1",
                        "RTP_LLM_CP_DEFER_PADDED_SCATTER": "1",
                    },
                ), patch.object(cp, "triton_kv_scatter", side_effect=scatter):
                    actual = fixture._run(new, data)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if not new.fmha_impl.total_local_ids_is_identity:
                    self.assertEqual(
                        widths, [512 if removed and data.ids.numel() < 1024 else 256]
                    )
                self.assertIsNone(new._glm53_prefill_workspace._storage)
                if removed:
                    self.assertEqual(
                        torch.count_nonzero(actual[list(removed)]).item(), 0
                    )


if __name__ == "__main__":
    unittest.main()
