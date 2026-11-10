"""Padded projection gate views are consumed only by main eager prefix FMHA."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_paged_fmha as paged
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl


def _owner(*, prefill=True, verify=False, draft=False, graph=False, prefix=True):
    owner = SparseGqaFmhaImpl.__new__(SparseGqaFmhaImpl)
    owner.is_prefill = prefill
    owner.is_target_verify = verify
    owner._is_mtp_draft = draft
    owner.attn_inputs = SimpleNamespace(is_cuda_graph=graph)
    owner.head_num, owner.head_dim, owner.kv_head_num = 12, 256, 1
    owner.q_width = 3072
    owner._is_prefix_reuse_prefill = lambda: prefill and not verify and prefix
    owner._is_draft_graph = lambda: draft and graph
    owner._paged_kv_scale_args = lambda _: {}
    return owner


def _gate(rows, device="cpu"):
    # A nonzero offset and poison in the padding detect packed-row addressing.
    physical = torch.full(
        (rows + 2, 3712), float("nan"), dtype=torch.bfloat16, device=device
    )
    gate = physical[1 : rows + 1, :3072]
    gate.copy_(torch.randn(rows, 3072, dtype=torch.bfloat16, device=device))
    return physical, gate


class QsaStridedGateRoutingTest(unittest.TestCase):
    def test_low_level_layout_requires_opt_in_and_disjoint_packed_rows(self):
        _, packed = _gate(6)
        gate = packed.view(2, 3, 12, 256)
        self.assertFalse(gate.is_contiguous())
        for allow, graph, expected in (
            (False, False, False),
            (False, True, False),
            (True, True, False),
            (True, False, True),
        ):
            with self.subTest(allow=allow, graph=graph):
                self.assertEqual(
                    paged._output_gate_layout_supported(
                        gate, allow_strided=allow, graph_capture=graph
                    ),
                    expected,
                )
        bad = (
            gate[:, :, :, ::2],
            gate[:, :, :1].expand(2, 3, 12, 256),
            gate[:, :1].expand(2, 3, 12, 256),
            gate[:1].expand(2, 3, 12, 256),
        )
        for tensor in bad:
            self.assertFalse(
                paged._output_gate_layout_supported(
                    tensor, allow_strided=True, graph_capture=False
                )
            )
        self.assertTrue(
            paged._output_gate_layout_supported(
                gate.contiguous(), allow_strided=False, graph_capture=True
            )
        )

    def test_only_main_eager_prefix_forwards_the_original_strided_gate(self):
        _, gate = _gate(3)
        qkv = torch.zeros(3, 3584, dtype=torch.bfloat16)
        modes = (
            ({}, True),
            ({"prefill": False}, False),
            ({"verify": True}, False),
            ({"draft": True}, False),
            ({"draft": True, "graph": True}, False),
            ({"graph": True}, False),
            ({"prefix": False}, False),
        )
        for mode, fused in modes:
            with self.subTest(mode=mode):
                owner = _owner(**mode)
                owner.forward = Mock(return_value=torch.ones_like(gate))
                result = owner.forward_with_output_gate(qkv, None, 3, gate)
                if fused:
                    self.assertIs(owner.forward.call_args.kwargs["output_gate"], gate)
                    self.assertIs(result, owner.forward.return_value)
                else:
                    self.assertNotIn("output_gate", owner.forward.call_args.kwargs)
                    torch.testing.assert_close(
                        result, torch.sigmoid(gate), rtol=0, atol=0
                    )

    def test_decode_verify_and_draft_reject_direct_strided_gate(self):
        _, gate = _gate(3)
        qkv = torch.zeros(3, 3584, dtype=torch.bfloat16)
        for mode in (
            {"prefill": False},
            {"verify": True},
            {"draft": True},
            {"draft": True, "graph": True},
            {"graph": True},
            {"prefix": False},
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "fused QSA output gate"):
                    _owner(**mode).forward(qkv, output_gate=gate)

    def test_existing_contiguous_decode_verify_and_draft_graph_route_is_kept(self):
        _, strided = _gate(3)
        gate = strided.contiguous()
        qkv = torch.zeros(3, 3584, dtype=torch.bfloat16)
        for mode in (
            {"prefill": False},
            {"verify": True},
            {"draft": True, "graph": True},
        ):
            with self.subTest(mode=mode):
                owner = _owner(**mode)
                owner.forward = Mock(return_value=torch.ones_like(gate))
                owner.forward_with_output_gate(qkv, None, 3, gate)
                self.assertIs(owner.forward.call_args.kwargs["output_gate"], gate)

    def test_ragged_request_views_keep_offsets_strides_and_storage(self):
        _, gate = _gate(5)
        q = torch.zeros(5, 3072, dtype=torch.bfloat16)
        requests = [
            SimpleNamespace(
                start=start,
                end=end,
                selected_indices=torch.zeros(1, end - start, 32, dtype=torch.int32),
                kv_lens=torch.full((1, end - start), 128, dtype=torch.int32),
            )
            for start, end in ((0, 2), (2, 5))
        ]
        plan = SimpleNamespace(
            tokens=5,
            block_table=torch.ones(2, 1, dtype=torch.int32),
            page_size=128,
            requests=requests,
        )
        cache = SimpleNamespace(kv_cache_base=torch.empty(2, 2, 1, 128, 256))
        for grouped in (False, True):
            calls = []

            def attention(query, *args, **kwargs):
                calls.append(kwargs)
                b, h, s, d = query.shape
                shape = (b, s, h, d) if kwargs["bthd_output"] else (b, h, s, d)
                return torch.zeros(shape, dtype=query.dtype)

            with (
                patch.object(paged, "_grouped_split_supported", return_value=grouped),
                patch.object(paged, "sparse_paged_gqa_attn", side_effect=attention),
            ):
                actual = _owner()._forward_ragged_paged(
                    q, cache, plan, output_gate=gate
                )
            self.assertEqual(actual.shape, (5, 3072))
            self.assertEqual(len(calls), 2)
            for request, call in zip(requests, calls):
                view = call["output_gate"]
                self.assertTrue(call["allow_strided_output_gate"])
                self.assertEqual(view.data_ptr(), gate[request.start].data_ptr())
                self.assertEqual(view.stride()[1:], (3712, 256, 1))
                self.assertEqual(
                    view.untyped_storage().data_ptr(), gate.untyped_storage().data_ptr()
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaStridedGateKernelTest(unittest.TestCase):
    def test_grouped_single_split_and_merge_match_packed_gate_without_copy(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 grouped path is required")
        torch.manual_seed(3017)
        queries = 5
        q = torch.randn(1, 12, queries, 256, device="cuda", dtype=torch.bfloat16)
        cache = torch.randn(3, 2, 1, 128, 256, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.full((1, queries), 256, device="cuda", dtype=torch.int32)
        selected = torch.arange(256, device="cuda", dtype=torch.int32).view(1, 1, 256)
        selected = selected.expand(1, queries, 256).contiguous()
        storage, flat_gate = _gate(queries, "cuda")
        flat_gate[0, :3] = torch.tensor(
            [float("-inf"), float("inf"), float("nan")],
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = flat_gate.view(1, queries, 12, 256)
        packed_gate = gate.contiguous()
        original_contiguous = torch.Tensor.contiguous

        def forbid_gate_copy(tensor, *args, **kwargs):
            if (
                tensor.untyped_storage().data_ptr()
                == storage.untyped_storage().data_ptr()
            ):
                raise AssertionError("strided output gate was materialized")
            return original_contiguous(tensor, *args, **kwargs)

        def run(g, *, grouped, allow):
            return paged.sparse_paged_gqa_attn(
                q,
                cache,
                table,
                lengths,
                selected,
                page_size=128,
                kv_head_num=1,
                validate_selected=False,
                output_gate=g,
                bthd_output=grouped,
                allow_strided_output_gate=allow,
            )

        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            # Both grouped epilogues plus the shared generic merge are covered.
            for grouped, splits in ((True, 1), (True, 4), (False, 2)):
                with self.subTest(grouped=grouped, splits=splits):
                    with (
                        patch.object(
                            paged, "_grouped_split_supported", return_value=grouped
                        ),
                        patch.object(
                            paged,
                            "_grouped_split_config",
                            return_value=(32, 4, 2, splits),
                        ),
                    ):
                        expected = run(packed_gate, grouped=grouped, allow=False)
                        with patch.object(torch.Tensor, "contiguous", forbid_gate_copy):
                            actual = run(gate, grouped=grouped, allow=True)
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0, equal_nan=True
                        )
                        self.assertEqual(gate.stride(1), 3712)

            for allow, graph in ((False, False), (True, True)):
                with self.subTest(allow=allow, graph=graph):
                    with self.assertRaisesRegex(ValueError, "QSA output gate must"):
                        paged.sparse_paged_gqa_attn(
                            q,
                            cache,
                            table,
                            lengths,
                            selected,
                            page_size=128,
                            output_gate=gate,
                            allow_strided_output_gate=allow,
                            graph_capture=graph,
                        )


if __name__ == "__main__":
    unittest.main()
