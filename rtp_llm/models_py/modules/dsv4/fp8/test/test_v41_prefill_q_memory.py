"""V4.1 streamed Q projection, RoPE numerical boundaries and scratch lifetime."""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4 import chunk_env, prefill_workspace
from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillQKV
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


class PrefillQMemoryTest(unittest.TestCase):
    def test_workspace_capacity_tracks_chunk_switch(self):
        with patch.object(chunk_env, "FLASH_MLA_SPARSE_Q_CHUNK", 4):
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"DSV41_PREFILL_Q_CHUNKED": enabled}):
                    for rows in (0, 1, 4, 5, 131072):
                        self.assertEqual(
                            prefill_workspace.prefill_q_workspace_rows(rows),
                            min(rows, 4) if enabled == "1" else rows,
                        )

    def _case(self, rows, *, chunked):
        torch.manual_seed(7)
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.dim, owner.n_heads, owner.head_dim, owner.rope_head_dim = 5, 2, 8, 4
        owner.softmax_scale, owner.attn_sink = 0.5, torch.zeros(2)
        qr = torch.randn(rows, 4).bfloat16()
        wq = torch.randn(16, 4).bfloat16()
        wo = torch.randn(5, 16).bfloat16()
        events, pointers = [], []

        def project(x, *, out):
            events.append(("q", len(x)))
            pointers.append(out.data_ptr())
            return out.copy_(F.linear(x, wq))

        owner.wq_b = project
        owner._prefill_output_all_reduce = Mock()

        def output_project(o, freqs, *, out):
            events.append(("output", len(o)))
            attention.apply_rotary_emb(
                o[..., -owner.rope_head_dim :].unsqueeze(0), freqs, inverse=True
            )
            out.copy_(F.linear(o.flatten(1), wo))

        owner._prefill_output_proj_into = output_project
        angles = torch.randn(rows * 2, owner.rope_head_dim // 2)
        freqs = torch.polar(torch.ones_like(angles), angles)[::2]
        indices = (torch.arange(rows) % 7).view(rows, 1, 1).int()
        lengths = (torch.arange(rows) % 3 != 1).int()
        kv = torch.randn(7, 1, 8).bfloat16()

        def mla(*, q, kv, indices, topk_length, **kwargs):
            events.append(("mla", len(q)))
            gathered = kv[indices[:, 0, 0].long()]
            return (
                (
                    q.float() + gathered.float() * topk_length[:, None, None].float()
                ).bfloat16(),
                None,
                None,
            )

        reference_q = F.linear(qr, wq).view(rows, 2, 8)
        attention.rope_only(reference_q, freqs, 4)
        reference_o = mla(q=reference_q, kv=kv, indices=indices, topk_length=lengths)[0]
        reference = torch.empty(rows, owner.dim, dtype=torch.bfloat16)
        output_project(reference_o, freqs, out=reference)
        events.clear()

        common = SimpleNamespace(
            freqs_cis=freqs,
            workspace=PrefillWorkspace(
                torch.device("cpu"),
                q_rows=min(rows, 4) if chunked else rows,
                q_dim=16,
                reserve_cp=False,
                align_bytes=1,
            ),
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"DSV41_PREFILL_Q_CHUNKED": str(int(chunked))})
            )
            stack.enter_context(patch.object(attention, "_FLASH_MLA_SPARSE_Q_CHUNK", 4))
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {"flash_mla": SimpleNamespace(flash_mla_sparse_fwd=mla)},
                )
            )
            result = owner._prefill_sparse_attention(
                PrefillQKV(qr=qr, q=None, kv_full=kv[:, 0]),
                common,
                kv=kv,
                indices=indices,
                topk_length=lengths,
                profile_name="test.v41.q",
            )
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        owner._prefill_output_all_reduce.assert_called_once_with(result)
        if chunked:
            expected = []
            for start in range(0, rows, 4):
                count = min(4, rows - start)
                expected.extend((name, count) for name in ("q", "mla", "output"))
            self.assertEqual(events, expected)
            self.assertLessEqual(len(set(pointers)), 1)
        elif rows:
            self.assertEqual(events[0], ("q", rows))

    def test_chunked_matches_full_projection_including_padding_and_tail(self):
        for rows in (0, 1, 4, 5, 11):
            for chunked in (False, True):
                with self.subTest(rows=rows, chunked=chunked):
                    self._case(rows, chunked=chunked)

    def test_cpu_rope_falls_back_and_empty_does_not_launch(self):
        x = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.bfloat16)
        freqs = torch.tensor([[1j]], dtype=torch.complex64)
        with patch.object(attention, "rope_only_inplace") as kernel:
            result = attention._prefill_q_rope(x, freqs, 2)
            self.assertIs(result, x)
            self.assertEqual(result.tolist(), [[[1, 2, -4, 3]]])
            empty = x[:0]
            self.assertIs(attention._prefill_q_rope(empty, freqs[:0], 2), empty)
        kernel.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillQRopeCudaTest(unittest.TestCase):
    def test_inplace_rope_matches_bf16_eager_with_strided_frequencies(self):
        torch.manual_seed(9)
        for rows, heads in ((0, 128), (1, 128), (17, 7), (257, 128)):
            for strided in (False, True):
                with self.subTest(rows=rows, heads=heads, strided=strided):
                    base = torch.randn(
                        rows, heads, 512, device="cuda", dtype=torch.bfloat16
                    )
                    angles = torch.randn(rows * 2, 64, device="cuda")
                    freqs = torch.polar(torch.ones_like(angles), angles)[::2, ::2]
                    if not strided:
                        freqs = freqs.contiguous()
                    expected = base.clone()
                    with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "0"}):
                        attention._prefill_q_rope(expected, freqs, 64)
                    actual = base.clone()
                    pointer = actual.data_ptr()
                    with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "1"}):
                        result = attention._prefill_q_rope(actual, freqs, 64)
                    self.assertIs(result, actual)
                    self.assertEqual(actual.data_ptr(), pointer)
                    torch.testing.assert_close(
                        actual[..., :-64], base[..., :-64], rtol=0, atol=0
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_mxfp8_q_projection_is_identical_across_chunk_boundaries(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP8 Q projection requires SM100")
        from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

        torch.manual_seed(23)
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.n_heads, owner.head_dim, owner.rope_head_dim = 64, 512, 64
        q_dim, q_rank = owner.n_heads * owner.head_dim, 1280
        weight = (torch.randn(q_dim, q_rank, device="cuda") * 0.02).to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(q_dim // 32, q_rank // 32, device="cuda").to(
            torch.float8_e8m0fnu
        )
        owner.wq_b = V41MXFP8Linear(weight, scales)
        # Include an unaligned final tile and padded zero rows from two CP
        # request halves. The latter still have their own RoPE positions.
        for rows, chunk_rows in ((257, 128), (1025, 384)):
            with self.subTest(rows=rows, chunk_rows=chunk_rows):
                qr = torch.randn(rows, q_rank, device="cuda", dtype=torch.bfloat16)
                qr[121:128].zero_()
                qr[-3:].zero_()
                angles = torch.randn(rows * 2, 32, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)[::2]
                expected = owner._lin(owner.wq_b, qr).view(rows, 64, 512)
                attention.rope_only(expected, freqs, 64)
                workspace = PrefillWorkspace(
                    torch.device("cuda"),
                    q_rows=chunk_rows,
                    q_dim=q_dim,
                    reserve_cp=False,
                    align_bytes=1,
                )
                with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "1"}):
                    for start in range(0, rows, chunk_rows):
                        end = min(start + chunk_rows, rows)
                        actual = owner._project_prefill_q(
                            qr[start:end], freqs[start:end], workspace
                        )
                        torch.testing.assert_close(
                            actual, expected[start:end], rtol=0, atol=0
                        )

    def test_rope_does_not_allocate_a_q_sized_temporary(self):
        q = torch.randn(1025, 128, 512, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(1025, 32, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "1"}):
            attention._prefill_q_rope(q, freqs, 64)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            attention._prefill_q_rope(q, freqs, 64)
            torch.cuda.synchronize()
            self.assertLess(torch.cuda.max_memory_allocated() - allocated, 1 << 20)


if __name__ == "__main__":
    unittest.main()
