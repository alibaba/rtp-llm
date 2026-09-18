"""V4.1 decode Q RoPE and compressed global top-k slot contracts.

CUDA tests cover the target verify S=6 layout and replay with changed data.
These slot tests apply to global CSA/HCA pools where physical entries equal
logical compressed entries, not the padded INDEXER_KV pool.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.decode import compute_qkv
from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
    translate_local_to_global_slots,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb


def _freqs(rows, device):
    angles = torch.rand(rows, 32, device=device) * 6.28
    return torch.polar(torch.ones_like(angles), angles)


class V41DecodePreprocessingCPU(unittest.TestCase):
    def test_cpu_fallback_preserves_eager_result(self):
        q = torch.randn(2, 6, 64, 512, dtype=torch.bfloat16)
        freqs = _freqs(12, "cpu")
        expected = q.clone()
        apply_rotary_emb(expected[..., -64:], freqs)
        with patch.object(compute_qkv, "rope_only_inplace") as fused:
            self.assertFalse(compute_qkv._fused_v41_q_rope_supported(q, freqs, 64))
            compute_qkv._apply_v41_q_rope(q, freqs, 64)
            fused.assert_not_called()
        torch.testing.assert_close(q, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41DecodePreprocessingCUDA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.gate = patch.dict(os.environ, {"DSV41_FUSED_DECODE_Q_ROPE": "1"})
        self.gate.start()
        self.addCleanup(self.gate.stop)

    def _assert_rope(self, got, expected, before):
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.stride(), expected.stride())
        torch.testing.assert_close(got[..., :-64], before[..., :-64], rtol=0, atol=0)
        # Triton may contract float multiply/add while the complex reference
        # uses separate operations. Bound differences to one BF16 ULP (plus
        # a small absolute allowance around cancellation near zero).
        torch.testing.assert_close(got, expected, rtol=0.008, atol=1e-6)

    def _attention(self, batch, span):
        q = torch.randn(batch, span, 64 * 512, dtype=torch.bfloat16, device="cuda")
        qr = torch.randn(batch, span, 16, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(batch, span, 512, dtype=torch.bfloat16, device="cuda")
        projections = {"qa": qr, "qb": q, "kv": kv}
        attn = SimpleNamespace(
            rope_head_dim=64,
            head_dim=512,
            n_heads=64,
            skip_post_q_norm=True,
            freqs_cis=_freqs(2048, "cuda"),
            wq_a="qa",
            wq_b="qb",
            wkv="kv",
            q_norm=None,
            kv_norm=torch.ones(512, dtype=torch.bfloat16, device="cuda"),
            eps=1e-6,
            _lin=lambda layer, x: projections[layer].clone(),
            _rmsnorm_weighted=lambda x, weight: x,
        )
        starts = torch.tensor([0, 127, 1024, 1500][:batch], device="cuda")
        positions = (starts[:, None] + torch.arange(span, device="cuda")).flatten()
        return attn, projections, qr, positions

    def test_real_qkv_branch_matches_eager_multi_batch_verify(self):
        for batch in (1, 2, 4):
            for span in (1, 6):
                with self.subTest(batch=batch, span=span):
                    attn, projections, x, positions = self._attention(batch, span)
                    with patch.dict(os.environ, {"DSV41_FUSED_DECODE_Q_ROPE": "0"}):
                        expected = compute_qkv.decode_compute_qkv(attn, x, positions)
                    got = compute_qkv.decode_compute_qkv(attn, x, positions)
                    self._assert_rope(
                        got.q, expected.q, projections["qb"].view_as(got.q)
                    )
                    for name in ("qr", "kv", "freqs_cis"):
                        torch.testing.assert_close(
                            getattr(got, name), getattr(expected, name), rtol=0, atol=0
                        )

    def test_disabled_and_unsupported_dtype_or_layout_use_eager(self):
        freqs = _freqs(12, "cuda")
        cases = [
            (torch.randn(2, 6, 64, 512, device="cuda"), "1"),
            (torch.randn(2, 6, 64, 512, device="cuda", dtype=torch.bfloat16), "0"),
            (
                torch.randn(2, 6, 64, 1024, device="cuda", dtype=torch.bfloat16)[
                    ..., ::2
                ],
                "1",
            ),
        ]
        for q, enabled in cases:
            with self.subTest(dtype=q.dtype, stride=q.stride(), gate=enabled):
                expected = q.clone()
                apply_rotary_emb(expected[..., -64:], freqs)
                with patch.dict(
                    os.environ, {"DSV41_FUSED_DECODE_Q_ROPE": enabled}
                ), patch.object(compute_qkv, "rope_only_inplace") as fused:
                    self.assertFalse(
                        compute_qkv._fused_v41_q_rope_supported(q, freqs, 64)
                    )
                    compute_qkv._apply_v41_q_rope(q, freqs, 64)
                    fused.assert_not_called()
                torch.testing.assert_close(q, expected, rtol=0, atol=0)

    def test_zero_tiny_and_cancellation(self):
        q = torch.randn(1, 6, 64, 512, device="cuda", dtype=torch.bfloat16)
        q[:, 0].zero_()
        q[:, 1].mul_(1e-20)
        q[:, 2].fill_(1.0)
        freqs = _freqs(6, "cuda")
        freqs[2] = complex(0.70710677, 0.70710677)
        expected, before = q.clone(), q.clone()
        apply_rotary_emb(expected[..., -64:], freqs)
        compute_qkv._apply_v41_q_rope(q, freqs, 64)
        self._assert_rope(q, expected, before)

    def test_kernel_errors_propagate(self):
        q = torch.zeros(1, 6, 64, 512, device="cuda", dtype=torch.bfloat16)
        with patch.object(
            compute_qkv, "rope_only_inplace", side_effect=RuntimeError("kernel failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                compute_qkv._apply_v41_q_rope(q, _freqs(6, "cuda"), 64)

    def test_qkv_graph_replay_updates_inputs_and_positions(self):
        for batch, span in ((1, 1), (4, 6)):
            with self.subTest(batch=batch, span=span):
                attn, projections, x, positions = self._attention(batch, span)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        compute_qkv.decode_compute_qkv(attn, x, positions)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    got = compute_qkv.decode_compute_qkv(attn, x, positions)
                for _ in range(3):
                    projections["qb"].normal_()
                    positions.add_(7)
                    graph.replay()
                    with patch.dict(os.environ, {"DSV41_FUSED_DECODE_Q_ROPE": "0"}):
                        expected = compute_qkv.decode_compute_qkv(attn, x, positions)
                    self._assert_rope(
                        got.q, expected.q, projections["qb"].view_as(got.q)
                    )
                    torch.testing.assert_close(
                        got.freqs_cis, expected.freqs_cis, rtol=0, atol=0
                    )

    def _slot_inputs(self, ratio):
        eb = 128 // ratio
        table = torch.tensor(
            [[4, 0, 7], [5, 2, -1], [0, 6, 3], [8, 10, 11]],
            dtype=torch.int32,
            device="cuda",
        )
        values = torch.tensor(
            [
                -2,
                -1,
                0,
                1,
                eb - 1,
                eb,
                2 * eb - 1,
                2 * eb,
                3 * eb - 1,
                3 * eb,
                8 * eb,
                2147483647,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        selected = values.repeat(43)[:512].repeat(24, 1)
        requests = torch.arange(4, device="cuda", dtype=torch.int32).repeat_interleave(
            6
        )
        region = CSA_KV if ratio == 2 else HCA_KV
        pool = torch.empty(12, eb, 288, dtype=torch.uint8, device="cuda")
        attn = SimpleNamespace(
            compress_ratio=ratio,
            _cp_ctx=None,
            _kv_cache=SimpleNamespace(
                kernel_seq_size_per_block=128, seq_size_per_block=256
            ),
            _block_tables_by_type={region: table},
            _source_pool=lambda _: pool,
            _source_entries=lambda _, p: p.shape[1],
        )

        def reference():
            positions = (selected.long().clamp_min(0) + 1) * ratio - 1
            slots = AttentionV41FP8._slots(
                attn,
                region,
                positions.flatten(),
                requests[:, None].expand_as(selected).flatten().long(),
            ).view_as(selected)
            return slots.masked_fill(selected < 0, -1).int()

        def fused():
            return translate_local_to_global_slots(
                requests,
                table,
                selected,
                entries_per_block=eb,
                tokens_per_block_for_block_table=128 // ratio,
            )

        return table, selected, reference, fused

    def test_global_slots_match_original_slots_for_both_ratios(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                _, _, reference, fused = self._slot_inputs(ratio)
                got = fused()
                self.assertEqual(got.dtype, torch.int32)
                self.assertEqual(got.shape, (24, 512))
                torch.testing.assert_close(got, reference(), rtol=0, atol=0)

    def test_slot_graph_replay_updates_indices_and_block_table(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                table, selected, reference, fused = self._slot_inputs(ratio)
                fused()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    got = fused()
                for delta in (1, 3):
                    selected[:, :128].fill_(128 // ratio + delta)
                    table[:, 1].fill_(delta)
                    graph.replay()
                    torch.testing.assert_close(got, reference(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
