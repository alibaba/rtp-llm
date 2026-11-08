"""N5 two-phase boundary for the single-launch Q/K RoPE candidate."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _rope_tail
from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import is_supported, qk_rope


# Exact scalar operands from immutable N5/L0-cpu-edges-b05.json. Each edge is
# mapped to a distinct complex pair so all16 also fit a single Decode row.
_ROPE_EDGES = (
    (19, 46, 1.140625, -2.421875, -0.9046522974967957, 0.42615044116973877, 0, 0.00021457672119140625),
    (12, 5, 2.234375, -0.62890625, -0.2709486782550812, 0.9625937938690186, 0, -0.000019669532775878906),
    (9, 5, -3.1875, 2.65625, 0.6216099262237549, 0.7833269238471985, 1, -0.84765625),
    (10, 49, 0.71875, -0.9140625, 0.7901318669319153, -0.6129369139671326, 0, 0.007659912109375),
    (22, 28, -0.9140625, -0.1357421875, -0.9891483187675476, 0.14692047238349915, 1, -0.000025391578674316406),
    (29, 4, 3.15625, -1.28125, 0.9261506199836731, 0.37715381383895874, 1, 0.0037689208984375),
    (12, 62, -3.984375, 3.1875, 0.7808259725570679, 0.6247486472129822, 1, -0.0003490447998046875),
    (6, 62, 1.65625, -2.171875, 0.8054140210151672, 0.5927127003669739, 1, -0.76953125),
    (22, 7, 3.90625, -1.6953125, 0.9999995231628418, 0.000999772921204567, 1, -1.6953125),
    (26, 59, -2.046875, 2.1875, 0.6806167364120483, 0.7326396703720093, 1, -0.01080322265625),
    (26, 42, 0.466796875, -0.318359375, 0.5648878216743469, -0.825167715549469, 0, 0.00098419189453125),
    (3, 26, -0.8125, 0.00372314453125, 0.9999895691871643, 0.004565374460071325, 1, 0.000013768672943115234),
    (17, 27, -0.48046875, 0.55859375, 0.9914709329605103, 0.13032807409763336, 1, 0.490234375),
    (17, 22, -1.109375, 0.1328125, 0.9918851852416992, 0.12713700532913208, 1, -0.00933837890625),
    (25, 50, 5.96875, -4.6875, 0.8293553590774536, 0.5587214827537537, 1, -0.55078125),
    (2, 19, 3.3125, -0.92578125, 0.953084409236908, 0.3027046024799347, 1, 0.12060546875),
)


def _assert_rope_edges(test, tokens):
    q = torch.full((tokens, 64, 512), 0.5, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((tokens, 512), 0.25, device="cuda", dtype=torch.bfloat16)
    parts = torch.zeros(tokens, 32, 2, device="cuda", dtype=torch.float32)
    parts[..., 0] = 1
    targets = []
    for pair, (position, head, real, imag, cosine, sine, component, expected) in enumerate(_ROPE_EDGES):
        row = position if tokens == 32 else 0
        dim = 448 + 2 * pair
        values = torch.tensor([real, imag], device="cuda", dtype=torch.bfloat16)
        q[row, head, dim:dim + 2] = values
        kv[row, dim:dim + 2] = values
        parts[row, pair] = torch.tensor([cosine, sine], device="cuda", dtype=torch.float32)
        targets.append((row, head, dim + component, expected))
    freqs = torch.view_as_complex(parts).contiguous()
    saved = q.clone(), kv.clone(), freqs.clone()
    actual_q, actual_kv = qk_rope(q, kv, freqs)
    ref_q, ref_kv = _rope_tail(q, freqs), _rope_tail(kv, freqs)
    torch.cuda.synchronize()
    test.assertTrue(torch.equal(actual_q, ref_q))
    test.assertTrue(torch.equal(actual_kv, ref_kv))
    for row, head, dim, expected in targets:
        test.assertEqual(float(ref_q[row, head, dim]), expected)
        test.assertEqual(float(ref_kv[row, dim]), expected)
    for actual, before in zip((q, kv, freqs), saved):
        test.assertTrue(torch.equal(actual, before))
    test.assertEqual(len({t.data_ptr() for t in (q, kv, actual_q, actual_kv)}), 4)
    for tag in ("swa_kv", "global_kv_2"):
        test.assertTrue(torch.equal(encode_kv(actual_kv, pool_tag=tag), encode_kv(ref_kv, pool_tag=tag)))


class SharedFusedOpsDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_and_decode(self):
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(5105)
        for tokens in (32, 1):
            q = torch.randn(tokens, 10, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
            phase = torch.randn(tokens, 32, device="cuda")
            freqs = torch.polar(torch.ones_like(phase), phase).contiguous()
            saved = (q.clone(), kv.clone(), freqs.clone())
            self.assertTrue(is_supported(q, kv, freqs))
            actual_q, actual_kv = qk_rope(q, kv, freqs)
            torch.cuda.synchronize()
            ref_q, ref_kv = _rope_tail(q, freqs), _rope_tail(kv, freqs)
            # Accepted PPU FP32 complex contraction and one BF16 landing:
            # exact equality is the contract, not a fitted tolerance.
            self.assertTrue(torch.equal(actual_q, ref_q))
            self.assertTrue(torch.equal(actual_kv, ref_kv))
            for actual, expected in zip((q, kv, freqs), saved):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len({t.data_ptr() for t in (q, kv, actual_q, actual_kv)}), 4)
            for tag in ("swa_kv", "global_kv_2"):
                self.assertTrue(torch.equal(
                    encode_kv(actual_kv, pool_tag=tag), encode_kv(ref_kv, pool_tag=tag)
                ))
            self.assertFalse(is_supported(q[..., ::2], kv, freqs))
            with self.assertRaises(ValueError):
                qk_rope(q[..., ::2], kv, freqs)
        for tokens in (32, 1):
            _assert_rope_edges(self, tokens)


if __name__ == "__main__":
    unittest.main()
