"""Correctness UT for ``flash_mla_sparse_fp8_fwd`` (SM100 only).

Compares FP8 sparse prefill output against the BF16 reference kernel
``flash_mla_sparse_fwd`` (same kernel used by rtp-llm's DSV4 fp8
``attention.py`` today) and checks:

  1. the 16B slot at [512:528] is padding — kernel does not read it
     (bit-identical output whether the slot is zeroed or filled with garbage);
  2. FP8 output vs BF16 reference rel-diff below a per-shape tolerance.

FP8 kernel constraints:
  - Single per-tensor ``q_scale`` and ``k_scale`` (no per-token/per-group).
  - ``qk_scale = q_scale * k_scale`` is applied uniformly to the full raw P
    (both NoPE and RoPE contributions) after the QK MMA.
  - Consequence: Q_RoPE must be PRE-DIVIDED by ``qk_scale`` on the host so
    the kernel's uniform multiply produces the correct bf16-equivalent RoPE
    contribution.

Runs under bazel (py_test); requires Blackwell SM100 GPU (skips otherwise).
"""

import unittest

import torch
from flash_mla import flash_mla_sparse_fp8_fwd, flash_mla_sparse_fwd


class FlashMLASM100SparsePrefillFP8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cc = torch.cuda.get_device_capability(cls.device)

    def _require_sm100(self):
        if self.cc[0] != 10:
            self.skipTest(
                f"sparse_prefill_fp8_fwd requires SM100 (Blackwell); "
                f"got compute capability {self.cc}"
            )

    # ------------------------------------------------------------------
    # Packers.
    #
    # Layout (656 bytes/token):
    #   [0:512]   512 x float8_e4m3   NoPE  (scaled by k_scale for KV,
    #                                        q_scale for Q)
    #   [512:528] 16  bytes           padding (kernel does not read)
    #   [528:656] 64  x bfloat16      RoPE  (for Q: pre-divided by qk_scale
    #                                        so the kernel's uniform multiply
    #                                        cancels; for KV: raw bf16)
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_kv_656(kv_bf16: torch.Tensor):
        """Per-tensor FP8 pack. Returns (packed_bytes[s_kv, 656], k_scale: float)."""
        kv = kv_bf16.reshape(-1, 576)
        s_kv = kv.shape[0]
        nope = kv[:, :512].float()
        rope = kv[:, 512:]
        k_scale = max(nope.abs().max().item() / 448.0, 1e-6)
        nope_fp8 = (
            (nope / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        padding = torch.zeros(s_kv, 16, dtype=torch.uint8, device=kv.device)
        rope_u8 = rope.reshape(s_kv, 64).view(torch.uint8).view(s_kv, 128)
        return torch.cat([nope_fp8, padding, rope_u8], dim=1).contiguous(), k_scale

    @staticmethod
    def _pack_q_656(q_bf16: torch.Tensor, k_scale: float):
        """Per-tensor FP8 pack for Q. Q_RoPE is PRE-DIVIDED by ``qk_scale`` so
        the kernel's uniform ``qk_scale`` multiply cancels out and yields the
        correct RoPE contribution.

        Returns (packed_bytes[s_q, h_q, 656], q_scale: float).
        """
        s_q, h_q, d_qk = q_bf16.shape
        assert d_qk == 576
        nope = q_bf16[..., :512].float()
        rope = q_bf16[..., 512:].float()
        q_scale = max(nope.abs().max().item() / 448.0, 1e-6)
        qk_scale = q_scale * k_scale
        nope_fp8 = (
            (nope / q_scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        rope_prediv = (rope / qk_scale).to(torch.bfloat16)
        padding = torch.zeros(s_q, h_q, 16, dtype=torch.uint8, device=q_bf16.device)
        rope_u8 = rope_prediv.contiguous().view(torch.uint8).view(s_q, h_q, 128)
        return torch.cat([nope_fp8, padding, rope_u8], dim=-1).contiguous(), q_scale

    # ------------------------------------------------------------------
    # Reference/oracle
    # ------------------------------------------------------------------
    @staticmethod
    def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return (
            a.float() - b.float()
        ).abs().mean().item() / b.float().abs().mean().clamp_min(1e-9).item()

    def _make_inputs(self, s_q, s_kv, topk, seed=42, invalid_frac=0.0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        q = torch.randn(
            (s_q, 64, 576), device=self.device, dtype=torch.bfloat16, generator=g
        )
        kv = torch.randn(
            (s_kv, 1, 576), device=self.device, dtype=torch.bfloat16, generator=g
        )
        idx_rows = []
        for _ in range(s_q):
            perm = torch.randperm(
                s_kv, device=self.device, generator=g, dtype=torch.int32
            )
            row = perm[:topk].clone()
            if invalid_frac > 0.0:
                n_inv = int(topk * invalid_frac)
                row[-n_inv:] = -1
            idx_rows.append(row)
        indices = torch.stack(idx_rows).view(s_q, 1, topk)
        return q, kv, indices

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_padding_slot_is_unused(self):
        """Kernel must ignore bytes [512:528] in both Q and KV packs.
        Zeroed pad vs random-garbage pad must produce bit-identical output.
        """
        self._require_sm100()
        s_q, s_kv, topk = 128, 1024, 128
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=101)
        sm_scale = 576**-0.5

        kv_zero, k_scale = self._pack_kv_656(kv)
        q_zero, q_scale = self._pack_q_656(q, k_scale)

        g = torch.Generator(device=self.device).manual_seed(999)
        kv_junk = kv_zero.clone()
        kv_junk[:, 512:528] = torch.randint(
            0, 256, (s_kv, 16), device=self.device, generator=g, dtype=torch.int64
        ).to(torch.uint8)
        q_junk = q_zero.clone()
        q_junk[:, :, 512:528] = torch.randint(
            0, 256, (s_q, 64, 16), device=self.device, generator=g, dtype=torch.int64
        ).to(torch.uint8)

        out_zero, _, _ = flash_mla_sparse_fp8_fwd(
            q_zero, kv_zero, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        out_junk, _, _ = flash_mla_sparse_fp8_fwd(
            q_junk, kv_junk, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        max_abs = (out_zero.float() - out_junk.float()).abs().max().item()
        self.assertLess(
            max_abs,
            1e-6,
            f"kernel appears to read the 16B pad slot: max|zero - junk| = {max_abs}",
        )

    def test_fp8_vs_bf16_glm_shape(self):
        """GLM5-like shape: rel_diff < 0.05 vs bf16 reference."""
        self._require_sm100()
        s_q, s_kv, topk = 4096, 32768, 2048
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=42)
        sm_scale = 576**-0.5

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512)
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg, kv_pkg, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.05, f"rel_diff={rel:.4f} > 0.05")

    def test_fp8_vs_bf16_tiny(self):
        self._require_sm100()
        s_q, s_kv, topk = 64, 512, 64
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=7)
        sm_scale = 576**-0.5

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512)
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg, kv_pkg, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.05, f"rel_diff={rel:.4f} > 0.05")

    def test_topk_length_partial(self):
        self._require_sm100()
        s_q, s_kv, topk = 128, 2048, 256
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=11)
        sm_scale = 576**-0.5

        topk_length = torch.full(
            (s_q,), topk // 2, dtype=torch.int32, device=self.device
        )

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, d_v=512, topk_length=topk_length
        )
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg,
            kv_pkg,
            indices,
            sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            topk_length=topk_length,
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.05, f"rel_diff={rel:.4f} > 0.05")

    def test_attn_sink(self):
        self._require_sm100()
        s_q, s_kv, topk = 128, 2048, 256
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=13)
        sm_scale = 576**-0.5

        attn_sink = torch.randn(64, device=self.device, dtype=torch.float32)

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, d_v=512, attn_sink=attn_sink
        )
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg,
            kv_pkg,
            indices,
            sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            attn_sink=attn_sink,
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.05, f"rel_diff={rel:.4f} > 0.05")

    def test_invalid_indices(self):
        """indices with -1 sentinels — bf16 kernel supports them; fp8 must too."""
        self._require_sm100()
        s_q, s_kv, topk = 128, 2048, 256
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=17, invalid_frac=0.25)
        sm_scale = 576**-0.5

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512)
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg, kv_pkg, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.05, f"rel_diff={rel:.4f} > 0.05")

    def test_extreme_per_token_amax_variance(self):
        """Half tokens amax≈0.01, half amax≈10. Per-tensor scale is set by the
        large group; small-amax tokens lose resolution — accuracy is expected
        to degrade, but should not diverge catastrophically."""
        self._require_sm100()
        s_q, s_kv, topk = 128, 512, 128
        q, kv, indices = self._make_inputs(s_q, s_kv, topk, seed=19)

        kv = kv.clone()
        half = s_kv // 2
        kv[:half, :, :512] *= 0.01
        kv[half:, :, :512] *= 10.0

        sm_scale = 576**-0.5

        kv_pkg, k_scale = self._pack_kv_656(kv)
        q_pkg, q_scale = self._pack_q_656(q, k_scale)

        out_bf16, _, _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512)
        out_fp8, _, _ = flash_mla_sparse_fp8_fwd(
            q_pkg, kv_pkg, indices, sm_scale, q_scale=q_scale, k_scale=k_scale
        )
        rel = self._rel_diff(out_fp8, out_bf16)
        self.assertLess(rel, 0.20, f"extreme amax variance rel_diff={rel:.4f} > 0.20")


if __name__ == "__main__":
    unittest.main(verbosity=2)
