"""GPU tests for the SM90 wo_a path: weight regrouping and the grouped GEMM.

``fp8_einsum("bhr,hdr->bhd")`` is Blackwell-only, so on SM90 the wo_a projection
runs as one ``fp8_gemm_nt`` per ``o_group`` over strided views. Two things need
pinning and neither had a test:

* :func:`prepare_wo_a_grouped` reshapes the checkpoint's ``[G*R, K]`` weight and
  ``[G*R/128, K/128]`` scale into the per-group operands the GEMM takes. The claim
  is that group ``h`` is exactly rows ``[h*R, (h+1)*R)`` of the checkpoint tensor,
  and the same row range divided by 128 for the scale -- an off-by-one there
  produces a model that loads and serves confidently wrong numbers.
* :func:`wo_a_grouped_gemm` must compute ``out[m, h, :] = o[m, h, :] @ w[h].T``
  with the activation's per-token 1x128 scale and the weight's per-128x128-block
  scale, i.e. ``recipe=(1, 128, 128)``.

The scale *unpack* is covered without a GPU in ``wo_a_sm90_layout_test.py``.

Runs on the H20 lane. A box with CUDA where DeepGEMM cannot be imported fails
rather than skips: that is a broken build, not an absent capability.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_CUDA = torch.cuda.is_available()
_IMPORT_ERROR = None
try:
    import deep_gemm  # noqa: F401

    from rtp_llm.models_py.modules.dsv4.fp8._wo_a_sm90 import (
        prepare_wo_a_grouped,
        wo_a_grouped_gemm,
    )
except Exception as exc:  # noqa: BLE001 - re-raised as a failure below
    _IMPORT_ERROR = exc

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")


def _e8m0(exponents: torch.Tensor) -> torch.Tensor:
    """Build a ``float8_e8m0fnu`` tensor from biased exponent bytes."""
    return exponents.to(torch.uint8).view(torch.float8_e8m0fnu)


def _pack_ue8m0(exponents: torch.Tensor) -> torch.Tensor:
    """``[..., k_blocks]`` biased exponents -> ``[..., ceil(k_blocks/4)]`` int32.

    The layout ``fused_inv_rope_fp8_quant`` emits: four bytes per word, least
    significant first.
    """
    *lead, k_blocks = exponents.shape
    words = -(-k_blocks // 4)
    padded = torch.zeros(*lead, words * 4, dtype=torch.int64, device=exponents.device)
    padded[..., :k_blocks] = exponents.to(torch.int64)
    chunks = padded.reshape(*lead, words, 4)
    word = (
        chunks[..., 0]
        | (chunks[..., 1] << 8)
        | (chunks[..., 2] << 16)
        | (chunks[..., 3] << 24)
    )
    # Reinterpret into int32's signed range, as the kernel's store does.
    word = torch.where(word >= (1 << 31), word - (1 << 32), word)
    return word.to(torch.int32)


class WoASm90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("needs a CUDA GPU")
        if _IMPORT_ERROR is not None:
            raise AssertionError(
                "CUDA is present but DeepGEMM or the wo_a module could not be "
                f"imported, which is a build problem: {_IMPORT_ERROR!r}"
            )
        cls.device = "cuda:0"

    # ---- prepare_wo_a_grouped -------------------------------------------------

    @unittest.skipUnless(_HAS_E8M0, "torch build has no float8_e8m0fnu")
    def test_prepare_maps_each_group_to_its_row_range(self):
        # (G, R, K): the second shape is the tp_size > 1 form, where the rank holds
        # a row slice of every group rather than a subset of the groups.
        for G, R, K in ((2, 256, 512), (2, 128, 512), (4, 128, 256)):
            with self.subTest(G=G, R=R, K=K):
                g = torch.Generator(device="cpu").manual_seed(G * 1000 + R)
                # Values chosen so the fp8 cast is exact and each row is distinct.
                raw = (torch.randint(-4, 5, (G * R, K), generator=g)).float()
                weight_fp8 = raw.to(self.device).to(torch.float8_e4m3fn)
                exps = torch.randint(
                    120, 135, (G * R // 128, K // 128), generator=g
                ).to(self.device)
                scale_raw = _e8m0(exps)

                w, s = prepare_wo_a_grouped(weight_fp8, scale_raw, G, R, K)

                self.assertEqual(tuple(w.shape), (G, R, K))
                self.assertEqual(tuple(s.shape), (G, R // 128, K // 128))
                self.assertEqual(s.dtype, torch.float32)
                for h in range(G):
                    self.assertTrue(
                        torch.equal(
                            w[h].reshape(-1).view(torch.uint8),
                            weight_fp8[h * R : (h + 1) * R].reshape(-1).view(
                                torch.uint8
                            ),
                        ),
                        f"weight group {h} is not rows [{h * R}, {(h + 1) * R})",
                    )
                    want = torch.exp2(
                        exps[h * (R // 128) : (h + 1) * (R // 128)].float() - 127.0
                    )
                    self.assertTrue(
                        torch.equal(s[h], want),
                        f"scale group {h} is not the matching row range",
                    )

    @unittest.skipUnless(_HAS_E8M0, "torch build has no float8_e8m0fnu")
    def test_prepare_rejects_the_packed_scale_form(self):
        """The SM100 path's int32-packed scale cannot be un-packed per block here."""
        G, R, K = 2, 128, 512
        weight_fp8 = torch.zeros(G * R, K, device=self.device).to(torch.float8_e4m3fn)
        packed = torch.zeros(
            G * R // 128, K // 512, dtype=torch.int32, device=self.device
        )
        with self.assertRaisesRegex(AssertionError, "e8m0fnu"):
            prepare_wo_a_grouped(weight_fp8, packed, G, R, K)

    # ---- wo_a_grouped_gemm ---------------------------------------------------

    @unittest.skipUnless(_HAS_E8M0, "torch build has no float8_e8m0fnu")
    def test_grouped_gemm_matches_a_dequantised_reference(self):
        M, G, R, K = 64, 2, 256, 512
        gen = torch.Generator(device="cpu").manual_seed(20260816)

        act = torch.randint(-6, 7, (M, G, K), generator=gen).float()
        o_fp8 = act.to(self.device).to(torch.float8_e4m3fn)
        act_exps = torch.randint(124, 131, (M, G, K // 128), generator=gen).to(
            self.device
        )
        o_scale_packed = _pack_ue8m0(act_exps)
        self.assertEqual(tuple(o_scale_packed.shape), (M, G, K // 512))

        raw_w = torch.randint(-4, 5, (G * R, K), generator=gen).float()
        weight_fp8 = raw_w.to(self.device).to(torch.float8_e4m3fn)
        w_exps = torch.randint(124, 131, (G * R // 128, K // 128), generator=gen).to(
            self.device
        )
        weight, weight_scale = prepare_wo_a_grouped(
            weight_fp8, _e8m0(w_exps), G, R, K
        )

        out = wo_a_grouped_gemm(o_fp8, o_scale_packed, weight, weight_scale)
        self.assertEqual(tuple(out.shape), (M, G, R))
        self.assertEqual(out.dtype, torch.bfloat16)

        act_scale = torch.exp2(act_exps.float() - 127.0)  # [M, G, K/128]
        for h in range(G):
            a = o_fp8[:, h, :].float() * act_scale[:, h, :].repeat_interleave(
                128, dim=-1
            )
            w = weight[h].float() * weight_scale[h].repeat_interleave(
                128, dim=0
            ).repeat_interleave(128, dim=1)
            ref = a @ w.T
            got = out[:, h, :].float()
            rel = ((got - ref).norm() / ref.norm().clamp(min=1e-9)).item()
            self.assertLess(rel, 2e-2, f"group {h}: rel_fro = {rel}")

    @unittest.skipUnless(_HAS_E8M0, "torch build has no float8_e8m0fnu")
    def test_grouped_gemm_writes_into_a_supplied_out(self):
        """``wo_b`` hands its own ``[M, G, R]`` buffer down; it must be filled."""
        M, G, R, K = 32, 2, 128, 512
        gen = torch.Generator(device="cpu").manual_seed(5)
        o_fp8 = (
            torch.randint(-3, 4, (M, G, K), generator=gen)
            .float()
            .to(self.device)
            .to(torch.float8_e4m3fn)
        )
        packed = _pack_ue8m0(
            torch.full((M, G, K // 128), 127, dtype=torch.int64).to(self.device)
        )
        weight, weight_scale = prepare_wo_a_grouped(
            torch.randint(-3, 4, (G * R, K), generator=gen)
            .float()
            .to(self.device)
            .to(torch.float8_e4m3fn),
            _e8m0(torch.full((G * R // 128, K // 128), 127).to(self.device)),
            G,
            R,
            K,
        )
        out = torch.full((M, G, R), 7.0, dtype=torch.bfloat16, device=self.device)
        returned = wo_a_grouped_gemm(o_fp8, packed, weight, weight_scale, out=out)
        self.assertIs(returned, out)
        self.assertFalse(
            bool((out == 7.0).all()), "out was returned untouched"
        )


if __name__ == "__main__":
    unittest.main()
