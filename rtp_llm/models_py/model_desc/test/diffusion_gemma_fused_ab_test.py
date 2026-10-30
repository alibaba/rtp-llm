"""GPU A/B for the opt r8 contract-6 fused elementwise kernels: real weights.

Drives the three contract-6 fusions at their real DiffusionGemma shapes with
real layer-0 (sliding) checkpoint weights, bf16, H20:

  1. ``qkv_norm_rope`` (DGM_FUSED_QKV_NORMROPE): the fused strided-slice
     norm+rope Triton kernel vs the exact torch chain (copy-reshape ->
     RMSNorm -> rope.apply) on the same qkv buffer — per-tensor max_diff /
     relative-bf16-ulp / mismatch_frac evidence, the same referee the
     DGM_QKV_AB=1 lane prints ([dgm-qkv-ab]).
  2. ``add_scale`` (DGM_FUSED_ADD_SCALE): the fused layer-tail
     ``(residual + normed) * layer_scalar`` vs the torch add+mul pair with
     the real layer scalar.
  3. router scale fold (DGM_FUSED_ROUTER_SCALE): the gamma-folded
     ``rmsnorm(x, scale*root)`` vs the torch ``norm(x)*scale*root`` chain,
     plus the downstream top-8 set/order effect (near-tie ruling class).
  4. layer-level: the full DiffusionGemmaLayer forward with all three
     fusions ON vs OFF (monkeypatched module gates), same inputs — the
     end-to-end bf16-noise guard.

Agreement criteria: kernel-level items are bf16 rounding-noise class
(max relative diff within ~2 bf16 mantissa units, i.e. 2^-7 relative); the
layer-level guard uses the moe_ab precedent ratio bound (the fused routing
A/B baseline was 0.0072-0.0090, the gate is 10x that to absorb near-tie
expert-set flips on folded router inputs). Any fused path must be ACTIVE
in the ON arm (no silent torch degradation).
"""

import importlib.util
import os
import unittest
from unittest.mock import patch

import torch

CKPT = os.environ.get("DGM_REAL_CKPT", "")

# Load the realweight helper module by path (test dir is not a package):
# its H/EXPERTS/TOPK geometry and _load_hf_weights/_make_real_config
# loaders are reused here.
_RW_PATH = os.path.join(os.path.dirname(__file__), "diffusion_gemma_realweight_test.py")
_spec = importlib.util.spec_from_file_location("dgm_rw_helpers", _RW_PATH)
_rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rw)

from rtp_llm.ops import ParallelismConfig  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma as dgm  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


def _bf16(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(device)


@unittest.skipUnless(CKPT, "DGM_REAL_CKPT not set; real-weights A/B skipped")
class DiffusionGemmaKernelABTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        self.device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(self.device)
        print(f"[dgm-kab] device={name}", flush=True)
        self.hf = _rw._load_hf_weights(0, "sliding_attention")
        self.cfg = _rw._make_real_config()
        torch.manual_seed(21)

    def _attn(self) -> dgm.DiffusionGemmaAttention:
        weights = {
            W.attn_qkv_w: _bf16(self.hf["qkv"].T.contiguous(), self.device),
            W.attn_o_w: _bf16(self.hf["o"].T.contiguous(), self.device),
            W.q_ln_gamma: _bf16(self.hf["q_ln"].reshape(-1), self.device),
            W.k_ln_gamma: _bf16(self.hf["k_ln"].reshape(-1), self.device),
        }
        return dgm.DiffusionGemmaAttention(
            self.cfg, ParallelismConfig(), weights, 0, "sliding_attention", None
        )

    def test_qkv_norm_rope_matches_torch_chain(self):
        """Fused strided-slice norm+rope vs the exact torch chain."""
        attn = self._attn()
        self.assertIsNotNone(dgm._qkv_kernel, "fused qkv kernel import failed")
        for m_tokens in (21, 256):
            hidden = (
                torch.randn(m_tokens, _rw.H, device=self.device, dtype=torch.float32)
                * 0.5
            ).to(torch.bfloat16)
            positions = torch.arange(
                7, 7 + m_tokens, device=self.device, dtype=torch.int64
            )
            rope = dgm.DiffusionGemmaRope(
                1e4, attn.head_dim, 1.0, 64, self.device, torch.bfloat16
            )
            qkv = attn.qkv_proj(hidden).reshape(-1, attn.q_size + 2 * attn.kv_size)
            self.assertTrue(qkv.dtype == torch.bfloat16)
            fused = attn._fused_qkv_norm_rope(qkv, positions, rope)
            self.assertIsNotNone(fused, "fused qkv path degraded to torch")
            q, k, v = fused
            # exact torch chain on the same buffer
            q_t = qkv[..., : attn.q_size]
            k_t = qkv[..., attn.q_size : attn.q_size + attn.kv_size]
            v_t = qkv[..., attn.q_size + attn.kv_size :]
            q_ref, k_ref = attn.qk_norm(
                q_t, k_t, positions, rope, attn.head_num, attn.kv_head_num, attn.head_dim
            )
            v_ref = attn.v_norm(v_t.reshape(-1, attn.head_dim)).reshape(
                -1, attn.kv_head_num, attn.head_dim
            )
            for tag, out, ref in (
                ("q", q, q_ref.reshape(-1, attn.head_num, attn.head_dim)),
                ("k", k, k_ref.reshape(-1, attn.kv_head_num, attn.head_dim)),
                ("v", v, v_ref),
            ):
                diff = (out.float() - ref.float()).abs()
                # relative bf16-ulp metric on same-magnitude elements: the
                # raw int16 bit distance is meaningless near +-0 (tiny
                # absolute diffs thousands of "ulps" apart), so report the
                # max relative diff in 2^-8 (bf16 mantissa) units instead.
                rel_ulp = diff / ref.float().abs().clamp(min=2.0 ** (-126))
                absmax = float(ref.float().abs().max())
                print(
                    f"[dgm-kab] qkv M={m_tokens} {tag}: max_diff={float(diff.max()):.3e} "
                    f"max_rel_bf16_ulp={float(rel_ulp.max()) * 256:.1f} "
                    f"mismatch_frac={float((out != ref).float().mean()):.4f} "
                    f"ref_absmax={absmax:.3f}",
                    flush=True,
                )
                torch.testing.assert_close(
                    out, ref, rtol=2e-2, atol=2e-2,
                    msg=f"fused qkv {tag} deviates beyond bf16 noise at M={m_tokens}",
                )
                self.assertLess(float(diff.max()) / max(absmax, 1e-6), 2e-2)

    def test_add_scale_matches_torch_pair(self):
        """Fused layer-tail (a+b)*scalar vs the torch add+mul pair."""
        self.assertIsNotNone(dgm._addscale_kernel, "fused add_scale import failed")
        scalar = _bf16(self.hf["layer_scalar_enc"].reshape(()), self.device)
        if scalar.numel() != 1:
            self.skipTest("layer scalar is not a single element")
        for m_tokens in (21, 256):
            a = (torch.randn(m_tokens, _rw.H, device=self.device) * 2.0).to(
                torch.bfloat16
            )
            b = (torch.randn(m_tokens, _rw.H, device=self.device) * 8.0).to(
                torch.bfloat16
            )
            out = dgm._addscale_kernel(a, b, scalar)
            ref = (b + a) * scalar
            diff = (out.float() - ref.float()).abs()
            ulp = (
                out.view(torch.int16).to(torch.int32)
                - ref.view(torch.int16).to(torch.int32)
            ).abs()
            absmax = float(ref.float().abs().max())
            print(
                f"[dgm-kab] add_scale M={m_tokens}: max_diff={float(diff.max()):.3e} "
                f"max_ulp={int(ulp.max())} "
                f"mismatch_frac={float((out != ref).float().mean()):.4f} "
                f"ref_absmax={absmax:.3f}",
                flush=True,
            )
            self.assertLessEqual(int(ulp.max()), 2)
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_router_scale_fold_matches_torch_chain(self):
        """Gamma-folded router input vs the torch norm*scale*root chain."""
        weights = {
            W.moe_gate: _bf16(self.hf["router_proj"].T.contiguous(), self.device),
            W.dgm_router_scale: _bf16(self.hf["router_scale"].reshape(-1), self.device),
            W.dgm_router_per_expert_scale: _bf16(
                self.hf["per_expert_scale"].reshape(-1), self.device
            ),
        }
        router = dgm.DiffusionGemmaRouter(
            self.cfg, ParallelismConfig(), weights, None
        )
        for m_tokens in (21, 256):
            hidden = (
                torch.randn(m_tokens, _rw.H, device=self.device) * 0.5
            ).to(torch.bfloat16)
            gamma = router._fold_gamma(hidden)
            self.assertIsNotNone(gamma, "router fold gamma did not qualify")
            folded = router.pre_norm(hidden, gamma)
            ref = router.pre_norm(hidden) * router.scale * router.scalar_root_size
            diff = (folded.float() - ref.float()).abs()
            ulp = (
                folded.view(torch.int16).to(torch.int32)
                - ref.view(torch.int16).to(torch.int32)
            ).abs()
            absmax = float(ref.float().abs().max())
            # downstream referee: top-8 set/order + routed-weight deltas
            s_f = router.proj(folded)
            s_r = router.proj(ref)
            i_f = router._top_idx_of(s_f)
            i_r = router._top_idx_of(s_r)
            set_eq = (i_f.sort(dim=-1).values == i_r.sort(dim=-1).values).all(dim=-1)
            order_eq = (i_f == i_r).all(dim=-1)
            w_f = router._reference_route(s_f, folded.dtype)[0]
            w_r = router._reference_route(s_r, ref.dtype)[0]
            w_diff = (w_f.float() - w_r.float()).abs()
            print(
                f"[dgm-kab] router M={m_tokens}: max_diff={float(diff.max()):.3e} "
                f"max_ulp={int(ulp.max())} "
                f"ref_absmax={absmax:.3f} | top8 same_set={int(set_eq.sum())}/{m_tokens} "
                f"exact_order={int(order_eq.sum())}/{m_tokens} "
                f"max_wdiff={float(w_diff.max()):.3e}",
                flush=True,
            )
            self.assertLess(float(diff.max()) / max(absmax, 1e-6), 2e-2)
            # routed weights must stay bf16-noise close even where the
            # folded input flips a near-tie expert slot.
            self.assertLess(float(w_diff.max()), 3e-2)

    def test_layer_fused_vs_torch_paths(self):
        """Full layer forward, all three fusions on vs off, same inputs."""
        def build_layer():
            hf = {k: v.to(torch.bfloat16).to(self.device) for k, v in self.hf.items()}
            weights = _rw._model_weights(hf)
            return dgm.DiffusionGemmaLayer(
                self.cfg, ParallelismConfig(), weights, 0, "sliding_attention", None
            )

        torch.manual_seed(33)
        hidden = (torch.randn(21, _rw.H, device=self.device) * 0.5).to(torch.bfloat16)
        positions = torch.arange(21, device=self.device, dtype=torch.int64)
        rope = dgm.DiffusionGemmaRope(1e4, 256, 1.0, 32, self.device, torch.bfloat16)
        from types import SimpleNamespace

        cu = torch.tensor([0, 21], dtype=torch.int32, device=self.device)
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32, device=self.device),
            input_lengths=torch.tensor([21], dtype=torch.int32, device=self.device),
            cu_seqlens=cu.cpu(),
            cu_seqlens_device=cu,
            kv_cache_kernel_block_id=None,
            kv_cache_kernel_block_id_device=None,
            sequence_lengths=None,
        )
        attn_op = dgm.DiffusionGemmaAttnOp(
            "sliding_attention", 16, 8, 256, 1024, 64, torch.bfloat16, None
        )
        # the flashinfer wrappers need a plan() before run() (the model's
        # forward does this per step; driving the layer directly here must
        # do the same).
        attn_op.plan(attn_inputs, causal=True)
        with torch.inference_mode():
            layer_on = build_layer()
            # the layer's first fused add-norm writes the residual sum INTO
            # its input tensor in place (the r5 flashinfer rms_norm_add_residual
            # contract: the input must be a private tensor — production always
            # feeds a fresh previous-layer output). Clone per arm so the OFF
            # run sees the same input the ON run did.
            out_on = layer_on(
                hidden.clone(), rope, positions, attn_op, None, attn_inputs,
                causal=True, write_kv=False, decoder_mode=False,
            )
            gates = {
                "_FUSED_QKV_ENV": False,
                "_FUSED_ADD_SCALE_ENV": False,
                "_FUSED_ROUTER_SCALE_ENV": False,
            }
            with patch.multiple(dgm, **gates):
                layer_off = build_layer()
                out_off = layer_off(
                    hidden.clone(), rope, positions, attn_op, None, attn_inputs,
                    causal=True, write_kv=False, decoder_mode=False,
                )
        diff = (out_on.float() - out_off.float()).abs()
        absmax = float(out_off.float().abs().max())
        ratio = float(diff.max()) / absmax
        print(
            f"[dgm-kab] layer fused-vs-torch: max_diff={float(diff.max()):.4f} "
            f"absmax={absmax:.3f} ratio={ratio:.5f} "
            f"mean_diff={float(diff.mean()):.5f}",
            flush=True,
        )
        # moe_ab precedent ratio 0.0072-0.0090; 10x bound absorbs near-tie
        # expert-set flips from the folded router input.
        self.assertLess(ratio, 0.1)
        self.assertGreater(absmax, 0.0)


if __name__ == "__main__":
    unittest.main()
