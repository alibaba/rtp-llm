"""CPU tests for the PPU mHC (shifted/delayed hyper-connections) op.

Oracles:
* hand-built formulas from hc/fallback_impl.py:22-52 (the authoritative split);
* ``hc/delayed.py`` ``DelayedHCUnit``/``DelayedHCHead`` (imports cleanly on CPU;
  its CUDA branches are runtime-guarded) for the full cross-sublayer chain;
* ``hc/fallback_impl.py`` ``FallbackHCUnit`` internals for the
  ``strict_bf16_fn`` bf16-GEMM bit-match.

The Triton fast path cannot run on this CPU host; the torch path is the
authority and the tests below structure the comparisons so a remote PPU host
can re-run them with ``MHCConfig(use_triton=True)`` to compare both paths.
"""

import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import V41MHCResult
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCHead, DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.hc.fallback_impl import (
    FallbackHCUnit,
    _hc_split_sinkhorn,
)
from rtp_llm.models_py.modules.dsv41.ppu.mhc import (
    PPUMHC,
    MHCConfig,
    PPUStepMix,
    mhc_head,
    mhc_post,
    mhc_pre,
)

HC = 4
DIM = 5120
MIX_WIDTH = HC * (HC + 2)  # 24
TOKENS = 8


def _params(seed=0, dim=DIM, fn_scale=0.02, base_scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    fn = torch.randn(MIX_WIDTH, HC * dim, generator=gen) * fn_scale
    base = torch.randn(MIX_WIDTH, generator=gen) * base_scale
    scale = torch.randn(3, generator=gen)
    return {"fn": fn, "base": base, "scale": scale}


def _unit(cls, params, dim, **kwargs):
    return cls(
        params["fn"],
        params["base"],
        params["scale"],
        dim=dim,
        hc_mult=HC,
        hc_sinkhorn_iters=kwargs.get("sinkhorn_iters", 20),
        norm_eps=kwargs.get("rms_eps", 1e-6),
        hc_eps=kwargs.get("hc_eps", 1e-6),
    )


class MhcPreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.residual = torch.randn(TOKENS, HC, DIM)
        self.params = _params(seed=1)

    def test_pre_matches_handbuilt_oracle(self):
        result = mhc_pre(self.residual, self.params)

        flat = self.residual.flatten(-2).float()
        cfg = MHCConfig()
        mixes = F.linear(flat, self.params["fn"]) * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + cfg.rms_eps
        )
        pre, post, comb = _hc_split_sinkhorn(
            mixes, self.params["scale"], self.params["base"], HC, 20, 1e-6
        )

        torch.testing.assert_close(result.pre_mix, pre, rtol=0, atol=0)
        torch.testing.assert_close(result.post_mix, post.unsqueeze(-1), rtol=0, atol=0)
        torch.testing.assert_close(result.residual_mix, comb, rtol=0, atol=0)
        self.assertEqual(result.pre_mix.dtype, torch.float32)
        self.assertEqual(result.post_mix.dtype, torch.float32)
        self.assertEqual(result.residual_mix.dtype, torch.float32)
        self.assertEqual(tuple(result.post_mix.shape), (TOKENS, HC, 1))
        self.assertEqual(tuple(result.residual_mix.shape), (TOKENS, HC, HC))

    def test_first_sublayer_onehot_readout(self):
        result = mhc_pre(self.residual, self.params, state=None)
        torch.testing.assert_close(
            result.hidden, self.residual[:, 0, :], rtol=0, atol=0
        )
        # bf16 in -> bf16 out at every boundary.
        residual_bf16 = self.residual.bfloat16()
        result_bf16 = mhc_pre(residual_bf16, self.params)
        self.assertEqual(result_bf16.hidden.dtype, torch.bfloat16)
        torch.testing.assert_close(
            result_bf16.hidden, residual_bf16[:, 0, :], rtol=0, atol=0
        )

    def test_pre_readout_uses_previous_pre_mix(self):
        previous = mhc_pre(self.residual, _params(seed=2))
        result = mhc_pre(self.residual, self.params, state=previous)
        expected = (
            (self.residual.float() * previous.pre_mix.float().unsqueeze(-1))
            .sum(dim=-2)
            .to(self.residual.dtype)
        )
        torch.testing.assert_close(result.hidden, expected, rtol=0, atol=0)
        # Must NOT use this sublayer's own pre_mix.
        own = (self.residual.float() * result.pre_mix.unsqueeze(-1)).sum(dim=-2)
        self.assertFalse(torch.allclose(result.hidden, own.float()))

    def test_pre_bf16_readout_fp32_accumulate(self):
        residual = (self.residual * 0.1).bfloat16()
        previous = mhc_pre(residual, _params(seed=2))
        result = mhc_pre(residual, self.params, state=previous)
        self.assertEqual(result.hidden.dtype, torch.bfloat16)
        expected = (
            (residual.float() * previous.pre_mix.unsqueeze(-1))
            .sum(dim=-2)
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(result.hidden, expected, rtol=0, atol=0)

    def test_sinkhorn_matches_direct_iteration_and_doubly_stochastic(self):
        result = mhc_pre(self.residual, self.params)
        comb = result.residual_mix

        flat = self.residual.flatten(-2).float()
        mixes = F.linear(flat, self.params["fn"]) * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        # Independent oracle: re-derive the eps placement from the spec.
        eps = 1e-6
        raw = (
            mixes[:, 2 * HC :] * self.params["scale"][2] + self.params["base"][2 * HC :]
        ).view(TOKENS, HC, HC)
        oracle = raw.softmax(dim=-1) + eps
        oracle = oracle / (oracle.sum(dim=-2, keepdim=True) + eps)
        for _ in range(19):
            oracle = oracle / (oracle.sum(dim=-1, keepdim=True) + eps)
            oracle = oracle / (oracle.sum(dim=-2, keepdim=True) + eps)
        torch.testing.assert_close(comb, oracle, rtol=0, atol=0)
        # Doubly-stochastic pattern: row and column sums approximate 1 (the
        # eps-regularized sinkhorn converges linearly; 20 iterations leave
        # ~2e-2 on ill-conditioned rows).
        torch.testing.assert_close(
            comb.sum(dim=-1), torch.ones(TOKENS, HC), atol=5e-2, rtol=0
        )
        torch.testing.assert_close(
            comb.sum(dim=-2), torch.ones(TOKENS, HC), atol=5e-2, rtol=0
        )

    def test_config_kwargs_not_hardcoded(self):
        cfg = MHCConfig(sinkhorn_iters=3, hc_eps=1e-3, rms_eps=1e-2)
        result = mhc_pre(self.residual, self.params, config=cfg)

        flat = self.residual.flatten(-2).float()
        mixes = F.linear(flat, self.params["fn"]) * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + cfg.rms_eps
        )
        pre, post, comb = _hc_split_sinkhorn(
            mixes, self.params["scale"], self.params["base"], HC, 3, 1e-3
        )
        torch.testing.assert_close(result.pre_mix, pre, rtol=0, atol=0)
        torch.testing.assert_close(result.post_mix, post.unsqueeze(-1), rtol=0, atol=0)
        torch.testing.assert_close(result.residual_mix, comb, rtol=0, atol=0)
        # And it differs from the default config's result.
        default = mhc_pre(self.residual, self.params)
        self.assertFalse(torch.allclose(result.residual_mix, default.residual_mix))

    def test_scale_3x1_squeeze_accepted(self):
        params = dict(self.params)
        params["scale"] = self.params["scale"].reshape(3, 1)
        result = mhc_pre(self.residual, params)
        reference = mhc_pre(self.residual, self.params)
        torch.testing.assert_close(
            result.residual_mix, reference.residual_mix, rtol=0, atol=0
        )
        torch.testing.assert_close(result.pre_mix, reference.pre_mix, rtol=0, atol=0)

    def test_batched_layout_matches_flat(self):
        batched = self.residual.reshape(2, 4, HC, DIM)
        result_b = mhc_pre(batched, self.params)
        result_f = mhc_pre(self.residual, self.params)
        self.assertEqual(tuple(result_b.hidden.shape), (2, 4, DIM))
        torch.testing.assert_close(
            result_b.hidden, result_f.hidden.reshape(2, 4, DIM), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result_b.pre_mix, result_f.pre_mix.reshape(2, 4, HC), rtol=0, atol=0
        )

    def test_strict_bf16_fn_bitmatches_fallback_impl(self):
        residual = (
            torch.randn(TOKENS, HC, DIM, generator=torch.Generator().manual_seed(3))
            * 0.1
        ).bfloat16()
        unit = FallbackHCUnit(
            self.params["fn"],
            self.params["base"],
            self.params["scale"],
            dim=DIM,
            hc_mult=HC,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        result = mhc_pre(residual, self.params, config=MHCConfig(strict_bf16_fn=True))
        # Recompute FallbackHCUnit's internal mixes + split bit-for-bit.
        x_flat = residual.flatten(-2)
        rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + 1e-6).to(
            residual.dtype
        )
        mixes = (F.linear(x_flat, unit._fn_bf16()) * rsqrt).float()
        pre, post, comb = _hc_split_sinkhorn(
            mixes, self.params["scale"], self.params["base"], HC, 20, 1e-6
        )
        torch.testing.assert_close(result.pre_mix, pre, rtol=0, atol=0)
        torch.testing.assert_close(result.post_mix, post.unsqueeze(-1), rtol=0, atol=0)
        torch.testing.assert_close(result.residual_mix, comb, rtol=0, atol=0)
        # Default (fp32 fn) path differs from the bf16-fn mixes.
        default = mhc_pre(residual, self.params)
        self.assertFalse(torch.equal(default.pre_mix, result.pre_mix))


class MhcPostTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.residual = torch.randn(TOKENS, HC, DIM)
        self.branch = torch.randn(TOKENS, DIM)
        self.params = _params(seed=4)
        self.state = mhc_pre(self.residual, self.params)

    def test_post_compose_indexing(self):
        result = mhc_post(self.residual, self.branch, state=self.state)
        post = self.state.post_mix.squeeze(-1)  # [T, hc]
        comb = self.state.residual_mix  # [T, hc, hc]; comb[t, j, i]
        expected = torch.einsum("tji,tjd->tid", comb, self.residual) + post[
            ..., None
        ] * self.branch.unsqueeze(-2)
        torch.testing.assert_close(result.hidden, expected)
        self.assertEqual(tuple(result.hidden.shape), (TOKENS, HC, DIM))
        # The transposed orientation is a different tensor.
        transposed = torch.einsum("tij,tjd->tid", comb, self.residual) + post[
            ..., None
        ] * self.branch.unsqueeze(-2)
        self.assertFalse(torch.allclose(result.hidden, transposed))

    def test_post_output_carries_no_mixes(self):
        result = mhc_post(self.residual, self.branch, state=self.state)
        self.assertIsNone(result.post_mix)
        self.assertIsNone(result.residual_mix)
        self.assertIsNone(result.pre_mix)

    def test_post_bf16_boundary(self):
        residual = (self.residual * 0.1).bfloat16()
        branch = (self.branch * 0.1).bfloat16()
        state = mhc_pre(residual, self.params)
        result = mhc_post(residual, branch, state=state)
        self.assertEqual(result.hidden.dtype, torch.bfloat16)
        post = state.post_mix.squeeze(-1)
        comb = state.residual_mix
        expected = (
            post.float().unsqueeze(-1) * branch.float().unsqueeze(-2)
            + torch.matmul(comb.float().transpose(-1, -2), residual.float())
        ).to(torch.bfloat16)
        torch.testing.assert_close(result.hidden, expected, rtol=0, atol=0)

    def test_post_rejects_missing_state(self):
        with self.assertRaises(ValueError) as ctx:
            mhc_post(self.residual, self.branch, state=None)
        self.assertIn("post", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            mhc_post(
                self.residual,
                self.branch,
                state=PPUStepMix(hidden=self.residual[:, 0, :]),
            )
        self.assertIn("post_mix", str(ctx.exception))


class MhcHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.residual = torch.randn(TOKENS, HC, DIM)
        self.state = mhc_pre(self.residual, _params(seed=5))

    def test_head_uses_only_pre_mix(self):
        noisy = PPUStepMix(
            hidden=self.residual[:, 0, :],
            post_mix=torch.randn(TOKENS, HC, 1),
            residual_mix=torch.randn(TOKENS, HC, HC),
            pre_mix=self.state.pre_mix,
        )
        result = mhc_head(self.residual, state=noisy)
        expected = (
            (self.residual.float() * self.state.pre_mix.unsqueeze(-1))
            .sum(dim=-2)
            .to(self.residual.dtype)
        )
        torch.testing.assert_close(result.hidden, expected, rtol=0, atol=0)
        self.assertEqual(tuple(result.hidden.shape), (TOKENS, DIM))
        self.assertIsNone(result.post_mix)
        self.assertIsNone(result.residual_mix)

    def test_head_requires_pre_mix(self):
        with self.assertRaises(ValueError) as ctx:
            mhc_head(self.residual, state=None)
        self.assertIn("head", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            mhc_head(
                self.residual,
                state=PPUStepMix(hidden=self.residual[:, 0, :]),
            )
        self.assertIn("pre_mix", str(ctx.exception))


class DelayedChainTest(unittest.TestCase):
    """Full attn-pre -> post -> ffn-pre -> post -> head vs hc/delayed.py."""

    def test_chain_matches_delayed_hc_unit(self):
        torch.manual_seed(6)
        tokens, dim = 6, 64
        attn_params = _params(seed=7, dim=dim)
        ffn_params = _params(seed=8, dim=dim)
        residual = torch.randn(tokens, HC, dim)
        attn_branch = torch.randn(tokens, dim)
        ffn_branch = torch.randn(tokens, dim)

        # --- oracle: rtp's DelayedHCUnit (CPU path) ---
        attn = _unit(DelayedHCUnit, attn_params, dim)
        ffn = _unit(DelayedHCUnit, ffn_params, dim)
        ffn.set_previous(attn)
        y1, post1, comb1 = attn.pre(residual)
        r1 = attn.post(attn_branch, residual, post1, comb1)
        y2, post2, comb2 = ffn.pre(r1)
        r2 = ffn.post(ffn_branch, r1, post2, comb2)
        head = DelayedHCHead(ffn)
        final = head.head(r2)

        # --- ours: PPUMHC threading PPUStepMix ---
        ops = PPUMHC()
        s1 = ops(
            residual,
            None,
            phase="pre",
            parameters=attn_params,
            context=None,
            state=None,
        )
        torch.testing.assert_close(s1.hidden, y1, rtol=0, atol=0)
        p1 = ops(residual, attn_branch, phase="post", parameters=attn_params, state=s1)
        torch.testing.assert_close(p1.hidden, r1, rtol=0, atol=0)
        s2 = ops(r1, None, phase="pre", parameters=ffn_params, context=None, state=s1)
        torch.testing.assert_close(s2.hidden, y2, rtol=0, atol=0)
        p2 = ops(r1, ffn_branch, phase="post", parameters=ffn_params, state=s2)
        torch.testing.assert_close(p2.hidden, r2, rtol=0, atol=0)
        h = ops(r2, None, phase="head", parameters=ffn_params, state=s2)
        torch.testing.assert_close(h.hidden, final, rtol=0, atol=0)

        # The mixes carried by our states match the oracle's too.
        torch.testing.assert_close(s1.pre_mix, attn.pre_mix_out, rtol=0, atol=0)
        torch.testing.assert_close(s2.pre_mix, ffn.pre_mix_out, rtol=0, atol=0)
        torch.testing.assert_close(
            s2.post_mix.squeeze(-1), post2.squeeze(-1), rtol=0, atol=0
        )
        torch.testing.assert_close(s2.residual_mix, comb2, rtol=0, atol=0)

    def test_chain_vs_fallback_composition_bf16(self):
        """Composing the fallback_impl math directly (bf16, tolerance)."""
        torch.manual_seed(9)
        tokens, dim = 6, 64
        params = _params(seed=10, dim=dim)
        residual = (torch.randn(tokens, HC, dim) * 0.1).bfloat16()
        branch = (torch.randn(tokens, dim) * 0.1).bfloat16()

        unit = FallbackHCUnit(
            params["fn"],
            params["base"],
            params["scale"],
            dim=dim,
            hc_mult=HC,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        # Fallback is non-delayed: its pre readout uses its own pre. Compose the
        # delayed semantics from its primitives instead.
        x_flat = residual.flatten(-2)
        rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + 1e-6).to(
            residual.dtype
        )
        mixes = (F.linear(x_flat, unit._fn_bf16()) * rsqrt).float()
        pre, post, comb = _hc_split_sinkhorn(
            mixes, params["scale"], params["base"], HC, 20, 1e-6
        )
        r_prime = (
            post.to(residual.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
            + torch.matmul(comb.to(residual.dtype).transpose(-1, -2), residual)
        ).to(residual.dtype)
        oracle_readout = (
            (residual.float() * pre.unsqueeze(-1)).sum(dim=-2).to(residual.dtype)
        )

        # Our default path keeps fn in fp32: same math, small bf16-GEMM diffs.
        state = mhc_pre(residual, params)
        out = mhc_post(residual, branch, state=state)
        readout = mhc_head(residual, state=state)
        torch.testing.assert_close(state.pre_mix, pre, rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(
            out.hidden.float(), r_prime.float(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            readout.hidden.float(), oracle_readout.float(), rtol=1e-2, atol=1e-2
        )


class ContractTest(unittest.TestCase):
    def test_ppu_step_mix_is_frozen_contract_subclass(self):
        self.assertTrue(issubclass(PPUStepMix, V41MHCResult))
        mix = PPUStepMix(hidden=torch.zeros(2))
        self.assertIsNone(mix.post_mix)
        self.assertIsNone(mix.residual_mix)
        self.assertIsNone(mix.pre_mix)
        with self.assertRaises(Exception):
            mix.pre_mix = torch.ones(2)  # frozen
        # The base contract dataclass has no pre_mix attribute.
        base = V41MHCResult(hidden=torch.zeros(2))
        self.assertFalse(hasattr(base, "pre_mix"))

    def test_phase_dispatch_and_errors(self):
        torch.manual_seed(0)
        residual = torch.randn(TOKENS, HC, DIM)
        params = _params(seed=11)
        ops = PPUMHC()

        with self.assertRaises(ValueError) as ctx:
            ops(residual, None, phase="mid", parameters=params)
        self.assertIn("mid", str(ctx.exception))
        self.assertIn("pre", str(ctx.exception))
        self.assertIn("head", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            ops(residual, None, phase="pre", parameters=None)
        self.assertIn("fn", str(ctx.exception))

        with self.assertRaises(KeyError) as ctx:
            ops(residual, None, phase="pre", parameters={"fn": params["fn"]})
        self.assertIn("base", str(ctx.exception))
        self.assertIn("scale", str(ctx.exception))

        bad_fn = dict(params)
        bad_fn["fn"] = torch.randn(12, HC * DIM)
        with self.assertRaises(ValueError) as ctx:
            ops(residual, None, phase="pre", parameters=bad_fn)
        self.assertIn("fn", str(ctx.exception))

        bad_scale = dict(params)
        bad_scale["scale"] = torch.randn(2)
        with self.assertRaises(ValueError) as ctx:
            ops(residual, None, phase="pre", parameters=bad_scale)
        self.assertIn("scale", str(ctx.exception))

        flat = torch.randn(TOKENS, HC * DIM)
        with self.assertRaises(ValueError) as ctx:
            ops(flat, None, phase="pre", parameters=params)
        self.assertIn("residual", str(ctx.exception))

    def test_context_is_accepted_and_unused(self):
        residual = torch.randn(TOKENS, HC, DIM)
        params = _params(seed=12)
        result = PPUMHC()(
            residual,
            None,
            phase="pre",
            parameters=params,
            context=object(),
            state=None,
        )
        torch.testing.assert_close(result.hidden, residual[:, 0, :], rtol=0, atol=0)

    def test_state_without_pre_mix_selects_lane_zero(self):
        residual = torch.randn(TOKENS, HC, DIM)
        params = _params(seed=13)
        bare = V41MHCResult(hidden=residual[:, 0, :])
        result = mhc_pre(residual, params, state=bare)
        torch.testing.assert_close(result.hidden, residual[:, 0, :], rtol=0, atol=0)

    def test_mismatched_state_token_layout_rejected(self):
        residual = torch.randn(TOKENS, HC, DIM)
        params = _params(seed=14)
        other = mhc_pre(torch.randn(TOKENS + 3, HC, DIM), _params(seed=15))
        with self.assertRaises(ValueError) as ctx:
            mhc_pre(residual, params, state=other)
        self.assertIn("pre_mix", str(ctx.exception))


class TritonSelectionTest(unittest.TestCase):
    def test_use_triton_override_falls_back_on_cpu(self):
        # use_triton=True cannot launch kernels on CPU tensors; the torch
        # authority path must run instead of crashing.
        residual = torch.randn(TOKENS, HC, DIM)
        params = _params(seed=16)
        forced = mhc_pre(residual, params, config=MHCConfig(use_triton=True))
        default = mhc_pre(residual, params, config=MHCConfig(use_triton=False))
        torch.testing.assert_close(forced.hidden, default.hidden, rtol=0, atol=0)
        torch.testing.assert_close(forced.pre_mix, default.pre_mix, rtol=0, atol=0)

    def test_mhc_triton_module_is_lazily_imported(self):
        code = (
            "import sys; "
            "import rtp_llm.models_py.modules.dsv41.ppu.mhc; "
            "print('triton' in sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
