"""Q=6 numerical contracts for the PPU DSpARK verify / proposal block.

Pure torch, CPU only: this file must run without a PPU allocation, a CUDA
toolchain, or any ``rtp_llm`` import. Every helper is a literal transcription
of one kernel or reference implementation, so the assertion set is a semantic
pin rather than a re-implementation of the framework:

  * ``golden_hc_pre``      -> ``hc/fallback_impl._hc_split_sinkhorn`` and
                             ``ppu/hc/ppu_hc_head.PpuHCHead._head_chunk``
  * ``ppu_hc_pre``         -> ``kernels/ppu/hc/pre_big_fuse_kernel._mhc_pre_big_fuse``
                             (``fuse_norm=True`` path) plus
                             ``kernels/ppu/hc/ppu_hc_pre_quant``
  * ``hc_post``            -> ``kernels/ppu/hc/post_kernel._mhc_post_fwd`` and
                             ``kernels/ppu/hc/ppu_hc_sum._hc_post_sum``
  * ``golden_gate``        -> ``fused_moe/utils/fp8_fp4/gate.py``
                             ``_select_routes_with_nonfinite_fallback``
  * ``ppu_gate``           -> ``kernels/ppu/moe/ppu_gate_padding._gate_padded``
  * ``row_group_fp8``      -> ``kernels/ppu/hc/ppu_hc_pre_quant`` quant block
  * ``verify_swa_rows`` / ``draft_block_rows``
                           -> ``fp8/decode/_fused_prepare_meta_triton`` and
                              ``model_desc/deepseek_v4_dspark_model._build_noncausal_indices``

The point of the q=1/2/4/6 sweep is the DSpARK regression class: MTP2 only ever
ran q<=4, and the verify span is now gamma+1 = 6 rows. A layout mistake that
reduces HC / the router / the FP8 scale over ``[B, Q]`` instead of per token
would show up as a batched-versus-rowwise mismatch, so every numeric helper is
checked for exact row locality at all four widths.
"""

import os
import unittest

import torch


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *([os.pardir] * 6))
)

HC_MULT = 4
DIM = 4096
HIDDEN_BLOCK = 512  # math.gcd(512, 4096) in the PPU kernels
HC_EPS = 1e-6
NORM_EPS = 1e-6
SINKHORN_ITERS = 20
POST_MULT = 2.0
Q_WIDTHS = (1, 2, 4, 6)


def _fp32_ulp_bound(t: torch.Tensor, ulps: float = 2.0) -> float:
    """``ulps`` times the FP32 ULP at the tensor's largest magnitude."""
    mag = t.abs().max().item()
    if mag == 0.0:
        return 0.0
    return ulps * float(2.0 ** (torch.frexp(torch.tensor(mag)).exponent.item() - 24))


def _bf16_ulp_scale(t: torch.Tensor) -> float:
    """Largest 1-ULP step of a BF16 tensor, used for "<= N ULP" bounds."""
    mag = t.abs().max().item()
    if mag == 0.0:
        return 0.0
    return float(2.0 ** (torch.frexp(torch.tensor(mag)).exponent.item() - 8))


def _assert_close_ulp(case, got, want, ulps, label):
    """Assert ``got == want`` at ``ulps`` of the dtype's ULP at max magnitude.

    Elementwise paths in torch (sigmoid, softmax) pick their vector width from
    the element count, so a batched call can drift by a single ULP from a
    per-row call on otherwise identical data. Anything above one ULP is a real
    semantic difference, which is what these contracts guard.
    """
    if want.dtype == torch.bfloat16:
        bound = ulps * _bf16_ulp_scale(want)
    else:
        bound = _fp32_ulp_bound(want, ulps)
    diff = (got.float() - want.float()).abs().max().item()
    case.assertLessEqual(diff, bound, f"{label}: max|d|={diff:.3e} > {bound:.3e}")


# ---------------------------------------------------------------------------
# HC pre: reference readout vs the PPU fused kernel
# ---------------------------------------------------------------------------
def golden_hc_pre(x, fn, scale, base, norm_weight, *, fuse_denominator=False):
    """Reference mHC readout + RMSNorm.

    ``fuse_denominator=False`` mirrors the non-fused reference path used by
    ``Block._hc_pre_norm`` (bf16 ``x_pre`` feeds the RMSNorm, so the mean of
    squares sees bf16-rounded values). ``fuse_denominator=True`` mirrors the
    SGLang fused boundary the PPU kernel documents: the denominator uses the
    FP32 readout while the numerator keeps the bf16 boundary.
    """
    hc = HC_MULT
    x = x.to(torch.bfloat16).reshape(-1, hc, DIM)
    flat = x.flatten(-2).float()  # [T, hc*dim]
    inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = (flat @ fn.float().t()) * inv  # [T, 24]
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + HC_EPS
    post = POST_MULT * torch.sigmoid(
        mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)

    readout = (pre.unsqueeze(-1) * x.float()).sum(-2)  # [T, dim]
    denom_src = readout if fuse_denominator else readout.to(torch.bfloat16).float()
    y_inv = torch.rsqrt(denom_src.square().mean(-1, keepdim=True) + NORM_EPS)
    y = readout.to(torch.bfloat16).float() * y_inv * norm_weight.float()
    return y.to(torch.bfloat16), post, comb


def ppu_prenorm(x, fn, splits=64):
    """Split-K prenorm GEMM: ``[splits, rows, 24]`` partials reduced over splits."""
    hc = HC_MULT
    flat = x.to(torch.bfloat16).reshape(-1, hc * DIM)
    rows = flat.shape[0]
    k_total = hc * DIM
    k_chunk = (k_total + splits - 1) // splits
    partial_mixes = torch.zeros(splits, rows, (hc + 2) * hc, dtype=torch.float32)
    partial_sq = torch.zeros(splits, rows, dtype=torch.float32)
    for s in range(splits):
        lo, hi = s * k_chunk, min((s + 1) * k_chunk, k_total)
        if lo >= hi:
            continue
        chunk = flat[:, lo:hi].float()
        partial_mixes[s] = chunk @ fn.float()[:, lo:hi].t()
        partial_sq[s] = chunk.square().sum(-1)
    return partial_mixes.sum(0), partial_sq.sum(0)


def ppu_hc_pre_from_mixes(mixes, sq, x, scale, base, norm_weight):
    """Row-local tail of ``_mhc_pre_big_fuse``: rms scale, sinkhorn, readout, norm."""
    hc = HC_MULT
    x = x.to(torch.bfloat16).reshape(-1, hc, DIM)
    rows = x.shape[0]

    rms = torch.rsqrt(sq / (hc * DIM) + HC_EPS)
    mixes = (mixes * rms.unsqueeze(-1)).contiguous()

    # ``.contiguous()`` on the three column slices keeps torch's elementwise
    # vector width identical for the batched and the single-row call; a strided
    # slice moves a few values by one FP32 ULP on CPU, which is a transcription
    # artifact and not a property of the kernel.
    head = mixes[:, :hc].contiguous()
    mid = mixes[:, hc : 2 * hc].contiguous()
    tail = mixes[:, 2 * hc :].contiguous()
    pre = torch.sigmoid(head * scale[0] + base[:hc]) + HC_EPS
    post = POST_MULT * torch.sigmoid(
        mid * scale[1] + base[hc : 2 * hc]
    )
    cm = (tail * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    cm = cm.softmax(-1) + HC_EPS
    cm = cm / (cm.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + HC_EPS)
        cm = cm / (cm.sum(-2, keepdim=True) + HC_EPS)

    view = x.float()
    readout = torch.zeros(rows, DIM, dtype=torch.float32)
    sq_per_pos = torch.zeros(rows, HIDDEN_BLOCK, dtype=torch.float32)
    for i0 in range(DIM // HIDDEN_BLOCK):
        lo, hi = i0 * HIDDEN_BLOCK, (i0 + 1) * HIDDEN_BLOCK
        ol = torch.zeros(rows, hi - lo, dtype=torch.float32)
        for lane in range(hc):
            ol += pre[:, lane : lane + 1] * view[:, lane, lo:hi]
        readout[:, lo:hi] = ol
        # Lane-wise accumulation then one cross-lane reduction, exactly the
        # two-level order the kernel uses for the fused denominator.
        sq_per_pos += ol.square()
    sumsq = sq_per_pos.reshape(rows, -1).sum(-1)
    inv_rms = torch.rsqrt(sumsq / DIM + NORM_EPS)
    y = (
        readout.to(torch.bfloat16).float()
        * inv_rms.unsqueeze(-1)
        * norm_weight.float()
    )
    return y.to(torch.bfloat16), post, cm, inv_rms


def ppu_hc_pre(x, fn, scale, base, norm_weight, *, splits=64):
    """Transcription of ``_mhc_pre_big_fuse`` with ``fuse_norm=True``.

    ``splits`` plays the role of ``launch_geometry()['NUM_SPLITS']``: the
    prenorm GEMM emits ``[splits, T, 24]`` FP32 partials and the consumer
    reduces them over ``dim=0``. Per-row reductions stay per-row.
    """
    mixes, sq = ppu_prenorm(x, fn, splits)
    return ppu_hc_pre_from_mixes(mixes, sq, x, scale, base, norm_weight)


def hc_post(x, residual, post, comb):
    """Transcription of ``_mhc_post_fwd``: out[h] = post[h]*x + sum_k comb[k,h]*res[k]."""
    first = post.squeeze(-1).unsqueeze(-1) * x.unsqueeze(-2)
    second = torch.matmul(comb.transpose(-1, -2), residual.float())
    return (first.float() + second).to(torch.bfloat16)


def hc_post_naive(x, residual, post, comb):
    """Same math with an explicit [hc, hc, dim] materialisation."""
    post = post.squeeze(-1) if post.dim() == 3 else post
    out = torch.zeros_like(residual).float()
    for h in range(HC_MULT):
        acc = post[:, h].unsqueeze(-1) * x.float()
        for k in range(HC_MULT):
            acc = acc + comb[:, k, h].unsqueeze(-1) * residual[:, k, :].float()
        out[:, h, :] = acc
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# MoE gate
# ---------------------------------------------------------------------------
def golden_gate(scores, bias, topk, route_scale):
    """``_select_routes_with_nonfinite_fallback`` with sqrt-softplus scores."""
    active = torch.nn.functional.softplus(scores).sqrt()
    ranking = active + bias
    indices = ranking.topk(topk, dim=-1)[1]
    weights = active.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    return weights * route_scale, indices


def ppu_gate(scores, bias, topk, route_scale, pad_to=None):
    """``_gate_padded``: two-pass argmax select, then one FP32 normalize."""
    rows, experts = scores.shape
    pad = max(2, 1 << (topk - 1).bit_length()) if pad_to is None else pad_to
    active = torch.sqrt(torch.where(scores > 20.0, scores, torch.log1p(torch.exp(scores))))
    ranking = active + bias
    idx = torch.full((rows, pad), -1, dtype=torch.long)
    w = torch.zeros((rows, pad), dtype=torch.float32)
    rank = ranking.clone()
    for k in range(topk):
        sel = rank.argmax(dim=-1)
        w[:, k] = active.gather(1, sel.unsqueeze(-1)).squeeze(-1)
        idx[:, k] = sel
        rank = rank.scatter(1, sel.unsqueeze(-1), float("-inf"))
    # Normalization only ever touches the real topk columns.
    norm = w[:, :topk] / (w[:, :topk].sum(-1, keepdim=True) + 1e-12) * route_scale
    w = torch.cat([norm, torch.zeros((rows, pad - topk), dtype=torch.float32)], dim=1)
    return w, idx


# ---------------------------------------------------------------------------
# Per-token FP8 quantization of the HC readout
# ---------------------------------------------------------------------------
GROUP = 128


def row_group_fp8(y):
    """Transcription of the quant block in ``_mhc_pre_norm_quant``."""
    rows, dim = y.shape
    q = y.to(torch.bfloat16).float().view(rows, dim // GROUP, GROUP)
    amax = q.abs().amax(-1).clamp_min(1e-4)  # [rows, dim/128]
    inv = 448.0 / amax
    codes = (q * inv.unsqueeze(-1)).clamp(-448.0, 448.0)
    return codes, amax * (1.0 / 448.0)


# ---------------------------------------------------------------------------
# Verify / draft index geometry
# ---------------------------------------------------------------------------
WINDOW = 128
HCA_RATIO = 128
HCA_DENSE_WIDTH = 64  # max_seq_len // 128 in the metadata builder


def verify_swa_rows(start_pos, q_len, window=WINDOW):
    """``swa_abs_idx`` from ``_fused_prepare_meta_kernel`` (left aligned, -1 pad)."""
    rows = []
    dense = []
    for sp in start_pos:
        for q in range(q_len):
            abs_pos = sp + q
            win_start = max(abs_pos - window + 1, 0)
            cand = win_start + torch.arange(window)
            rows.append(torch.where(cand <= abs_pos, cand, torch.full_like(cand, -1)))
            cmp_len = (abs_pos + 1) // HCA_RATIO
            h = torch.arange(HCA_DENSE_WIDTH)
            dense.append((h, cmp_len))
    return torch.stack(rows), dense


def draft_block_rows(prefix_len, gamma, window=WINDOW):
    """``_build_noncausal_indices``: one shared context+query set for every row."""
    committed = min(prefix_len, window)
    topk = ((window + gamma + 127) // 128) * 128
    cols = torch.arange(topk)
    in_context = cols < committed
    in_query = cols < committed + gamma
    local = torch.where(
        in_context,
        prefix_len - committed + cols,
        torch.where(in_query, prefix_len + cols - committed, torch.full_like(cols, -1)),
    )
    row = local.clone()
    return row.unsqueeze(0).expand(gamma, topk).clone()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class DsparkQ6NumericTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261112)
        self.fn = torch.randn((HC_MULT + 2) * HC_MULT, HC_MULT * DIM) * 0.02
        self.scale = torch.randn(3) * 0.1 + 1.0
        self.base = torch.randn((HC_MULT + 2) * HC_MULT) * 0.1
        self.norm_weight = torch.randn(DIM) * 0.1 + 1.0

    def _hc_input(self, batch, q_len):
        x = torch.randn(batch * q_len, HC_MULT * DIM) * 0.5
        # Keep the readout far from zero: the reference norm uses rsqrt(mean^2).
        x = x + 1.0
        return x.view(batch, q_len, HC_MULT, DIM)

    def test_hc_pre_downstream_is_row_local(self):
        """From fixed mixes on, [B, Q] and a per-row loop must agree bitwise."""
        hl = (HC_MULT + 2) * HC_MULT
        for q in Q_WIDTHS:
            x = self._hc_input(3, q)
            rows = 3 * q
            mixes = torch.randn(rows, hl) * 2.0
            sq = torch.rand(rows) * 64.0 + 1.0
            y_b, post_b, comb_b, inv_b = ppu_hc_pre_from_mixes(
                mixes, sq, x, self.scale, self.base, self.norm_weight
            )
            y_l, post_l, comb_l, inv_l = [], [], [], []
            for r in range(rows):
                y, post, comb, inv = ppu_hc_pre_from_mixes(
                    mixes[r : r + 1],
                    sq[r : r + 1],
                    x.reshape(-1, HC_MULT, DIM)[r : r + 1],
                    self.scale,
                    self.base,
                    self.norm_weight,
                )
                y_l.append(y)
                post_l.append(post)
                comb_l.append(comb)
                inv_l.append(inv)
            with self.subTest(q=q):
                _assert_close_ulp(self, y_b, torch.cat(y_l), 1, f"q{q} y")
                _assert_close_ulp(self, post_b, torch.cat(post_l), 1, f"q{q} post")
                _assert_close_ulp(self, comb_b, torch.cat(comb_l), 3, f"q{q} comb")
                _assert_close_ulp(self, inv_b, torch.cat(inv_l), 1, f"q{q} inv")

    def test_hc_pre_layout_is_permutation_equivariant(self):
        """No cross-row contamination: a row permutation permutes the output."""
        for q in Q_WIDTHS:
            x = self._hc_input(3, q)
            flat = x.reshape(-1, HC_MULT, DIM)
            perm = torch.randperm(flat.shape[0])
            y_a, post_a, comb_a, _ = ppu_hc_pre(
                x, self.fn, self.scale, self.base, self.norm_weight
            )
            y_p, post_p, comb_p, _ = ppu_hc_pre(
                flat[perm], self.fn, self.scale, self.base, self.norm_weight
            )
            with self.subTest(q=q):
                _assert_close_ulp(self, y_a[perm], y_p, 1, f"q{q} y")
                _assert_close_ulp(self, post_a[perm], post_p, 1, f"q{q} post")
                _assert_close_ulp(self, comb_a[perm], comb_p, 3, f"q{q} comb")

    def test_hc_prenorm_matches_a_rowwise_gemm(self):
        """Split-K GEMM batches rows; only BLAS reduction order may differ."""
        for q in Q_WIDTHS:
            x = self._hc_input(2, q)
            flat = x.reshape(-1, HC_MULT, DIM)
            mixes_b, sq_b = ppu_prenorm(x, self.fn)
            mixes_l, sq_l = [], []
            for r in range(flat.shape[0]):
                m, s = ppu_prenorm(flat[r : r + 1], self.fn)
                mixes_l.append(m)
                sq_l.append(s)
            with self.subTest(q=q):
                self.assertLess(
                    (mixes_b - torch.cat(mixes_l)).abs().max().item(), 1e-5
                )
                self.assertLess((sq_b - torch.cat(sq_l)).abs().max().item(), 1e-2)

    def test_hc_pre_matches_reference_readout(self):
        """The fused boundary stays within 1 BF16 ULP of the reference readout."""
        for q in Q_WIDTHS:
            x = self._hc_input(2, q)
            y_ppu, post_ppu, comb_ppu, _ = ppu_hc_pre(
                x, self.fn, self.scale, self.base, self.norm_weight
            )
            y_ref, post_ref, comb_ref = golden_hc_pre(
                x, self.fn, self.scale, self.base, self.norm_weight
            )
            y_fused_ref, _, _ = golden_hc_pre(
                x,
                self.fn,
                self.scale,
                self.base,
                self.norm_weight,
                fuse_denominator=True,
            )
            with self.subTest(q=q):
                # post/comb are pure elementwise/4x4 math: tight equality.
                self.assertTrue(torch.allclose(post_ppu, post_ref, atol=2e-6))
                self.assertTrue(torch.allclose(comb_ppu, comb_ref, atol=5e-6))
                ulp = _bf16_ulp_scale(y_ref)
                diff = (y_ppu.float() - y_ref.float()).abs().max().item()
                self.assertLessEqual(diff, 2.0 * ulp)
                cos = torch.nn.functional.cosine_similarity(
                    y_ppu.float().flatten(), y_ref.float().flatten(), dim=0
                )
                self.assertGreater(cos.item(), 0.99999)
                # The documented SGLang fused boundary is the closer target.
                fused_diff = (y_ppu.float() - y_fused_ref.float()).abs().max().item()
                self.assertLessEqual(fused_diff, diff)

    def test_hc_pre_split_count_is_ulp_only(self):
        """``launch_geometry`` changes NUM_SPLITS with M; that is FP32-ULP only."""
        x = self._hc_input(1, 6)
        y64, post64, comb64, inv64 = ppu_hc_pre(
            x, self.fn, self.scale, self.base, self.norm_weight, splits=64
        )
        y16, post16, comb16, inv16 = ppu_hc_pre(
            x, self.fn, self.scale, self.base, self.norm_weight, splits=16
        )
        # Measured on M=6: inv_rms 3e-7 relative, post 1.5e-6 absolute,
        # comb 8e-7 absolute, and 1 of 24576 readout elements flips one BF16 ULP.
        rel = (inv64 - inv16).abs() / inv64
        self.assertLess(rel.max().item(), 1e-6)
        self.assertLess((post64 - post16).abs().max().item(), 5e-6)
        self.assertLess((comb64 - comb16).abs().max().item(), 5e-6)
        flips = int((y64 != y16).sum())
        self.assertLessEqual(flips, y64.numel() // 4096)

    def test_hc_post_comb_orientation(self):
        """``comb[k, h]`` + transpose is the reference writeback, not a transpose bug."""
        x = self._hc_input(1, 6)
        _, post, comb, _ = ppu_hc_pre(
            x, self.fn, self.scale, self.base, self.norm_weight
        )
        residual = self._hc_input(1, 6).reshape(-1, HC_MULT, DIM)
        sublayer = torch.randn(6, DIM) * 0.5
        out = hc_post(sublayer, residual, post, comb)
        ref = hc_post_naive(sublayer, residual, post, comb)
        ulp = _bf16_ulp_scale(ref)
        self.assertLessEqual((out.float() - ref.float()).abs().max().item(), 1.5 * ulp)

    def test_gate_rowwise_at_every_width(self):
        for q in Q_WIDTHS:
            rows = 4 * q
            scores = torch.randn(rows, 256) * 2.0
            bias = torch.randn(256) * 0.05
            w_b, i_b = ppu_gate(scores, bias, 6, 2.5)
            w_ref, i_ref = golden_gate(scores, bias, 6, 2.5)
            w_l, i_l = [], []
            for r in range(rows):
                wr, ir = ppu_gate(scores[r : r + 1], bias, 6, 2.5)
                w_l.append(wr)
                i_l.append(ir)
            with self.subTest(q=q):
                self.assertTrue(torch.equal(i_b[:, :6], i_ref))
                self.assertTrue(torch.allclose(w_b[:, :6], w_ref, atol=2e-6))
                self.assertTrue(torch.equal(w_b, torch.cat(w_l)))
                self.assertTrue(torch.equal(i_b, torch.cat(i_l)))
                # Padded routes are inert: weight exactly 0, index -1.
                self.assertTrue(torch.equal(w_b[:, 6:], torch.zeros(rows, 2)))
                self.assertTrue(torch.equal(i_b[:, 6:], torch.full((rows, 2), -1)))

    def test_gate_normalization_is_not_a_batch_scale(self):
        """A duplicated row set must produce identical weights row by row."""
        scores = torch.randn(6, 256)
        bias = torch.zeros(256)
        w6, _ = ppu_gate(scores, bias, 6, 1.0)
        w12, _ = ppu_gate(torch.cat([scores, scores]), bias, 6, 1.0)
        self.assertTrue(torch.equal(w6, w12[:6]))
        self.assertTrue(torch.allclose(w6.sum(-1), torch.full((6,), 1.0), atol=2e-6))

    def test_fp8_scale_is_per_token(self):
        """One outlier row must not move any other row's scale."""
        y = torch.randn(6, DIM)
        codes_a, scale_a = row_group_fp8(y)
        y2 = y.clone()
        y2[3] *= 1000.0
        codes_b, scale_b = row_group_fp8(y2)
        keep = torch.arange(6) != 3
        self.assertTrue(torch.equal(scale_a[keep], scale_b[keep]))
        self.assertTrue(torch.equal(codes_a[keep], codes_b[keep]))
        # And the swapped row's own scale tracks its own amax only.
        expected = y2[3].to(torch.bfloat16).float().view(-1, GROUP).abs().amax(-1)
        self.assertTrue(
            torch.allclose(scale_b[3], expected * (1.0 / 448.0), atol=1e-6)
        )

    def test_verify_rows_are_causal_and_match_single_token_decode(self):
        """Row q of a 6-row verify equals a q=1 step at its own position."""
        start = torch.tensor([0, 1, 200, 5000])
        rows6, _ = verify_swa_rows(start.tolist(), 6)
        pos = 0
        for r, sp in enumerate(start.tolist()):
            for q in range(6):
                single, _ = verify_swa_rows([sp + q], 1)
                with self.subTest(start=sp, q=q):
                    self.assertTrue(torch.equal(rows6[pos], single[0]))
                valid = rows6[pos][rows6[pos] >= 0]
                self.assertTrue(bool((valid <= sp + q).all()))
                pos += 1

    def test_hca_dense_half_is_causal_per_row(self):
        start = [pick for pick in (0, 127, 128, 129, 255, 256, 1000)]
        _, dense = verify_swa_rows(start, 6)
        pos = 0
        for sp in start:
            for q in range(6):
                h, cmp_len = dense[pos]
                with self.subTest(start=sp, q=q):
                    self.assertEqual(cmp_len, (sp + q + 1) // HCA_RATIO)
                    valid = int((h < cmp_len).sum())
                    self.assertEqual(valid, min(int(cmp_len), h.numel()))
                pos += 1
        # A later q in the same block must not see the block it completes.
        cmp = [(sp + q + 1) // HCA_RATIO for sp in start for q in range(6)]
        self.assertEqual(cmp[0], 0)

    def test_draft_block_is_deliberately_non_causal(self):
        """DSpARK's proposal rows share one index set; that is by design."""
        for gamma in (5, 6):
            rows = draft_block_rows(300, gamma)
            with self.subTest(gamma=gamma):
                self.assertEqual(rows.shape[0], gamma)
                for i in range(1, gamma):
                    self.assertTrue(torch.equal(rows[0], rows[i]))
                # The block does include the whole query block (future rows).
                valid = rows[0][rows[0] >= 0]
                self.assertEqual(int(valid.max()), 300 + gamma - 1)
                # Width is the window plus the block, rounded to 128.
                self.assertEqual(rows.shape[1] % 128, 0)
                self.assertGreaterEqual(rows.shape[1], WINDOW + gamma)


class SourceFingerprintTest(unittest.TestCase):
    """Pin the constants the transcriptions above depend on."""

    FILES = {
        "pre_big_fuse_kernel": "rtp_llm/models_py/kernels/ppu/hc/pre_big_fuse_kernel.py",
        "pre_quant": "rtp_llm/models_py/kernels/ppu/hc/ppu_hc_pre_quant.py",
        "prenorm": "rtp_llm/models_py/modules/dsv4/ppu/hc/ppu_hc_prenorm.py",
        "post": "rtp_llm/models_py/kernels/ppu/hc/post_kernel.py",
        "gate_padding": "rtp_llm/models_py/kernels/ppu/moe/ppu_gate_padding.py",
        "gate_ref": "rtp_llm/models_py/modules/factory/fused_moe/utils/fp8_fp4/gate.py",
        "meta": "rtp_llm/models_py/modules/dsv4/fp8/decode/_fused_prepare_meta_triton.py",
        "dspark": "rtp_llm/models_py/model_desc/deepseek_v4_dspark_model.py",
        "qkv_norm": "rtp_llm/models_py/kernels/ppu/attention/ppu_qkv_norm.py",
        "overlap": "rtp_llm/models_py/modules/dsv4/ppu/attention/ppu_decode_attention.py",
    }

    EXPECT = {
        "pre_big_fuse_kernel": [
            "mhc_post_mult_value",
            "mhc_pre_eps",
            "mixes_shared",
            "sumsq[0] / hidden_size + norm_eps",
            "T.Kernel(num_tokens, threads=96)",
        ],
        "pre_quant": [
            "hidden_size // 128",
            "448.0",
            "1e-4",
            "column_scales",
        ],
        "prenorm": [
            "m <= 256",
            "m <= 512",
            "m >= 8192",
            "K=16384",
            "N=24",
        ],
        "post": ["a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]"],
        "gate_padding": ["1.0e-12", "slots < K", "PAD"],
        "gate_ref": ["1e-12", "topk"],
        "meta": ["abs_pos - WINDOW + 1", "abs_pos_p1 // 128"],
        "dspark": ["prefix_lengths", "gamma"],
        "qkv_norm": ["(1, 2, 3, 4, 5, 6)", "row * RD"],
        "overlap": ["1, 2, 3, 4, 5, 6"],
    }

    def test_constants(self):
        missing_root = not os.path.isdir(os.path.join(ROOT, "rtp_llm"))
        for name, rel in self.FILES.items():
            path = os.path.join(ROOT, rel)
            if not os.path.isfile(path):
                if missing_root:
                    self.skipTest("rtp_llm source tree not next to the test")
                self.fail(f"missing source file {rel}")
            with open(path, "r", encoding="utf-8") as handle:
                body = handle.read()
            for needle in self.EXPECT[name]:
                with self.subTest(file=name, needle=needle):
                    self.assertIn(needle, body)

    def test_no_q4_cap_left_on_the_ppu_verify_path(self):
        for name in ("meta", "overlap", "qkv_norm"):
            path = os.path.join(ROOT, self.FILES[name])
            if not os.path.isfile(path):
                self.skipTest("rtp_llm source tree not next to the test")
            with open(path, "r", encoding="utf-8") as handle:
                body = handle.read()
            with self.subTest(file=name):
                self.assertNotIn("(1, 2, 3, 4)", body)
                self.assertNotIn("q_len=1/2/3/4", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
