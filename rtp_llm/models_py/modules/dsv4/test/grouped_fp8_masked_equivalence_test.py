"""Numerical-equivalence test for the masked layout of the grouped-FP8 MoE path.

``GroupedFP8Strategy`` has two implementations of the same math:

  * ``_local_experts`` -- the contiguous path, whose expert buffer is
    ``sum_e align(count_e, 128)`` rows tall. That height lives on the device, so
    allocating it needs ``num_recv.cpu()``: one device-to-host sync per layer,
    which is what refuses CUDA-graph capture.
  * ``_local_experts_masked`` -- the same stages over a fixed ``[E, max_m, ...]``
    buffer, with the per-expert row counts staying on the device as the masked
    GEMM's ``masked_m``. Static shapes, no sync, capturable.

The masked path is only correct if it computes the same numbers, and it should do
so *exactly*: it does not reorder any accumulation, it only stops computing rows
that the first GEMM never writes and nothing downstream reads. The comparison here
is therefore bit-equality, which is a much stronger regression guard than a
tolerance -- a tolerance would hide a mis-sized buffer that happens to land close.
The fused SwiGLU kernel is pinned off while doing so, because it reaches the two
paths in different commits and differs from the sequence it replaces by one fp8
ULP; a second case runs with it on and checks agreement to fp8 rounding instead.

``max_m`` is deliberately a multiple of 128 (the strategy uses
``align(N, 128)``). ``ep_scatter`` tiles ``m_indices`` in ``BLOCK_E = 128`` row
blocks and writes past the end of the buffer given shorter uniform segments; at
``max_m = 32`` the relative error against the contiguous path was 0.44-0.50 over
four repeats, and the first call in a fresh process still looked correct because
the stomped page was untouched.

The token counts therefore span more than one ``align(N, 128)`` step: 4/24/96 all
land on a single 128-row block, while 128/129/256 and ``_MASKED_MAX_N`` put each
expert segment across several ``BLOCK_E`` blocks and switch ``t_rows`` between the
two arms of ``min(max_m, align(N, _MASKED_TILE))`` -- which is where a stride or
offset regression in the row slicing would show up, and where ``output_index``
stops being a no-op remap.

Needs a Hopper GPU and a DeepGEMM carrying the SM90 FP8 grouped kernels. Absent
CUDA it skips; with CUDA present but the kernels missing it fails, since on the H20
lane that is a broken build rather than a missing capability.
"""

import os
import sys
import unittest
from contextlib import contextmanager

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    GroupedFP8Strategy,
    MoeCfg,
    _has_grouped_fp8_kernel,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import _MASKED_MAX_N
from rtp_llm.utils.model_weight import W

# Small but shape-legal: dim and moe_inter_dim must be multiples of the FP8 block
# (128), and the routed-expert tensors keep V4's [E, N, K] / [E, K, N] layout.
_E = 8
_D = 1024
_INTER = 512
_TOPK = 2
_SWIGLU_LIMIT = 10.0
_FP8_BLOCK = 128


@contextmanager
def _env(**kw):
    """Temporarily set env vars; ``None`` pops the var."""
    saved = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cfg() -> MoeCfg:
    return MoeCfg(
        layer_id=0,
        dim=_D,
        moe_inter_dim=_INTER,
        n_routed_experts=_E,
        n_activated_experts=_TOPK,
        swiglu_limit=_SWIGLU_LIMIT,
        ep_size=1,
        ep_rank=0,
        n_local_experts=_E,
        local_expert_start=0,
        local_expert_end=_E,
        max_tokens_per_rank=128,
    )


def _make_weights(device: str, seed: int = 20260814) -> dict:
    """Routed-expert weights in the checkpoint's FP8 block-quantised layout."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def fp8(*shape):
        t = (torch.randn(*shape, generator=g) * 0.5).clamp(-4, 4)
        return t.to(device).to(torch.float8_e4m3fn)

    def scale(*shape):
        # Constant, positive, and a power of two so the dequantised values stay
        # representable; the test is about layout and row counts, not rounding.
        return torch.full(shape, 0.0625, dtype=torch.float32, device=device)

    return {
        W.v4_routed_w1_w: fp8(_E, _INTER, _D),
        W.v4_routed_w3_w: fp8(_E, _INTER, _D),
        W.v4_routed_w2_w: fp8(_E, _D, _INTER),
        W.v4_routed_w1_s: scale(_E, _INTER // _FP8_BLOCK, _D // _FP8_BLOCK),
        W.v4_routed_w3_s: scale(_E, _INTER // _FP8_BLOCK, _D // _FP8_BLOCK),
        W.v4_routed_w2_s: scale(_E, _D // _FP8_BLOCK, _INTER // _FP8_BLOCK),
    }


def _inputs(n_tokens: int, device: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(n_tokens, _D, generator=g) * 0.5).to(device).to(torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(n_tokens)]
    ).to(device).to(torch.int64)
    w = torch.rand(n_tokens, _TOPK, generator=g).to(device).to(torch.float32)
    return x, w, idx


class GroupedFP8MaskedEquivalenceTest(unittest.TestCase):
    """The capturable masked path must be bit-identical to the contiguous one."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA")
        # Past this point the box has a GPU, so an absent kernel is a build
        # problem. Skipping here is what made this target able to report PASSED
        # with zero assertions.
        if not has_deep_gemm():
            raise AssertionError(
                "CUDA is present but deep_gemm is not importable; this lane is "
                "expected to carry it"
            )
        if not _has_grouped_fp8_kernel():
            raise AssertionError(
                "CUDA is present but the SM90 FP8 grouped kernel probe failed "
                "(needs Hopper and a DeepGEMM 2.x carrying "
                "m_grouped_fp8_gemm_nt_contiguous/_masked)"
            )
        cls.device = "cuda:0"
        cls.strategy = GroupedFP8Strategy(_cfg())
        cls.strategy.setup_weights(_make_weights(cls.device))

    def test_masked_matches_contiguous_bitwise(self):
        """Layout only: the fused SwiGLU is pinned off so it is not a variable.

        The fused kernel differs from the sequence it replaces by one fp8 e4m3 ULP
        on a fraction of quantised activations, and it did not land on both paths
        in the same commit, so leaving it on would make this comparison depend on
        which of the two paths had been converted.
        """
        for n_tokens in (4, 24, 96, 128, 129, 256, 512):
            with self.subTest(n_tokens=n_tokens), _env(DSV4_MOE_FUSED_SWIGLU="0"):
                x, w, idx = _inputs(n_tokens, self.device, seed=1000 + n_tokens)
                y_contig = self.strategy._local_experts(x, w, idx, 0)
                y_masked = self.strategy._local_experts_masked(x, w, idx, 0)
                self.assertEqual(tuple(y_masked.shape), (n_tokens, _D))
                self.assertEqual(y_masked.dtype, y_contig.dtype)
                self.assertTrue(
                    torch.equal(y_contig, y_masked),
                    "masked path diverged from contiguous at "
                    f"n_tokens={n_tokens}: max|diff|="
                    f"{(y_contig.float() - y_masked.float()).abs().max().item()}",
                )

    def test_masked_at_the_cap_agrees_within_fp8_granularity(self):
        """At ``_MASKED_MAX_N`` the two layouts stop being bit-identical.

        Measured: bit-equality holds through 512 tokens and fails at 1024, where
        6.3% of elements differ, ``max |diff|`` is 3.4e-03 and the relative
        Frobenius difference is 1.6e-03. The layouts hand DeepGEMM different M
        shapes -- ``[E, max_m]`` with a device-side count versus
        ``sum_e align(count_e, 128)`` rows -- and past some size it selects a
        different internal blocking, which reorders the fp32 accumulation.

        1.6e-03 is the granularity of the format the GEMM operands are in: one
        e4m3 mantissa step is 2**-9 = 2.0e-03. So the two paths agree to within
        the precision they compute in, which is the strongest claim available
        here, while a stride or row-count regression would be orders of magnitude
        larger -- the max_m=32 misalignment this file's header describes produced
        0.44-0.50.

        512 is above every decode shape this deployment reaches (N is
        ``batch * ep`` post-all-gather), so the bit-exact range covers production
        and 1024 is the refusal cap rather than a size decode runs at.
        """
        with _env(DSV4_MOE_FUSED_SWIGLU="0"):
            x, w, idx = _inputs(_MASKED_MAX_N, self.device, seed=2048)
            y_contig = self.strategy._local_experts(x, w, idx, 0).float()
            y_masked = self.strategy._local_experts_masked(x, w, idx, 0).float()
            self.assertEqual(tuple(y_masked.shape), (_MASKED_MAX_N, _D))
            rel_fro = (
                (y_contig - y_masked).norm() / y_contig.norm().clamp(min=1e-9)
            ).item()
            self.assertLess(
                rel_fro,
                5e-3,
                f"rel_fro={rel_fro:.3e} at n_tokens={_MASKED_MAX_N}, "
                f"max|diff|={(y_contig - y_masked).abs().max().item():.3e}",
            )

    def test_masked_path_is_capturable_and_replays(self):
        """The reason the masked layout exists: it can go inside a CUDA graph.

        Everything else in this file compares numbers; nothing checked the
        capturability that motivates the layout. Three facts land here at once and
        nowhere else:

        * the path performs no device->host sync -- a ``num_recv.cpu()`` anywhere
          in it makes ``torch.cuda.graph`` raise, which is exactly why the
          contiguous path cannot be captured;
        * ``max_m = align(N, 128)`` is a host-side constant under capture, so every
          buffer has a static shape;
        * replay reproduces the eager result, i.e. the captured graph reads the
          routing from ``masked_m`` on the device rather than from anything baked in
          at capture time.

        One eager call first, for the same reason the engine runs an eager forward
        before ``captureDecode()``: DeepGEMM's JIT cannot compile inside a capture.

        ``ep_size`` is 1 here (``_cfg()``), so no EP collective is captured -- that
        combination is what the deployment constraint in ``_assert_one_captured_size``
        is about and is left to the multi-card smoke.
        """
        n_tokens = 24
        with _env(DSV4_MOE_FUSED_SWIGLU="0"):
            x, w, idx = _inputs(n_tokens, self.device, seed=31337)

            # Compile and warm outside the capture.
            eager = self.strategy._local_experts_masked(x, w, idx, 0).clone()
            torch.cuda.synchronize()

            out = torch.empty_like(eager)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(self.strategy._local_experts_masked(x, w, idx, 0))

            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(out, eager),
                "replay diverged from eager: max|diff|="
                f"{(out.float() - eager.float()).abs().max().item()}",
            )

            # Replay again after changing the input in place: the graph must read
            # the new values rather than a capture-time copy.
            x.mul_(2.0)
            expected = self.strategy._local_experts_masked(x, w, idx, 0).clone()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(out, expected),
                "replay after mutating the input did not follow it: max|diff|="
                f"{(out.float() - expected.float()).abs().max().item()}",
            )

    def test_masked_matches_contiguous_with_fused_swiglu(self):
        """With the fused kernel on, the two paths agree to fp8 rounding.

        Bit-equality is not claimed here: whichever of the two paths is running the
        fused clamp/SiLU/mul/quant kernel differs from the explicit sequence by at
        most one fp8 e4m3 ULP per quantised element. The bound below is far tighter
        than a real layout or row-count bug could sneak under.
        """
        with _env(DSV4_MOE_FUSED_SWIGLU="1"):
            x, w, idx = _inputs(24, self.device, seed=11)
            y_contig = self.strategy._local_experts(x, w, idx, 0).float()
            y_masked = self.strategy._local_experts_masked(x, w, idx, 0).float()
        denom = y_contig.norm().clamp(min=1e-6)
        rel_fro = ((y_contig - y_masked).norm() / denom).item()
        self.assertLess(rel_fro, 1e-2, f"rel_fro={rel_fro}")

    def test_masked_is_repeatable(self):
        """Same input twice must give the same bytes.

        The masked path leaves rows above the live count untouched rather than
        zeroing the whole buffer, so a stale-read bug would show up here as
        run-to-run variation while a single comparison against the contiguous
        path could still pass.
        """
        x, w, idx = _inputs(24, self.device, seed=7)
        first = self.strategy._local_experts_masked(x, w, idx, 0).clone()
        second = self.strategy._local_experts_masked(x, w, idx, 0)
        self.assertTrue(torch.equal(first, second))



if __name__ == "__main__":
    unittest.main()
