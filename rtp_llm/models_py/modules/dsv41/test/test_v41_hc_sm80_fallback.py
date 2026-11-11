"""SM80/ROCm chunked reference path for V4.1 delayed hyper-connections."""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv41.hc.fallback_impl import _hc_split_sinkhorn


class V41HCSm80FallbackTest(unittest.TestCase):
    dim = 8
    hc = 4
    iters = 3
    eps = 1e-6

    def _unit(self):
        torch.manual_seed(7)
        fn = torch.randn(24, self.hc * self.dim)
        base = torch.randn(24)
        scale = torch.randn(3)
        return DelayedHCUnit(
            fn,
            base,
            scale,
            dim=self.dim,
            hc_mult=self.hc,
            hc_sinkhorn_iters=self.iters,
            norm_eps=self.eps,
            hc_eps=self.eps,
        )

    def test_chunked_pre_and_post_match_reference(self):
        x = torch.randn(1, 5, self.hc, self.dim)
        previous = self._unit()
        previous.pre_mix_out = torch.randn(1, 5, self.hc)
        unit = self._unit()
        unit.set_previous(previous)
        with mock.patch.dict(
            os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "1000"}
        ):
            y_full, post_full, comb_full = unit.pre(x)
        with mock.patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "2"}):
            y_chunk, post_chunk, comb_chunk = unit.pre(x)
        torch.testing.assert_close(y_chunk, y_full)
        torch.testing.assert_close(post_chunk, post_full)
        torch.testing.assert_close(comb_chunk, comb_full)

        flat = x.reshape(-1, self.hc * self.dim).float()
        mixes = F.linear(flat, unit.fn.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.eps
        )
        pre, post, comb = _hc_split_sinkhorn(
            mixes, unit.scale, unit.base, self.hc, self.iters, self.eps
        )
        want_y = (
            x.reshape(-1, self.hc, self.dim).float()
            * previous.pre_mix_out.reshape(-1, self.hc).unsqueeze(-1)
        ).sum(-2)
        torch.testing.assert_close(y_full.reshape(-1, self.dim), want_y)
        torch.testing.assert_close(post_full.reshape(-1, self.hc), post)
        torch.testing.assert_close(comb_full.reshape(-1, self.hc, self.hc), comb)

        residual = torch.randn(1, 5, self.hc, self.dim)
        out = unit.post(y_full, residual.clone(), post_full, comb_full)
        want = post_full.float() * y_full.float().unsqueeze(-2) + torch.matmul(
            comb_full.float().transpose(-1, -2), residual.float()
        )
        torch.testing.assert_close(out.float(), want)

    def test_no_predecessor_reads_lane_zero(self):
        unit = self._unit()
        x = torch.randn(1, 5, self.hc, self.dim)
        with mock.patch.dict(os.environ, {"DSV4_HC_FALLBACK_CHUNK_TOKENS": "2"}):
            y, _, _ = unit.pre(x)
        torch.testing.assert_close(y, x[..., 0, :])


if __name__ == "__main__":
    unittest.main()
