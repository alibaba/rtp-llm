import unittest

import torch

from rtp_llm.models_py.modules.hy_v4.ihc import Hy4IHCHead, Hy4IHCUnit
from rtp_llm.utils.model_weight import W


class Hy4IhcTest(unittest.TestCase):
    def _unit_weights(self, hidden: int, hc: int):
        return {
            W.hy4_ihc_attn_fn: torch.randn(2 * hc, hc * hidden).float(),
            W.hy4_ihc_attn_scale: torch.tensor([0.2, -0.1]).float(),
            W.hy4_ihc_attn_base: torch.randn(2 * hc).float(),
        }

    def test_pre_post_match_fp32_reference(self):
        torch.manual_seed(7)
        hidden, hc = 5, 4
        weights = self._unit_weights(hidden, hc)
        unit = Hy4IHCUnit(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=2,
        )
        channels = torch.randn(7, hc, hidden, dtype=torch.bfloat16)
        read, post = unit.pre(channels)

        flat = channels.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        mixes = torch.nn.functional.linear(flat, weights[W.hy4_ihc_attn_fn]) * rstd
        pre_raw, post_raw = mixes.chunk(2, dim=-1)
        pre_ref = (
            torch.sigmoid(
                pre_raw * weights[W.hy4_ihc_attn_scale][0]
                + weights[W.hy4_ihc_attn_base][:hc]
            )
            + 1e-6
        )
        post_ref = (
            2.0
            * torch.sigmoid(
                post_raw * weights[W.hy4_ihc_attn_scale][1]
                + weights[W.hy4_ihc_attn_base][hc:]
            )
            + 1e-6
        )
        read_ref = (pre_ref.unsqueeze(-1) * channels.float()).sum(1).bfloat16()
        torch.testing.assert_close(read, read_ref)
        torch.testing.assert_close(post, post_ref)

        block = torch.randn(7, hidden, dtype=torch.bfloat16)
        actual = unit.post(block, channels, post)
        expected = (
            channels.float() + post_ref.unsqueeze(-1) * block.float().unsqueeze(1)
        ).bfloat16()
        torch.testing.assert_close(actual, expected)

    def test_head_and_single_stream_expansion(self):
        torch.manual_seed(11)
        hidden, hc = 3, 4
        unit_weights = self._unit_weights(hidden, hc)
        unit = Hy4IHCUnit(
            unit_weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-6,
            kind="attn",
        )
        single = torch.randn(2, hidden)
        expanded = unit.prepare_input(single)
        self.assertEqual(tuple(expanded.shape), (2, hc, hidden))
        for idx in range(hc):
            torch.testing.assert_close(expanded[:, idx], single)

        head_weights = {
            W.hy4_ihc_head_fn: torch.randn(hc, hc * hidden).float(),
            W.hy4_ihc_head_scale: torch.tensor([0.15]).float(),
            W.hy4_ihc_head_base: torch.randn(hc).float(),
        }
        head = Hy4IHCHead(
            head_weights,
            hidden_size=hidden,
            hc_mult=hc,
            hc_eps=1e-6,
            norm_eps=1e-6,
            chunk_size=1,
        )
        actual = head(expanded.bfloat16())
        flat = expanded.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(flat, head_weights[W.hy4_ihc_head_fn])
        gates = (
            torch.sigmoid(
                logits * rstd * head_weights[W.hy4_ihc_head_scale]
                + head_weights[W.hy4_ihc_head_base]
            )
            + 1e-6
        )
        expected = (gates.unsqueeze(-1) * expanded.float()).sum(1).bfloat16()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
