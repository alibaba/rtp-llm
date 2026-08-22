from __future__ import annotations

import unittest

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep import (
    _sm120_uses_replicated_tp_tokens,
)


def _cfg(*, tp_size: int, ep_size: int) -> MoeCfg:
    local = 256 // ep_size
    return MoeCfg(
        layer_id=2,
        dim=4096,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=7.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=local,
        local_expert_start=0,
        local_expert_end=local,
        max_tokens_per_rank=4096,
        tp_size=tp_size,
    )


class Sm120TpEpLayoutTest(unittest.TestCase):
    def test_tp4_ep4_world4_skips_token_gather(self):
        self.assertTrue(
            _sm120_uses_replicated_tp_tokens(_cfg(tp_size=4, ep_size=4), 4)
        )

    def test_cp4_ep4_keeps_token_gather(self):
        # CP reports effective attention tp_size=1.
        self.assertFalse(
            _sm120_uses_replicated_tp_tokens(_cfg(tp_size=1, ep_size=4), 4)
        )

    def test_dp4_ep4_keeps_token_gather(self):
        self.assertFalse(
            _sm120_uses_replicated_tp_tokens(_cfg(tp_size=1, ep_size=4), 4)
        )

    def test_mixed_world_does_not_assume_overlapping_groups(self):
        self.assertFalse(
            _sm120_uses_replicated_tp_tokens(_cfg(tp_size=4, ep_size=4), 8)
        )


if __name__ == "__main__":
    unittest.main()
