"""Operator qualification stays independent of YOCO scheduling profiles."""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import layers
from rtp_llm.models_py.modules.dsv41.ppu.block import _local_prefill_mega_mhc_eligible
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

SMALL = "RTP_V41_PPU_CP8_YOCO_SMALL_OPS"  # Retired, deliberately has no effect.
LOCAL = "RTP_V41_PPU_LOCAL_PREFILL_MEGA_MHC"  # Retired.
FAST = "RTP_V41_PPU_CP8_YOCO_FAST_PREFILL"  # Retained framework profile.


def cp8_state(*, mode="compact", compact=False):
    cp = NS(
        cp_size=8,
        kv_cache_sharded=True,
        gather_restore_positions=torch.tensor([0, 2]) if compact else None,
        global_positions=torch.arange(6),
    )
    return NS(
        prefill=True,
        idle=False,
        context=cp,
        runtime=NS(policy=SharedPrefillPolicy(mode=mode)),
        use_compact_cp8_ops=lambda context: compact and context is cp,
        lengths=(3, 3),
        query_prefixes=(7, 7),
        _restored_rows=None,
    )


class YOCOOperatorEligibility(unittest.TestCase):
    def test_retired_selectors_do_not_disable_operators_or_enable_framework(self):
        for compact in (False, True):
            for retired in (None, "0", "1"):
                for fast in (None, "0", "1"):
                    with self.subTest(compact=compact, retired=retired, fast=fast):
                        with patch.dict(os.environ):
                            for key in (SMALL, LOCAL, FAST):
                                os.environ.pop(key, None)
                            if retired is not None:
                                os.environ[SMALL] = os.environ[LOCAL] = retired
                            if fast is not None:
                                os.environ[FAST] = fast
                            state = cp8_state(compact=compact)
                            self.assertTrue(layers.cp8_yoco_operator_eligible(state))
                            self.assertEqual(
                                layers.cp8_yoco_fast_prefill_enabled(state), fast == "1"
                            )

    def test_original_cp8_policy_and_layout_domain(self):
        for mode in ("full", "exact"):
            self.assertFalse(layers.cp8_yoco_operator_eligible(cp8_state(mode=mode)))
        for mutate in (
            lambda s: setattr(s, "prefill", False),
            lambda s: setattr(s, "idle", True),
            lambda s: setattr(s, "context", None),
            lambda s: setattr(s.context, "cp_size", 1),
            lambda s: setattr(s.context, "cp_size", 4),
            lambda s: setattr(s.context, "kv_cache_sharded", False),
            lambda s: setattr(s.context, "gather_restore_positions", torch.tensor([0])),
        ):
            state = cp8_state()
            mutate(state)
            self.assertFalse(layers.cp8_yoco_operator_eligible(state))
            with patch.dict(os.environ, {FAST: "1"}):
                self.assertFalse(layers.cp8_yoco_fast_prefill_enabled(state))

    @torch.inference_mode()
    def test_local_mega_keeps_bounded_policy_and_ignores_retired_selector(self):
        state = NS(
            prefill=True,
            idle=False,
            cache=NS(sharded=False),
            context=None,
            runtime=NS(
                policy=SharedPrefillPolicy(mode="compact"),
                model=NS(_is_speculative=False),
            ),
        )
        for flag in ("0", "1"):
            with patch.dict(os.environ, {LOCAL: flag, FAST: "0", SMALL: "0"}):
                self.assertTrue(_local_prefill_mega_mhc_eligible(state))
                for kwargs in (
                    {"mode": "full"},
                    {"mode": "exact"},
                    {"keep_tokens": 141},
                    {"window": 129},
                    {"max_rollback_tokens": 1},
                    {"encoder_replay": True},
                ):
                    state.runtime.policy = SharedPrefillPolicy(
                        **{"mode": "compact", **kwargs}
                    )
                    self.assertFalse(_local_prefill_mega_mhc_eligible(state), kwargs)
                state.runtime.policy = SharedPrefillPolicy(mode="compact")

    def test_full_and_compact_row_metadata_automatic_with_ragged_fallback(self):
        for compact in (False, True):
            for lengths, prefixes in (((3, 3), (7, 7)), ((2, 4), (7, 11))):
                state = cp8_state(compact=compact)
                state.lengths, state.query_prefixes = lengths, prefixes
                with patch.dict(os.environ, {SMALL: "0", FAST: "0"}):
                    positions, requests, selector = SharedForwardState._build_full_rows(
                        state, state.context, selected_only=compact
                    )
                expected_p = torch.cat(
                    [torch.arange(p, p + n) for p, n in zip(prefixes, lengths)]
                )
                expected_r = torch.repeat_interleave(
                    torch.arange(2), torch.tensor(lengths)
                )
                if compact:
                    selected = state.context.gather_restore_positions
                    expected_p, expected_r = expected_p[selected], expected_r[selected]
                self.assertTrue(torch.equal(positions, expected_p))
                self.assertTrue(torch.equal(requests, expected_r))
                self.assertEqual(selector, slice(None))
                cached = SharedForwardState._build_full_rows(
                    state, state.context, selected_only=compact
                )
                self.assertIs(cached[0], positions)
                self.assertIs(cached[1], requests)


if __name__ == "__main__":
    unittest.main()
