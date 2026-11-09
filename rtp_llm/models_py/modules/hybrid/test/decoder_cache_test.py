import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.model_desc.decoder_cache import (
    decoder_cache_forward,
    should_drain_mla_before_moe,
    wait_for_mla_prefetch,
)


class TransferFailure(RuntimeError):
    pass


class RecordingCache:
    def __init__(self, events, fail_begin=False):
        self.events = events
        self.active = False
        self.fail_begin = fail_begin

    def begin_forward(self, fmha):
        self.active = True
        self.events.append("begin")
        if self.fail_begin:
            raise TransferFailure("prefetch submission failed")

    def end_forward(self, *, failed):
        self.events.append(("end", failed))
        self.active = False

    def wait_prefetch_complete(self):
        self.events.append("wait")


class DecoderCacheTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cache = RecordingCache(self.events)
        self.fmha = SimpleNamespace()
        self.model = SimpleNamespace(
            layered_prefill_cache=self.cache,
            pinned_mla_groups={3: (self.cache, 1)},
            prepare_fmha_impl=lambda inputs: self.fmha,
        )

    def test_success_binds_model_caches_before_body_and_releases_transfers(self):
        with decoder_cache_forward(self.model, None) as fmha:
            self.assertIs(fmha, self.fmha)
            self.assertIs(fmha.pinned_mla_groups, self.model.pinned_mla_groups)
            self.assertIs(fmha.layered_prefill_cache, self.cache)
            self.events.append("compute")
        self.assertEqual(self.events, ["begin", "compute", ("end", False)])
        self.assertFalse(self.cache.active)

    def test_body_failure_is_propagated_after_failed_cleanup(self):
        with self.assertRaisesRegex(TransferFailure, "decoder failed"):
            with decoder_cache_forward(self.model, None):
                raise TransferFailure("decoder failed")
        self.assertEqual(self.events, ["begin", ("end", True)])
        self.assertFalse(self.cache.active)

    def test_partial_begin_failure_is_cleaned_up(self):
        self.cache.fail_begin = True
        with self.assertRaisesRegex(TransferFailure, "prefetch submission failed"):
            with decoder_cache_forward(self.model, None):
                self.fail("decoder must not run after submission failure")
        self.assertEqual(self.events, ["begin", ("end", True)])
        self.assertFalse(self.cache.active)

    def test_supplied_graph_impl_rebinds_current_model_working_set(self):
        graph_impl = SimpleNamespace(pinned_mla_groups={})
        self.model.prepare_fmha_impl = lambda inputs: self.fail("unexpected prepare")
        with decoder_cache_forward(self.model, None, graph_impl) as fmha:
            self.assertIs(fmha, graph_impl)
            self.assertIs(fmha.pinned_mla_groups, self.model.pinned_mla_groups)

    def test_no_layered_cache_keeps_pinned_cache_binding(self):
        self.model.layered_prefill_cache = None
        with decoder_cache_forward(self.model, None) as fmha:
            self.assertIs(fmha.pinned_mla_groups, self.model.pinned_mla_groups)
            self.assertIsNone(fmha.layered_prefill_cache)
        self.assertEqual(self.events, [])

    def test_diagnostics_finish_after_cache_release(self):
        self.model._prefill_cache_diagnostics = SimpleNamespace(
            finish=lambda cache, failed: self.events.append(("diagnostics", failed))
        )
        with decoder_cache_forward(self.model, None):
            pass
        self.assertEqual(
            self.events, ["begin", ("end", False), ("diagnostics", False)]
        )

    def test_wrapped_and_direct_impl_wait_for_nonzero_group_offset(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                self.events.clear()
                fmha = SimpleNamespace(
                    layered_prefill_cache=self.cache,
                    pinned_mla_groups=self.model.pinned_mla_groups,
                )
                if wrapped:
                    fmha = SimpleNamespace(fmha_impl=fmha)
                wait_for_mla_prefetch(fmha, 3)
                self.assertEqual(self.events, ["wait", "wait"])

    def test_absent_cache_does_not_wait(self):
        wait_for_mla_prefetch(SimpleNamespace(), 3)
        self.assertEqual(self.events, [])

    def test_drain_policy_keeps_existing_switch_and_moe_strategy_contract(self):
        config = SimpleNamespace(moe_layer_index=[3])
        for drain, cross_layer, strategy, layer, expected in (
            ("0", "0", "mega_moe", 3, False),
            ("1", "0", "mega_moe_fp8", 3, True),
            ("0", "1", "mega_moe", 3, True),
            ("1", "1", "mega_moe", 2, False),
            ("1", "1", "other_moe", 3, False),
        ):
            with self.subTest(case=(drain, cross_layer, strategy, layer)), patch.dict(
                os.environ,
                {
                    "RTP_LLM_DSA_MLA_DRAIN_BEFORE_MOE": drain,
                    "RTP_LLM_CP_KV_CROSS_LAYER": cross_layer,
                },
            ):
                self.assertEqual(
                    should_drain_mla_before_moe(
                        config, SimpleNamespace(moe_strategy=strategy), layer
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
