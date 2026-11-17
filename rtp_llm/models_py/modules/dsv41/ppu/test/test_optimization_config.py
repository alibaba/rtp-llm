"""Compatibility and precedence contracts for shared operator switches."""

import os
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    ALIASES,
    DEFAULT_ENABLED,
    VALUE_ALIASES,
    enabled,
    env_value,
    graph_mhc_options,
    value,
)


class OptimizationConfigTest(unittest.TestCase):
    def test_unset_uses_qualified_defaults_and_retains_other_callers_defaults(self):
        for alias in ALIASES:
            for default in (False, True):
                with self.subTest(alias=alias, default=default):
                    self.assertEqual(
                        enabled(alias, default=default, env={}),
                        alias in DEFAULT_ENABLED or default,
                    )

    def test_qualified_defaults_support_explicit_opt_out(self):
        for name in DEFAULT_ENABLED:
            with self.subTest(name=name):
                self.assertTrue(enabled(name, env={}))
                self.assertEqual(env_value(name, env={}), "1")
                self.assertFalse(enabled(name, default=True, env={name: "0"}))
                self.assertEqual(env_value(name, env={name: "0"}), "0")
                self.assertTrue(enabled(name, env={name: "1"}))

    def test_workload_precision_and_diagnostic_flags_keep_their_defaults(self):
        from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import enabled as phase_enabled

        names = (
            "FAKE_BALANCE_EXPERT", "RTP_V41_PPU_EPLB", "RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN",
            "RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION", "RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS",
            "RTP_V41_PPU_QKV_STREAMS", "RTP_V41_PPU_PACKED_ATTENTION",
            "RTP_V41_PPU_DRAFT_LM_HEAD_BF16", "RTP_V41_PPU_MLA_PACKED_VERIFY",
        )
        with patch.dict(os.environ, {}, clear=True):
            before = dict(os.environ)
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(env_value(name), "0")
                    self.assertFalse(phase_enabled(name))
                    self.assertTrue(phase_enabled(name, default=True))
            self.assertEqual(dict(os.environ), before)

    def test_each_canonical_switch_controls_all_its_aliases(self):
        for alias, canonical in ALIASES.items():
            for value in ("0", "1"):
                with self.subTest(alias=alias, value=value):
                    self.assertEqual(
                        enabled(alias, env={canonical: value}), value == "1"
                    )
                    self.assertEqual(
                        enabled(
                            alias, env={canonical: value, alias: str(1 - int(value))}
                        ),
                        value == "1",
                    )

    def test_legacy_flags_remain_independent_across_phases(self):
        for alias, canonical in ALIASES.items():
            for other in ALIASES:
                if other == alias:
                    continue
                with self.subTest(alias=alias, other=other):
                    self.assertEqual(
                        enabled(other, env={alias: "1"}), enabled(other, env={})
                    )
                    self.assertTrue(enabled(other, default=True, env={alias: "0"}))
            self.assertEqual(enabled(canonical, env={alias: "1"}), enabled(canonical, env={}))

    def test_legacy_parsers_are_preserved(self):
        for value in ("true", "yes", "on", " 1 ", "garbage", ""):
            for alias in ALIASES:
                with self.subTest(alias=alias, value=value):
                    if alias in (
                        "RTP_V41_PPU_GRAPH_BATCHED_WO_A",
                        "RTP_V41_PPU_PROPOSAL_BATCHED_WO_A",
                    ):
                        if value in ("garbage", ""):
                            with self.assertRaises(ValueError):
                                enabled(alias, env={alias: value})
                        else:
                            self.assertTrue(enabled(alias, env={alias: value}))
                    elif alias in (
                        "RTP_V41_PPU_CP8_GROUPED_QO_ROPE",
                        "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE",
                        "RTP_V41_PPU_CP8_WO_A_BMM",
                    ):
                        with self.assertRaises(ValueError):
                            enabled(alias, env={alias: value})
                    else:
                        self.assertFalse(enabled(alias, env={alias: value}))

    def test_canonical_validation_precedes_legacy_parsing(self):
        for alias, canonical in ALIASES.items():
            self.assertTrue(enabled(alias, env={canonical: "1", alias: "invalid"}))
            for value in ("", "true", "yes", " 1 ", "2"):
                with self.subTest(canonical=canonical, value=value):
                    with self.assertRaisesRegex(ValueError, canonical):
                        enabled(alias, env={canonical: value, alias: "1"})

    def test_process_environment_is_read_without_mutation_or_caching(self):
        alias, canonical = next(iter(ALIASES.items()))
        with patch.dict(os.environ, {canonical: "1"}, clear=True):
            before = dict(os.environ)
            self.assertTrue(enabled(alias))
            self.assertEqual(dict(os.environ), before)
            os.environ[canonical] = "0"
            self.assertFalse(enabled(alias, default=True))

    def test_unknown_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            enabled("RTP_V41_PPU_TYPO", env={})

    def test_mode_and_tuning_aliases(self):
        for alias, canonical in VALUE_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(value(alias, default="dot", env={}), "dot")
                self.assertEqual(value(alias, default="1", env={alias: "off"}), "off")
                self.assertEqual(
                    value(
                        alias, default="dot", env={alias: "off", canonical: "epilogue"}
                    ),
                    "epilogue",
                )
                for other in VALUE_ALIASES:
                    if other != alias:
                        self.assertEqual(
                            value(other, default="default", env={alias: "off"}),
                            "default",
                        )
                # Parsing remains the consumer's responsibility; unsupported
                # settings must reach its existing validation unchanged.
                self.assertEqual(
                    value(alias, default="1", env={canonical: "invalid"}), "invalid"
                )
        with self.assertRaisesRegex(ValueError, "unknown"):
            value("RTP_V41_PPU_TYPO", default="1", env={})

    def test_graph_mhc_defaults_and_phase_independence(self):
        for phase in ("VERIFY", "PROPOSAL"):
            self.assertEqual(
                graph_mhc_options(phase, reduce_block_m=True, env={}), ("dot", 1)
            )
            self.assertEqual(
                graph_mhc_options(phase, reduce_block_m=False, env={}), ("dot", None)
            )
            other = "VERIFY" if phase == "PROPOSAL" else "PROPOSAL"
            self.assertEqual(
                graph_mhc_options(
                    phase,
                    reduce_block_m=True,
                    env={f"RTP_V41_PPU_{other}_MHC_POST": "off"},
                ),
                ("dot", 1),
            )

    def test_graph_mhc_modes_validation_and_disabled_scopes(self):
        for phase in ("VERIFY", "PROPOSAL"):
            for post in ("off", "dot", "epilogue"):
                for rows in ("0", "1", "2", "4", "8"):
                    env = {
                        "RTP_V41_PPU_MHC_POST": post,
                        "RTP_V41_PPU_MHC_REDUCE_BLOCK_M": rows,
                        f"RTP_V41_PPU_{phase}_MHC_POST": "invalid",
                    }
                    self.assertEqual(
                        graph_mhc_options(phase, reduce_block_m=True, env=env),
                        (
                            None if post == "off" else post,
                            None if rows == "0" else int(rows),
                        ),
                    )
            for setting in ("POST", "REDUCE_BLOCK_M"):
                env = {f"RTP_V41_PPU_MHC_{setting}": "invalid"}
                with self.assertRaises(ValueError):
                    graph_mhc_options(phase, reduce_block_m=True, env=env)
                self.assertEqual(
                    graph_mhc_options(None, reduce_block_m=True, env=env), (None, None)
                )
            self.assertEqual(
                graph_mhc_options(
                    phase,
                    reduce_block_m=False,
                    env={"RTP_V41_PPU_MHC_REDUCE_BLOCK_M": "invalid"},
                ),
                ("dot", None),
            )


if __name__ == "__main__":
    unittest.main()
