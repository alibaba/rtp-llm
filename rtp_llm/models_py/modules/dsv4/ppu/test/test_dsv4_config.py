"""PPU admission gates, with native runtime and provider construction stubbed."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch


class Dsv4ConfigTest(unittest.TestCase):
    def setUp(self):
        name = "rtp_llm.models_py.modules.dsv4.ppu.manifest"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parents[6] / "rtp_llm/models_py/modules/dsv4/ppu/manifest.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.provider = Mock(
            side_effect=lambda options, **kwargs: NS(execution_options=options, **kwargs)
        )
        self.modules = patch.dict(sys.modules, {
            "rtp_llm.ops": NS(KvCacheDataType=NS(FP8="fp8")),
            name.rsplit(".", 1)[0] + ".ppu_module_provider": NS(PpuModuleProvider=self.provider),
            name.rsplit(".", 1)[0] + ".ppu_decode_provider": NS(PpuDecodeProvider=self.provider),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = NS(hidden_size=4096, num_layers=43, model_type="deepseek_v4",
            gen_num_per_cycle=0, lora_infos=None,
            attn_config=NS(kv_cache_dtype="fp8", kernel_tokens_per_block=256, tokens_per_block=256),
            eplb_config=NS(enable_eplb=lambda: False))
        self.parallel = NS(role_type=NS(name="PREFILL"), pp_size=1, tp_size=4,
            dp_size=1, world_size=4, ep_size=1,
            prefill_cp_config=NS(is_enabled=lambda: False),
        )
        self.moe = NS(use_deepep_moe=True, use_deepep_low_latency=False,
                      use_deepep_internode=False, use_all_gather=False)

    def build(self):
        return self.module.build_provider(self.config, self.parallel, self.moe, 128)

    def test_prefill_tp(self):
        self.assertEqual(self.build().execution_options,
                         self.module.PREFILL_EXECUTION_OPTIONS)

    def test_fused_role_is_rejected_before_forward(self):
        self.parallel.role_type.name = "PDFUSION"
        with self.assertRaisesRegex(ValueError, "separate Prefill and Decode"):
            self.build()
        self.provider.assert_not_called()

    def test_decode_target_and_mtp(self):
        self.parallel.role_type.name = "DECODE"
        self.parallel.ep_size = 4
        self.moe.use_deepep_low_latency = True
        for gamma in (0, 1, 2, 3):
            self.config.gen_num_per_cycle = gamma
            self.assertEqual(self.build().execution_options["DSV4_PPU_DECODE_METADATA"],
                             "reference")
        self.config.model_type = "deepseek_v4_mtp"
        self.config.num_layers = 1
        self.config.attn_config.layer_compress_ratios = [0]
        self.build()

    def test_decode_dspark_draft_is_admitted_at_block_five(self):
        self.parallel.role_type.name = "DECODE"
        self.parallel.ep_size = 4
        self.moe.use_deepep_low_latency = True
        self.config.model_type = "deepseek_v4_dspark"
        self.config.num_layers = 3
        self.config.attn_config.layer_compress_ratios = [0, 0, 0]
        self.config.dspark_target_layer_ids = [40, 41, 42]
        self.config.gen_num_per_cycle = 5
        self.assertEqual(self.build().execution_options["DSV4_PPU_DECODE_METADATA"],
                         "reference")
        with patch.object(self.config, "gen_num_per_cycle", 9):
            with self.assertRaisesRegex(ValueError, "speculative depth"):
                self.build()

    def test_dspark_draft_rejects_deep_compressed_ratio(self):
        self.config.model_type = "deepseek_v4_dspark"
        self.config.num_layers = 3
        self.config.attn_config.layer_compress_ratios = [0, 0, 4]
        self.config.dspark_target_layer_ids = [40, 41, 42]
        with self.assertRaisesRegex(ValueError, "DSpARK draft layers"):
            self.build()

    def test_dspark_target_carries_the_same_proposal_width(self):
        # The target verifies gamma+1 rows, so it holds gamma=5 as well and
        # must pass the same gate even though it is a 43-layer V4.
        self.config.dspark_target_layer_ids = [40, 41, 42]
        self.config.gen_num_per_cycle = 5
        self.assertEqual(self.build().execution_options,
                         self.module.PREFILL_EXECUTION_OPTIONS)
        self.config.gen_num_per_cycle = 9
        with self.assertRaisesRegex(ValueError, "speculative depth"):
            self.build()

    def test_decode_ep1_rejected_for_single_and_multiple_tp_ranks(self):
        self.parallel.role_type.name = "DECODE"
        self.moe.use_deepep_low_latency = True
        for tp in (1, 4):
            with self.subTest(tp=tp):
                self.parallel.tp_size = self.parallel.world_size = tp
                with self.assertRaisesRegex(ValueError, "Decode requires EP"):
                    self.build()
        self.provider.assert_not_called()

    def test_invalid_configuration_fails_before_construction(self):
        cases = [(self.config, "hidden_size", 7168), (self.config, "num_layers", 61),
                 (self.config, "gen_num_per_cycle", 4), (self.parallel, "pp_size", 2),
                 (self.parallel, "world_size", 8), (self.parallel, "ep_size", 2),
                 (self.config.attn_config, "kv_cache_dtype", "bf16"),
                 (self.config.attn_config, "kernel_tokens_per_block", 128),
                 (self.config.attn_config, "tokens_per_block", 512),
                 (self.parallel.prefill_cp_config, "is_enabled", lambda: True)]
        for obj, field, value in cases:
            with self.subTest(field=field), patch.object(obj, field, value):
                with self.assertRaises(ValueError):
                    self.build()
        self.provider.assert_not_called()

    def test_rejects_wrong_deepep_role(self):
        self.parallel.ep_size = 4
        self.moe.use_deepep_low_latency = True
        with self.assertRaisesRegex(ValueError, "DeepEP"):
            self.build()

    def test_reuse_requires_explicit_opt_in(self):
        self.module.validate_cache_config(NS(reuse_cache=False))
        with self.assertRaisesRegex(ValueError, "prefix reuse"):
            self.module.validate_cache_config(NS(reuse_cache=True))
        with patch.dict(os.environ, {"DSV4_PPU_PREFIX_REUSE": "1"}):
            self.module.validate_cache_config(NS(reuse_cache=True))
        for value in ("0", "true", ""):
            with patch.dict(os.environ, {"DSV4_PPU_PREFIX_REUSE": value}):
                with self.assertRaisesRegex(ValueError, "prefix reuse"):
                    self.module.validate_cache_config(NS(reuse_cache=True))

    def test_pdl_is_not_an_execution_option(self):
        # PDL is fixed off inside the HC module; a stale env entry is inert
        # instead of contradicting the manifest.
        for options in (self.module.PREFILL_EXECUTION_OPTIONS,
                        self.module.DECODE_EXECUTION_OPTIONS):
            self.assertNotIn("DSV4_MHC_POST_PDL", options)
        os.environ["DSV4_MHC_POST_PDL"] = "1"
        self.build()

    def test_removed_attention_switches_fail_closed(self):
        self.parallel.role_type.name = "DECODE"
        self.parallel.tp_size = 1
        self.parallel.dp_size = self.parallel.world_size = self.parallel.ep_size = 16
        self.moe.use_deepep_low_latency = True
        for removed in ("DSV4_PPU_HCA_SPLIT", "DSV4_PPU_WOA_SMALL_TILE"):
            with self.subTest(removed=removed):
                with patch.dict(os.environ, {removed: "1"}):
                    with self.assertRaisesRegex(ValueError, "was removed"):
                        self.build()
        self.provider.assert_not_called()

    def test_decode_attn_profile_is_validated_against_topology(self):
        self.parallel.role_type.name = "DECODE"
        self.parallel.tp_size = 1
        self.parallel.dp_size = self.parallel.world_size = self.parallel.ep_size = 16
        self.moe.use_deepep_low_latency = True
        with patch.dict(os.environ, {"DSV4_PPU_DECODE_ATTN_PROFILE": "large_ep_full"}):
            provider = self.build()
        self.assertEqual(provider.attn_profile, "large_ep_full")
        self.assertEqual(provider.ep_size, 16)
        self.assertEqual(provider.tp_size, 1)

        self.parallel.ep_size = self.parallel.world_size = 8
        self.parallel.dp_size = 8
        with patch.dict(os.environ, {"DSV4_PPU_DECODE_ATTN_PROFILE": "large_ep_full"}):
            with self.assertRaisesRegex(ValueError, "requires EP16/32 with TP1"):
                self.build()

        self.parallel.ep_size = self.parallel.world_size = 16
        self.parallel.dp_size = 16
        with patch.dict(os.environ, {"DSV4_PPU_DECODE_ATTN_PROFILE": "large_ep"}):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                self.build()


if __name__ == "__main__":
    unittest.main()
