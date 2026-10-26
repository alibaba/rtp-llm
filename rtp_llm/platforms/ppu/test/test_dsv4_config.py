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
        name = "rtp_llm.platforms.ppu.models.dsv4.manifest"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parents[1] / "models/dsv4/manifest.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.provider = Mock(side_effect=lambda options: NS(execution_options=options))
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
            attn_config=NS(kv_cache_dtype="fp8", kernel_tokens_per_block=256),
            eplb_config=NS(enable_eplb=lambda: False))
        self.parallel = NS(role_type=NS(name="PREFILL"), pp_size=1, tp_size=4,
            dp_size=1, world_size=4, ep_size=1,
            prefill_cp_config=NS(is_enabled=lambda: False),
            ffn_disaggregate_config=NS(enable_ffn_disaggregate=False))
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

    def test_invalid_configuration_fails_before_construction(self):
        cases = [(self.config, "hidden_size", 7168), (self.config, "num_layers", 61),
                 (self.config, "gen_num_per_cycle", 4), (self.parallel, "pp_size", 2),
                 (self.parallel, "world_size", 8), (self.parallel, "ep_size", 2),
                 (self.config.attn_config, "kv_cache_dtype", "bf16"),
                 (self.config.attn_config, "kernel_tokens_per_block", 128),
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

    def test_reuse_is_explicitly_rejected(self):
        self.module.validate_cache_config(NS(reuse_cache=False))
        with self.assertRaisesRegex(ValueError, "prefix reuse"):
            self.module.validate_cache_config(NS(reuse_cache=True))

    def test_unsupported_execution_option(self):
        os.environ["DSV4_MHC_POST_PDL"] = "1"
        with self.assertRaisesRegex(ValueError, "DSV4_MHC_POST_PDL=0"):
            self.build()
        self.provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
