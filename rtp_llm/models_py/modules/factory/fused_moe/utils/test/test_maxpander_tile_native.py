import inspect
import unittest
from types import SimpleNamespace

import deep_gemm

from rtp_llm.models_py.modules.maxpander import (
    MaxPanderMegaMoeTilePolicy,
    _supports_mega_moe_tile_override,
)


class MaxPanderNativeTileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _supports_mega_moe_tile_override(
            getattr(deep_gemm, "fp8_fp4_mega_moe", None),
            getattr(deep_gemm, "get_block_m_for_mega_moe", None),
        ):
            raise unittest.SkipTest(
                "DeepGEMM does not provide the optional block_m API"
            )

    def test_explicit_api_is_available_and_defaults_to_native(self):
        parameter = inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters["block_m"]
        self.assertEqual(parameter.default, 0)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_default_and_explicit_calls_do_not_leak_state(self):
        for tokens, expected in (
            (32, 16),
            (256, 32),
            (512, 64),
            (1024, 64),
            (2048, 128),
            (4096, 240),
        ):
            with self.subTest(tokens=tokens):
                arguments = (4, 480, 5760, tokens, 6, "fp8xfp4")
                self.assertEqual(
                    deep_gemm.get_block_m_for_mega_moe(*arguments), expected
                )
                self.assertEqual(
                    deep_gemm.get_block_m_for_mega_moe(*arguments, block_m=128), 128
                )
                self.assertEqual(
                    deep_gemm.get_block_m_for_mega_moe(*arguments, block_m=64), 64
                )
                self.assertEqual(
                    deep_gemm.get_block_m_for_mega_moe(*arguments), expected
                )

    def test_invalid_tiles_are_rejected(self):
        for tile in (-1, 48, 80, 256):
            with self.subTest(tile=tile), self.assertRaises(RuntimeError):
                deep_gemm.get_block_m_for_mega_moe(
                    4, 480, 5760, 1024, 6, "fp8xfp4", block_m=tile
                )

    def test_native_candidate_set_is_preserved(self):
        for tile in (8, 16, 32, 64, 96, 128, 192, 240):
            with self.subTest(tile=tile):
                self.assertEqual(
                    deep_gemm.get_block_m_for_mega_moe(
                        4, 480, 5760, 1024, 6, "fp8xfp4", block_m=tile
                    ),
                    tile,
                )

    def test_v41_prefill_grid_matches_off_at_every_length(self):
        config = SimpleNamespace(
            ep_size=4,
            n_routed_experts=384,
            physical_expert_num=480,
            max_tokens_per_rank=5760,
            n_activated_experts=6,
        )
        policy = MaxPanderMegaMoeTilePolicy(
            config, deep_gemm.fp8_fp4_mega_moe, deep_gemm.get_block_m_for_mega_moe
        )
        self.assertEqual(
            [
                policy.block_m(tokens)
                for tokens in (32, 256, 512, 1024, 2048, 3072, 4096)
            ],
            [16, 32, 64, 128, 192, 240, 192],
        )
        for tokens in (32, 256, 512, 1024, 2048, 3072, 4096):
            with self.subTest(tokens=tokens):
                kernel = policy.kernel_for(tokens)
                self.assertIs(kernel.func, deep_gemm.fp8_fp4_mega_moe)
                self.assertEqual(kernel.keywords, {"block_m": policy.block_m(tokens)})

    def test_v4_and_v41_policy_matches_native_geometry(self):
        for experts, physical in ((256, 320), (384, 480), (384, 768)):
            for ranks in (2, 4, 8):
                config = SimpleNamespace(
                    ep_size=ranks,
                    n_routed_experts=experts,
                    physical_expert_num=physical,
                    max_tokens_per_rank=4096,
                    n_activated_experts=6,
                )
                policy = MaxPanderMegaMoeTilePolicy(
                    config,
                    deep_gemm.fp8_fp4_mega_moe,
                    deep_gemm.get_block_m_for_mega_moe,
                )
                for tokens in (0, 1, 32, 255, 256, 511, 512, 1023, 1024, 2048, 4096):
                    with self.subTest(
                        experts=experts, physical=physical, ranks=ranks, tokens=tokens
                    ):
                        logical = deep_gemm.get_block_m_for_mega_moe(
                            ranks, experts, 4096, tokens, 6, "fp8xfp4"
                        )
                        self.assertEqual(policy.block_m(tokens), logical)
                        self.assertEqual(
                            deep_gemm.get_block_m_for_mega_moe(
                                ranks,
                                physical,
                                4096,
                                tokens,
                                6,
                                "fp8xfp4",
                                **policy.kernel_for(tokens).keywords,
                            ),
                            logical,
                        )


if __name__ == "__main__":
    unittest.main()
