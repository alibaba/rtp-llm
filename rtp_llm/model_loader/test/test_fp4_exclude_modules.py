import unittest
from unittest.mock import Mock

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.mixed_fp4_quant_weight import MixedFp4Weight
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class Fp4ExcludeModulesTest(unittest.TestCase):
    wrapper_cases = (
        (PerGroupFp4Weight, False),
        (MixedFp4Weight, True),
    )
    checkpoint_names = (
        "model.layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight",
    )

    def _config(self, mixed_attention, exclude_modules):
        config = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=True,
            mixed_attention=mixed_attention,
        )
        config.exclude_modules = set(exclude_modules)
        return config

    def _weight(self):
        return FfnAtomicWeight(
            name=W.ffn_w13,
            weights=[CkptWeightInfo(name) for name in self.checkpoint_names],
            config=FfnConfig(align_size=16),
        )

    def test_no_match_keeps_original_fp4_wrapper(self):
        for wrapper, mixed_attention in self.wrapper_cases:
            for exclude_modules in (set(), {"cross_attn"}):
                with self.subTest(
                    wrapper=wrapper.__name__, exclude_modules=exclude_modules
                ):
                    config = self._config(mixed_attention, exclude_modules)
                    weight = self._weight()

                    self.assertTrue(wrapper.support(config, weight))
                    self.assertIsInstance(WeightModule.create(weight, config), wrapper)

    def test_all_substring_matches_fall_back_to_atomic_weight(self):
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {"mlp"})
                weight = self._weight()

                self.assertFalse(wrapper.support(config, weight))
                self.assertIs(WeightModule.create(weight, config), weight)

    def test_all_regex_matches_fall_back_to_atomic_weight(self):
        pattern = r"re:\.mlp\.(gate|up)_proj\.weight$"
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {pattern})
                weight = self._weight()

                self.assertFalse(wrapper.support(config, weight))

    def test_modelopt_glob_matches_parameter_suffixes(self):
        pattern = "model.layers.*.mlp"
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {pattern})
                weight = self._weight()

                self.assertFalse(wrapper.support(config, weight))

    def test_multiple_globs_continue_after_non_match(self):
        patterns = (
            "model.layers.*.attention",
            "model.layers.*.mlp",
        )
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, patterns)
                config.exclude_modules = patterns
                weight = self._weight()

                self.assertFalse(wrapper.support(config, weight))

    def test_modelopt_glob_partial_fused_match_is_rejected(self):
        pattern = "model.layers.*.gate_proj"
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {pattern})
                with self.assertRaisesRegex(
                    ValueError, r"Cannot partially exclude fused FP4 weight"
                ):
                    wrapper.support(config, self._weight())

    def test_partial_match_raises_descriptive_error(self):
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {"gate_proj"})

                with self.assertRaisesRegex(
                    ValueError,
                    r"Cannot partially exclude fused FP4 weight.*"
                    r"excluded=.*gate_proj.*quantized=.*up_proj",
                ):
                    wrapper.support(config, self._weight())

    def test_public_matcher_with_empty_exclude_modules(self):
        for wrapper, mixed_attention in self.wrapper_cases:
            for matches in ((False, False), (True, True), (True, False)):
                with self.subTest(wrapper=wrapper.__name__, matches=matches):
                    config = self._config(mixed_attention, set())
                    config.is_module_excluded = Mock(side_effect=matches)
                    weight = self._weight()

                    if any(matches) and not all(matches):
                        with self.assertRaisesRegex(
                            ValueError,
                            r"Cannot partially exclude fused FP4 weight.*"
                            r"excluded=.*gate_proj.*quantized=.*up_proj",
                        ):
                            wrapper.support(config, weight)
                    else:
                        self.assertEqual(
                            wrapper.support(config, weight), not all(matches)
                        )
                    self.assertEqual(
                        [
                            call.args[0]
                            for call in config.is_module_excluded.call_args_list
                        ],
                        list(self.checkpoint_names),
                    )

    def test_public_matcher_is_preferred_when_available(self):
        for wrapper, mixed_attention in self.wrapper_cases:
            with self.subTest(wrapper=wrapper.__name__):
                config = self._config(mixed_attention, {"mlp"})
                config.is_module_excluded = Mock(return_value=False)

                self.assertTrue(wrapper.support(config, self._weight()))
                self.assertEqual(
                    [call.args[0] for call in config.is_module_excluded.call_args_list],
                    list(self.checkpoint_names),
                )


if __name__ == "__main__":
    unittest.main()
