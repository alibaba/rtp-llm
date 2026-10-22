import json
import tempfile
import unittest

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.ops import HiddenStateCaptureDtype, ProfilingDebugLoggingConfig, TaskType


class ModelConfigOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig()
        self.config.num_layers = 8

    def test_capture_defaults_are_disabled(self) -> None:
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [])
        self.assertEqual(
            self.config.hidden_state_capture_dtype, HiddenStateCaptureDtype.BF16
        )

    def test_capture_override_preserves_layer_order(self) -> None:
        self.config.apply_override_args(
            '{"hidden_state_capture_layer_ids":[6,1,4],'
            '"hidden_state_capture_dtype":"fp8_e4m3"}'
        )

        self.assertEqual(self.config.hidden_state_capture_layer_ids, [6, 1, 4])
        self.assertEqual(
            self.config.hidden_state_capture_dtype,
            HiddenStateCaptureDtype.FP8_E4M3,
        )

    def test_empty_layer_list_disables_capture(self) -> None:
        self.config.hidden_state_capture_layer_ids = [1]
        self.config.apply_override_args('{"hidden_state_capture_layer_ids":[]}')
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [])

    def test_zero_layer_config_only_accepts_empty_capture_list(self) -> None:
        self.config.num_layers = 0

        self.config.apply_override_args('{"hidden_state_capture_layer_ids":[]}')
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [])

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.config.apply_override_args('{"hidden_state_capture_layer_ids":[0]}')

    def test_build_model_config_layer_override_precedes_capture_validation(self) -> None:
        # An empty checkpoint directory keeps precision initialization real,
        # without requiring model weights or a quantization config.
        with tempfile.TemporaryDirectory() as ckpt_path:
            for layer_id in (3, 4):
                with self.subTest(layer_id=layer_id):
                    config = ModelConfig()
                    config.num_layers = 8
                    model_args = ModelArgs()
                    model_args.ckpt_path = ckpt_path
                    model_args.task_type = TaskType.LANGUAGE_MODEL
                    model_args.act_type = "bf16"
                    model_args.json_model_override_args = json.dumps(
                        {"hidden_state_capture_layer_ids": [layer_id]}
                    )
                    kv_cache_config = KVCacheConfig()
                    debug_config = ProfilingDebugLoggingConfig()
                    debug_config.hack_layer_num = 4

                    if layer_id == 3:
                        build_model_config(
                            config, model_args, kv_cache_config, debug_config
                        )
                        self.assertEqual(config.hidden_state_capture_layer_ids, [3])
                    else:
                        with self.assertRaisesRegex(ValueError, "out-of-range"):
                            build_model_config(
                                config, model_args, kv_cache_config, debug_config
                            )
                        self.assertEqual(config.hidden_state_capture_layer_ids, [])
                    self.assertEqual(config.num_layers, 4)

    def test_invalid_capture_layers_are_rejected(self) -> None:
        invalid_overrides = [
            ('{"hidden_state_capture_layer_ids":[1,1]}', "unique"),
            ('{"hidden_state_capture_layer_ids":[-1]}', "out-of-range"),
            ('{"hidden_state_capture_layer_ids":[8]}', "out-of-range"),
            ('{"hidden_state_capture_layer_ids":[true]}', "list of integers"),
            ('{"hidden_state_capture_layer_ids":"1,2"}', "list of integers"),
        ]
        for override, error in invalid_overrides:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, error
            ):
                self.config.apply_override_args(override)

    def test_invalid_capture_dtype_does_not_partially_update_layers(self) -> None:
        self.config.hidden_state_capture_layer_ids = [7]

        with self.assertRaises(ValueError) as context:
            self.config.apply_override_args(
                '{"hidden_state_capture_layer_ids":[6,1,4],'
                '"hidden_state_capture_dtype":"float16"}'
            )
        self.assertEqual(
            str(context.exception),
            "hidden_state_capture_dtype must be 'bf16' or 'fp8_e4m3'",
        )

        self.assertEqual(self.config.hidden_state_capture_layer_ids, [7])
        self.assertEqual(
            self.config.hidden_state_capture_dtype, HiddenStateCaptureDtype.BF16
        )

    def test_capture_dtype_overrides_and_model_config_dump(self) -> None:
        for enum_name, dtype in HiddenStateCaptureDtype.__members__.items():
            with self.subTest(enum_name=enum_name):
                self.config.apply_override_args(
                    f'{{"hidden_state_capture_dtype":"{enum_name.lower()}"}}'
                )
                self.assertEqual(self.config.hidden_state_capture_dtype, dtype)

                self.config.hidden_state_capture_dtype = dtype
                self.assertIn(
                    f"hidden_state_capture_dtype: {enum_name.lower()}\n",
                    self.config.to_string(),
                )


if __name__ == "__main__":
    unittest.main()
