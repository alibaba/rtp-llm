import json
import unittest

from rtp_llm.config.model_config import ModelConfig, apply_layer_num_override
from rtp_llm.ops import HiddenStateCaptureDtype


class ModelConfigOverrideTest(unittest.TestCase):
    CAPTURE_LIMIT_DEFAULTS = {
        "hidden_state_capture_max_inflight_samples": 64,
        "hidden_state_capture_max_inflight_bytes": 1073741824,
        "hidden_state_capture_max_inflight_tokens": 131072,
        "hidden_state_capture_max_pending_batches": 8,
        "hidden_state_capture_poll_interval_ms": 1000,
    }

    def setUp(self) -> None:
        self.config = ModelConfig()
        self.config.num_layers = 8

    def test_capture_defaults_are_disabled(self) -> None:
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [])
        self.assertEqual(
            self.config.hidden_state_capture_dtype, HiddenStateCaptureDtype.BF16
        )

        for name, default in self.CAPTURE_LIMIT_DEFAULTS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(self.config, name), default)
                self.assertIn(f"{name}: {default}\n", self.config.to_string())

    def test_capture_limit_overrides_round_trip_and_do_not_enable_capture(self) -> None:
        for name in self.CAPTURE_LIMIT_DEFAULTS:
            for value in (1, 17, 2**63 - 1):
                with self.subTest(name=name, value=value):
                    self.config.apply_override_args(json.dumps({name: value}))
                    self.assertEqual(getattr(self.config, name), value)
                    self.assertIn(f"{name}: {value}\n", self.config.to_string())
                    self.assertEqual(self.config.hidden_state_capture_layer_ids, [])

    def test_invalid_capture_limits_do_not_partially_update_capture_fields(
        self,
    ) -> None:
        self.config.hidden_state_capture_layer_ids = [7]
        for name in self.CAPTURE_LIMIT_DEFAULTS:
            for value in (0, -1, True, False, 1.5, "1", None, [], {}, 2**63):
                with self.subTest(name=name, value=value):
                    overrides = dict.fromkeys(self.CAPTURE_LIMIT_DEFAULTS, 17)
                    overrides.update(
                        hidden_state_capture_layer_ids=[6, 1, 4],
                        hidden_state_capture_dtype="fp8_e4m3",
                    )
                    overrides[name] = value
                    with self.assertRaisesRegex(
                        ValueError, f"^{name} must be a positive int64 integer$"
                    ):
                        self.config.apply_override_args(json.dumps(overrides))
                    self.assertEqual(self.config.hidden_state_capture_layer_ids, [7])
                    self.assertEqual(
                        self.config.hidden_state_capture_dtype,
                        HiddenStateCaptureDtype.BF16,
                    )
                    for limit_name, default in self.CAPTURE_LIMIT_DEFAULTS.items():
                        self.assertEqual(getattr(self.config, limit_name), default)

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

    def test_layer_override_precedes_capture_layer_validation(self) -> None:
        apply_layer_num_override(self.config, 4)

        self.config.apply_override_args('{"hidden_state_capture_layer_ids":[3]}')
        self.assertEqual(self.config.num_layers, 4)
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [3])

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.config.apply_override_args('{"hidden_state_capture_layer_ids":[4]}')
        self.assertEqual(self.config.hidden_state_capture_layer_ids, [3])

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

    def test_capture_dtype_override_round_trips_pybind_enum(self) -> None:
        dtype_by_name = {
            name.lower(): dtype
            for name, dtype in HiddenStateCaptureDtype.__members__.items()
        }
        self.assertEqual(set(dtype_by_name), {"bf16", "fp8_e4m3"})

        for dtype_name, expected_dtype in dtype_by_name.items():
            with self.subTest(dtype_name=dtype_name):
                self.config.apply_override_args(
                    f'{{"hidden_state_capture_dtype":"{dtype_name}"}}'
                )
                self.assertEqual(self.config.hidden_state_capture_dtype, expected_dtype)

    def test_capture_dtype_enum_round_trips_and_is_in_model_config_dump(
        self,
    ) -> None:
        for enum_name, dtype in HiddenStateCaptureDtype.__members__.items():
            with self.subTest(enum_name=enum_name):
                self.config.hidden_state_capture_dtype = dtype

                self.assertEqual(self.config.hidden_state_capture_dtype, dtype)
                self.assertIn(
                    f"hidden_state_capture_dtype: {enum_name.lower()}\n",
                    self.config.to_string(),
                )


if __name__ == "__main__":
    unittest.main()
