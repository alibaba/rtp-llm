from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from rtp_llm.kv_cache_subscriber.config import (
    _env_bool,
    _normalize_dtype,
    build_parser,
    config_from_args,
)
from rtp_llm.kv_cache_subscriber.test_utils import make_config


class SubscriberConfigTest(unittest.TestCase):
    def test_cli_trims_and_filters_comma_separated_rtp_endpoints(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--rtp-endpoints",
                "10.0.0.1:8089, ,10.0.0.2:8098",
                "--kvcm-url",
                "http://kvcm.test:6382",
                "--instance-id",
                "instance-a",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(
            config.rtp_endpoints,
            ("10.0.0.1:8089", "10.0.0.2:8098"),
        )

    def test_spectrum_url_is_derived_from_virtual_service_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KVCM_VSERVICE_ID": "search-kvcm",
                "KVCM_INSTANCE_ID": "instance-a",
            },
            clear=True,
        ):
            config = config_from_args(build_parser().parse_args([]))

        self.assertEqual(config.kvcm_url, "spectrum://search-kvcm:6382")
        self.assertEqual(config.instance_id, "instance-a")

    def test_invalid_boolean_environment_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"KVCM_RESET_ON_START": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                build_parser()

    def test_boolean_environment_aliases_are_supported(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "yes"}):
            self.assertTrue(_env_bool("TEST_BOOL", False))
        with patch.dict(os.environ, {"TEST_BOOL": "off"}):
            self.assertFalse(_env_bool("TEST_BOOL", True))

    def test_rtp_model_environment_populates_kvcm_deployment_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KVCM_URL": "http://kvcm.test:6382",
                "KVCM_INSTANCE_ID": "instance-a",
                "CHECKPOINT_PATH": "/models/Qwen2-0.5B/",
                "ACT_TYPE": "bf16",
                "TP_SIZE": "2",
                "DP_SIZE": "1",
                "PP_SIZE": "1",
            },
            clear=True,
        ):
            config = config_from_args(build_parser().parse_args([]))

        self.assertEqual(config.model_name, "Qwen2-0.5B")
        self.assertEqual(config.model_dtype, "bfloat16")
        self.assertEqual(config.tp_size, 2)

    def test_model_dtype_aliases_are_normalized(self) -> None:
        self.assertEqual(_normalize_dtype(" BF16 "), "bfloat16")
        self.assertEqual(_normalize_dtype("fp16"), "float16")
        self.assertEqual(_normalize_dtype("float8_e4m3fn"), "float8_e4m3fn")

    def test_required_kvcm_identity_fields_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "KVCM_URL"):
            make_config(kvcm_url="").validate()
        with self.assertRaisesRegex(ValueError, "KVCM_INSTANCE_ID"):
            make_config(instance_id="").validate()

    def test_positive_intervals_and_batch_size_are_validated(self) -> None:
        invalid_configs = [
            (make_config(poll_interval_s=0), "poll_interval_s"),
            (make_config(deletion_confirmations=0), "deletion_confirmations"),
            (make_config(engine_failure_threshold=0), "engine_failure_threshold"),
            (make_config(kvcm_report_batch_size=0), "kvcm_report_batch_size"),
            (make_config(tp_size=0), "tp_size"),
        ]
        for config, field_name in invalid_configs:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
