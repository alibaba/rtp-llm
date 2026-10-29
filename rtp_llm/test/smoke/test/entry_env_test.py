import os
import unittest
from unittest.mock import patch

from smoke.entry import _expand_required_env


class EntryEnvironmentExpansionTest(unittest.TestCase):
    def test_expands_explicit_braced_placeholder(self):
        with patch.dict(
            os.environ, {"QWEN4_EXP_CHECKPOINT": "/mnt/models/qwen4"}, clear=True
        ):
            self.assertEqual(
                _expand_required_env(
                    "--checkpoint ${QWEN4_EXP_CHECKPOINT}/mtp",
                    field="smoke_args",
                ),
                "--checkpoint /mnt/models/qwen4/mtp",
            )

    def test_rejects_missing_or_empty_required_variable(self):
        for environment in ({}, {"QWEN4_EXP_CHECKPOINT": ""}):
            with self.subTest(environment=environment), patch.dict(
                os.environ, environment, clear=True
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "non-empty environment variable QWEN4_EXP_CHECKPOINT"
                ):
                    _expand_required_env("${QWEN4_EXP_CHECKPOINT}", field="model_path")

    def test_leaves_literal_paths_unchanged(self):
        self.assertEqual(
            _expand_required_env("/mnt/models/qwen4", field="model_path"),
            "/mnt/models/qwen4",
        )

    def test_hf_fixture_is_resolved_before_building_server_arguments(self):
        uri = "hf://owner/model@" + "a" * 40
        with patch.dict(os.environ, {"QWEN4_EXP_CHECKPOINT": uri}), patch(
            "smoke.entry.resolve_hf_checkpoint", return_value="/cache/pinned"
        ) as resolve:
            self.assertEqual(
                _expand_required_env(
                    "--checkpoint ${QWEN4_EXP_CHECKPOINT}", field="smoke_args"
                ),
                "--checkpoint /cache/pinned",
            )
        resolve.assert_called_once_with(uri)


if __name__ == "__main__":
    unittest.main()
