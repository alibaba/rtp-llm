import os
import unittest
from enum import Enum
from unittest import mock

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_custom_output_test import (
    PostLayersProcessor,
    run_post_layers,
)


class Stage(str, Enum):
    PRE = "pre_final_norm"
    POST = "post_final_norm"


class LegacyHandler:
    def extend_forward_args(self):
        return ["last_hidden_states"]

    def trigger_mode(self):
        return "context"

    def extend_forward(self, **kwargs):
        return next(iter(kwargs.values())).clone()


class Handler(LegacyHandler):
    def __init__(self, stage=Stage.POST, selected=False):
        self.stage = stage
        self.selected = selected

    def hidden_state_stage(self):
        return self.stage

    def extend_forward_args(self):
        return ["selected_hidden_states" if self.selected else "last_hidden_states"]


class Model:
    supports_pre_final_norm = True

    def initialize(self, resources):
        return True


class CustomOutputTestBase(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in (
            "CUSTOM_OUTPUT_TRACKED_TOKEN_ID",
            "CUSTOM_OUTPUT_TOKEN_POSITION",
            "CUSTOM_OUTPUT_EXPECTED_TOKEN_ID",
        ):
            os.environ.pop(name, None)


class CustomOutputProtocolTest(CustomOutputTestBase):
    def test_legacy_handler_defaults_to_post(self):
        processor = PostLayersProcessor()
        processor.set_handler(LegacyHandler())
        self.assertFalse(processor.uses_pre_final_norm())

    def test_enum_string_and_reset(self):
        processor = PostLayersProcessor()
        for stage in (Stage.PRE, "pre_final_norm"):
            processor.set_handler(Handler(stage))
            self.assertTrue(processor.uses_pre_final_norm())
        processor.set_handler(None)
        self.assertFalse(processor.uses_pre_final_norm())
        processor.set_handler(Handler())
        self.assertFalse(processor.uses_pre_final_norm())

    def test_invalid_stage_fails_registration(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported.*hidden_state_stage"):
            PostLayersProcessor().set_handler(Handler("pre_norm_typo"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
class CustomOutputPostLayersTest(CustomOutputTestBase):
    def setUp(self):
        super().setUp()
        self.hidden = (
            torch.arange(24, device="cuda", dtype=torch.float32).reshape(6, 4) + 1
        )
        self.normalized = self.hidden * torch.rsqrt(
            self.hidden.square().mean(-1, keepdim=True) + 1e-5
        )

    def run_case(
        self, stage, python_norm, selected, all_logits, pre_hidden=None, model=None
    ):
        env = {"CUSTOM_OUTPUT_TRACKED_TOKEN_ID": "42"} if selected else {}
        with mock.patch.dict(os.environ, env):
            # One decode row, then two variable-length context requests.
            return run_post_layers(
                model or Model(),
                Handler(stage, selected),
                self.normalized if python_norm else self.hidden,
                torch.tensor([0, 2, 5], dtype=torch.int32),
                torch.tensor([1, 4], dtype=torch.int32) if selected else None,
                1,
                python_norm,
                all_logits,
                pre_hidden,
            )

    def test_pre_and_post_keep_lm_logits_identical(self):
        for python_norm in (True, False):
            for selected in (True, False):
                for all_logits in (True, False):
                    with self.subTest(
                        python_norm=python_norm,
                        selected=selected,
                        all_logits=all_logits,
                    ):
                        rows = [1, 4] if selected else [2, 5]
                        pre = self.run_case(
                            Stage.PRE,
                            python_norm,
                            selected,
                            all_logits,
                            self.hidden[rows] if python_norm else None,
                        )
                        post = self.run_case(
                            Stage.POST, python_norm, selected, all_logits
                        )
                        self.assertEqual(pre["custom_output_error"], "")
                        self.assertEqual(post["custom_output_error"], "")
                        torch.testing.assert_close(
                            pre["custom_output"], self.hidden[rows]
                        )
                        torch.testing.assert_close(
                            post["custom_output"], self.normalized[rows]
                        )
                        torch.testing.assert_close(
                            pre["logits"], post["logits"], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            pre["hidden_states"], post["hidden_states"], rtol=0, atol=0
                        )

    def test_missing_or_malformed_python_capture_never_falls_back(self):
        for capture in (
            None,
            self.hidden[:1],
            self.hidden[:2, :2],
            self.hidden[:2].double(),
        ):
            with self.subTest(capture=capture):
                result = self.run_case(Stage.PRE, True, True, False, capture)
                self.assertIsNone(result["custom_output"])
                self.assertIn("pre_final_norm", result["custom_output_error"])

    def test_unsupported_python_model_fails_startup(self):
        model = Model()
        model.supports_pre_final_norm = False
        with self.assertRaisesRegex(RuntimeError, "does not support pre_final_norm"):
            self.run_case(Stage.PRE, True, True, False, model=model)

    def test_decode_does_not_invoke_handler(self):
        result = run_post_layers(
            Model(),
            Handler(Stage.PRE),
            self.normalized[:1],
            torch.tensor([0], dtype=torch.int32),
            None,
            1,
            True,
            False,
            None,
        )
        self.assertIsNone(result["custom_output"])
        self.assertEqual(result["custom_output_error"], "")

    def test_disabled_handler_and_legacy_post_path_preserve_generation(self):
        disabled = run_post_layers(
            Model(),
            None,
            self.normalized,
            torch.tensor([0, 2, 5], dtype=torch.int32),
            None,
            1,
            True,
            False,
            None,
        )
        legacy = run_post_layers(
            Model(),
            LegacyHandler(),
            self.normalized,
            torch.tensor([0, 2, 5], dtype=torch.int32),
            None,
            1,
            True,
            False,
            None,
        )
        self.assertIsNone(disabled["custom_output"])
        torch.testing.assert_close(legacy["custom_output"], self.normalized[[2, 5]])
        torch.testing.assert_close(disabled["logits"], legacy["logits"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
