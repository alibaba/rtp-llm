import os
import tempfile
import unittest
from unittest import mock

import torch

from rtp_llm.models.downstream_modules.custom_module import (
    CustomHandler,
    Trigger,
)
from rtp_llm.models.downstream_modules.utils import create_post_layers_module
from rtp_llm.utils.base_model_datatypes import (
    get_response_logits_with_custom_output_compat,
)

DUMMY_MODULE = """
SENTINEL = object()


class DummyModule:
    handler = None


def create_custom_module(config, tokenizer):
    module = DummyModule()
    module.config = config
    module.tokenizer = tokenizer
    return module
"""

DUMMY_NONE_MODULE = """
def create_custom_module(config, tokenizer):
    return None
"""

COMPILED_CAPABLE_MODULE = """
import torch


class Handler:
    def __init__(self):
        self.mlp = torch.nn.Linear(4, 1)

    def compiled_module(self):
        return self.mlp

    def extend_forward_args(self):
        return ["last_hidden_states"]


class Module:
    def __init__(self):
        self.handler = Handler()

    def get_handler(self):
        return self.handler


def create_custom_module(config, tokenizer):
    return Module()
"""

EAGER_ONLY_MODULE = COMPILED_CAPABLE_MODULE.replace(
    "return self.mlp", "return None"
)


class TriggerProtocolTest(unittest.TestCase):
    def test_default_trigger_is_context(self):
        handler = CustomHandler(None)
        self.assertEqual(handler.trigger_mode(), Trigger.CONTEXT)
        # the C++ side matches on the string value
        self.assertEqual(str(handler.trigger_mode().value), "context")

    def test_trigger_values(self):
        self.assertEqual(Trigger.CONTEXT.value, "context")
        self.assertEqual(Trigger.FINAL_STEP.value, "final_step")
        self.assertEqual(Trigger.EVERY_STEP.value, "every_step")


class CustomOutputResponseCompatTest(unittest.TestCase):
    def test_custom_output_is_returned_as_logits_without_request_flag(self):
        custom_output = torch.tensor([[1.0, 2.0]])

        result = get_response_logits_with_custom_output_compat(
            False, None, custom_output
        )

        self.assertIs(result, custom_output)

    def test_custom_output_takes_priority_over_vocabulary_logits(self):
        logits = torch.tensor([[3.0, 4.0]])
        custom_output = torch.tensor([[1.0, 2.0]])

        result = get_response_logits_with_custom_output_compat(
            True, logits, custom_output
        )

        self.assertIs(result, custom_output)

    def test_existing_return_logits_behavior_is_unchanged_without_custom_output(self):
        logits = torch.tensor([[3.0, 4.0]])

        self.assertIs(
            get_response_logits_with_custom_output_compat(True, logits, None), logits
        )
        self.assertIsNone(
            get_response_logits_with_custom_output_compat(False, logits, None)
        )


class CreatePostLayersModuleTest(unittest.TestCase):
    def test_unset_env_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUSTOM_OUTPUT_PROCESSOR", None)
            self.assertIsNone(create_post_layers_module(None, None))

    def test_loads_module_from_py_file(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(DUMMY_MODULE)
            path = tmp_file.name
        try:
            with mock.patch.dict(
                os.environ, {"CUSTOM_OUTPUT_PROCESSOR": path}
            ):
                module = create_post_layers_module("cfg", "tok")
                self.assertIsNotNone(module)
                self.assertEqual(module.config, "cfg")
                self.assertEqual(module.tokenizer, "tok")
        finally:
            os.unlink(path)

    def test_loads_relative_py_file_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            processor_name = "custom_output_processor.py"
            processor_path = os.path.join(ckpt_path, processor_name)
            with open(processor_path, "w") as processor_file:
                processor_file.write(DUMMY_MODULE)

            config = mock.Mock(ckpt_path=ckpt_path)
            with mock.patch.dict(
                os.environ, {"CUSTOM_OUTPUT_PROCESSOR": processor_name}
            ):
                module = create_post_layers_module(config, "tok")

            self.assertIsNotNone(module)
            self.assertIs(module.config, config)
            self.assertEqual(module.tokenizer, "tok")

    def test_relative_py_file_requires_checkpoint_path(self):
        with mock.patch.dict(
            os.environ, {"CUSTOM_OUTPUT_PROCESSOR": "custom_output_processor.py"}
        ):
            with self.assertRaisesRegex(RuntimeError, "checkpoint path"):
                create_post_layers_module(None, None)

    def test_module_returning_none_fails(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(DUMMY_NONE_MODULE)
            path = tmp_file.name
        try:
            with mock.patch.dict(
                os.environ, {"CUSTOM_OUTPUT_PROCESSOR": path}
            ):
                with self.assertRaises(RuntimeError):
                    create_post_layers_module(None, None)
        finally:
            os.unlink(path)

    def test_unknown_mode_fails(self):
        with mock.patch.dict(
            os.environ,
            {
                "CUSTOM_OUTPUT_PROCESSOR": "some.module",
                "CUSTOM_PROCESSOR_MODE": "jit",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "CUSTOM_PROCESSOR_MODE"):
                create_post_layers_module(None, None)

    def _write_module(self, source):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(source)
            return tmp_file.name

    def test_compiled_mode_marks_handler_for_aot(self):
        path = self._write_module(COMPILED_CAPABLE_MODULE)
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "CUSTOM_OUTPUT_PROCESSOR": path,
                    "CUSTOM_PROCESSOR_MODE": "compiled",
                },
            ):
                module = create_post_layers_module(None, None)
                # compile itself is deferred to injection time (after weight
                # init); create only validates and marks the handler
                self.assertTrue(module.get_handler()._aoti_requested)
        finally:
            os.unlink(path)

    def test_compiled_mode_without_compiled_module_fails(self):
        path = self._write_module(EAGER_ONLY_MODULE)
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "CUSTOM_OUTPUT_PROCESSOR": path,
                    "CUSTOM_PROCESSOR_MODE": "compiled",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "compiled_module"):
                    create_post_layers_module(None, None)
        finally:
            os.unlink(path)

    def test_missing_file_fails(self):
        with mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_PROCESSOR": "/nonexistent/processor.py"},
        ):
            with self.assertRaises(Exception):
                create_post_layers_module(None, None)


class EnsureAotiPackageTest(unittest.TestCase):
    def test_not_requested_returns_none(self):
        self.assertIsNone(CustomHandler(None).ensure_aoti_package())

    def test_requested_without_compiled_module_fails(self):
        handler = CustomHandler(None)
        handler._aoti_requested = True
        with self.assertRaisesRegex(RuntimeError, "compiled_module"):
            handler.ensure_aoti_package()


if __name__ == "__main__":
    unittest.main()
