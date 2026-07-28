import os
import tempfile
import unittest
from unittest import mock

from rtp_llm.models.downstream_modules.custom_module import (
    CustomHandler,
    Trigger,
)
from rtp_llm.models.downstream_modules.utils import create_post_layers_module

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

    def test_non_eager_mode_fails(self):
        with mock.patch.dict(
            os.environ,
            {
                "CUSTOM_OUTPUT_PROCESSOR": "some.module",
                "CUSTOM_PROCESSOR_MODE": "compiled",
            },
        ):
            with self.assertRaises(RuntimeError):
                create_post_layers_module(None, None)

    def test_missing_file_fails(self):
        with mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_PROCESSOR": "/nonexistent/processor.py"},
        ):
            with self.assertRaises(Exception):
                create_post_layers_module(None, None)


if __name__ == "__main__":
    unittest.main()
