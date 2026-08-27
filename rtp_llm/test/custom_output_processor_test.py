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


class CreatePostLayersModuleTest(unittest.TestCase):
    def test_unset_env_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUSTOM_OUTPUT_PROCESSOR", None)
            os.environ.pop("CUSTOM_OUTPUT_TOKEN_POSITION", None)
            os.environ.pop("CUSTOM_OUTPUT_TRACKED_TOKEN_ID", None)
            os.environ.pop("CUSTOM_OUTPUT_EXPECTED_TOKEN_ID", None)
            self.assertIsNone(create_post_layers_module(None, None))

    def test_selector_without_processor_fails(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUSTOM_OUTPUT_PROCESSOR", None)
            os.environ["CUSTOM_OUTPUT_TOKEN_POSITION"] = "-2"
            with self.assertRaisesRegex(RuntimeError, "requires CUSTOM_OUTPUT_PROCESSOR"):
                create_post_layers_module(None, None)

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

    def test_missing_file_fails(self):
        with mock.patch.dict(
            os.environ,
            {"CUSTOM_OUTPUT_PROCESSOR": "/nonexistent/processor.py"},
        ):
            with self.assertRaises(Exception):
                create_post_layers_module(None, None)


if __name__ == "__main__":
    unittest.main()
