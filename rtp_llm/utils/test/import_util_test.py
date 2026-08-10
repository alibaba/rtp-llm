import unittest
from unittest import mock

from rtp_llm.utils.import_util import load_external_model_packages


class ExternalModelPackagesTest(unittest.TestCase):
    @mock.patch("rtp_llm.utils.import_util.importlib.import_module")
    def test_imports_packages_in_order(self, import_module):
        load_external_model_packages(["plugin.models", "plugin.extra"])

        self.assertEqual(
            import_module.call_args_list,
            [mock.call("plugin.models"), mock.call("plugin.extra")],
        )

    @mock.patch("rtp_llm.utils.import_util.importlib.import_module")
    def test_empty_configuration_does_not_import(self, import_module):
        load_external_model_packages(None)
        load_external_model_packages([])

        import_module.assert_not_called()

    @mock.patch(
        "rtp_llm.utils.import_util.importlib.import_module",
        side_effect=ModuleNotFoundError("missing dependency"),
    )
    def test_import_failure_is_fail_fast_and_keeps_context(self, import_module):
        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to import external model package 'plugin.models'.*module search path",
        ) as raised:
            load_external_model_packages(["plugin.models", "plugin.extra"])

        self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)
        import_module.assert_called_once_with("plugin.models")


if __name__ == "__main__":
    unittest.main()
