import os
import sys
import tempfile
import unittest
from unittest import mock

from rtp_llm.config.model_args import ModelArgs
from rtp_llm.server.server_args.model_group_args import init_model_group_args
from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.utils.import_util import load_external_model_packages


class TestExternalModelPackages(unittest.TestCase):
    def setUp(self):
        self._env_mappings = dict(EnvArgumentParser._env_mappings)
        EnvArgumentParser._env_mappings = {}

    def tearDown(self):
        EnvArgumentParser._env_mappings = self._env_mappings

    def test_external_model_packages_env_binds_to_model_args(self):
        model_args = ModelArgs()
        parser = EnvArgumentParser(description="test")
        parser.set_root_config(model_args)
        init_model_group_args(parser, model_args)

        with mock.patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtp_llm.models"},
            clear=False,
        ):
            parser.parse_args([])

        self.assertEqual(
            model_args.external_model_packages, "atom.plugin.rtp_llm.models"
        )

    def test_load_external_model_packages_imports_csv_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for package_name in ("external_model_a", "external_model_b"):
                package_dir = os.path.join(tmpdir, package_name)
                os.mkdir(package_dir)
                with open(os.path.join(package_dir, "__init__.py"), "w") as f:
                    f.write("LOADED = True\n")

            with mock.patch.object(sys, "path", [tmpdir] + sys.path):
                load_external_model_packages(" external_model_a,external_model_b ")

            import external_model_a
            import external_model_b

            self.assertTrue(external_model_a.LOADED)
            self.assertTrue(external_model_b.LOADED)


if __name__ == "__main__":
    unittest.main()
