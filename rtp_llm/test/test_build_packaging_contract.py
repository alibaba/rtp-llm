import importlib.util
import os
import re
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from setuptools import find_namespace_packages

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "_rtp_llm_setup_under_test", PROJECT_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(os.environ, {"RTP_BAZEL_CONFIG": "--config=cuda12_9"}, clear=False):
        spec.loader.exec_module(module)
    return module


class BuildPackagingContractTest(TestCase):
    def test_dynamic_version_uses_release_version(self):
        setup_module = _load_setup_module()
        release_text = (PROJECT_ROOT / "rtp_llm" / "release_version.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r'^RELEASE_VERSION\s*=\s*["\']([^"\']+)["\']', release_text, re.M
        )
        assert match is not None
        expected = match.group(1)

        self.assertEqual(setup_module.get_release_version(), expected)
        self.assertEqual(setup_module.get_version_with_platform(), f"{expected}+cu129")

    def test_pytest_entry_points_are_packaged_with_tests(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        packages = set(
            find_namespace_packages(
                where=str(PROJECT_ROOT), include=["rtp_llm", "rtp_llm.*"]
            )
        )

        self.assertIn("rtp_llm.test.remote_tests", packages)
        self.assertIn("rtp_llm.test.smoke_framework", packages)
        find_cfg = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertNotIn("exclude", find_cfg)

        entry_points = pyproject["project"]["entry-points"]["pytest11"]
        for target in entry_points.values():
            module_name = target.split(":", 1)[0]
            module_path = PROJECT_ROOT / (module_name.replace(".", "/") + ".py")
            self.assertTrue(module_path.exists(), module_name)

        package_data = pyproject["tool"]["setuptools"]["package-data"]["rtp_llm"]
        self.assertIn("test/**/*.proto", package_data)
        self.assertIn("test/**/*.json", package_data)
