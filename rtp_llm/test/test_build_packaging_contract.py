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

# Hosts that serve internal-only / non-publicly-reproducible build artifacts. Direct wheel pins
# from these must never reach a publicly published wheel's install metadata. Kept deliberately
# narrow (see the review decision): the OSS `rtp-opensource.*` bucket and `download.pytorch.org`
# are public and allowed.
INTERNAL_ONLY_HOST_MARKERS = ("sinian-metrics-platform",)

# Platform extras whose stack is merged into install_requires of a PUBLICLY published wheel.
# `rocm` is intentionally excluded: its wheel is not published to public indexes, so its
# internal `sinian-metrics-platform` pins never ship to external users. If ROCm wheels ever
# become publicly published, add "rocm" here and relocate those pins to the internal overlay.
PUBLIC_PLATFORM_EXTRAS = ("cuda12", "cuda12_arm", "cuda12_9")


def _oss_optional_extras() -> dict:
    """Load [project.optional-dependencies] from the OSS extras file that ships in this repo."""
    extras_file = PROJECT_ROOT / "_build" / "oss_optional_extras.toml"
    with open(extras_file, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("optional-dependencies", {})


def _requirement_url(req: str) -> str:
    """Return the direct-reference URL of a `name @ url` requirement, else ''."""
    parts = req.split(" @ ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _is_local_path_reference(url: str) -> bool:
    """True if a direct reference points at a local path rather than a remote artifact."""
    if not url:
        return False
    if url.startswith("file:"):
        return True
    # Absolute or relative filesystem paths (never valid for a published wheel).
    return url.startswith(("/", "./", "../"))


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

    def test_public_platform_extras_have_no_internal_only_wheel_pins(self):
        """Publicly published wheels must not carry internal-only wheel sources in their metadata.

        setup.get_all_dependencies() merges the auto-detected platform's extras into
        install_requires, so any internal-only direct wheel pin in a public platform stack would
        leak the internal source into the published wheel's install metadata. Assert none of the
        public platform extras reference an internal-only host.
        """
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if any(marker in url for marker in INTERNAL_ONLY_HOST_MARKERS):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders,
            [],
            "Public platform extras must not pin internal-only wheels; move these to the "
            f"internal overlay (internal_source/pyproject_internal.toml):\n{offenders}",
        )

    def test_public_platform_extras_have_no_local_path_dependencies(self):
        """Public platform extras must not reference local filesystem paths (non-reproducible)."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                if _is_local_path_reference(_requirement_url(req)):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders, [], f"Public platform extras must not use local paths:\n{offenders}"
        )

    def test_public_platform_extras_use_https_for_direct_wheels(self):
        """Direct wheel pins in public platform extras must be fetched over HTTPS, not plaintext."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if url.startswith("http://"):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders, [], f"Public platform extras must use https:// wheel URLs:\n{offenders}"
        )

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
