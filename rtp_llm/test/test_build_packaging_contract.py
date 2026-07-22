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


def _load_platform_module():
    """Load _build/platform.py in isolation (stdlib-only, no side effects)."""
    spec = importlib.util.spec_from_file_location(
        "_rtp_build_platform_under_test", PROJECT_ROOT / "_build" / "platform.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    def test_pytest_testpaths_all_exist(self):
        """Every configured pytest testpath must exist.

        A stale/typo'd testpath (e.g. rtp_llm/models/multimodal/test vs the real
        rtp_llm/multimodal/test) is silently dropped by pytest, so those tests never run in CI.
        Assert each path resolves so such drift fails loudly at contract-test time instead.
        """
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        testpaths = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]
        missing = [p for p in testpaths if not (PROJECT_ROOT / p).exists()]
        self.assertEqual(
            missing, [], f"pyproject testpaths point at non-existent directories: {missing}"
        )

    def test_rocm_wheel_version_matches_dependency_abi(self):
        """The rocm wheel version suffix must track the ROCm ABI the rocm extras are built for.

        get_version_with_platform() stamps every OSS ROCm wheel with this suffix, so a stale value
        (e.g. rocm62 while the deps/toolchain moved to ROCm 7.2) makes cache/publish/rollback pick
        the wrong binary stack. Derive the ABI from the suffix and assert the rocm extras' wheels
        actually reference it — and that no wheel references a different ROCm ABI.
        """
        platform_module = _load_platform_module()
        suffix = platform_module.PLATFORM_CONFIG_VERSIONS.get("rocm", "")
        m = re.fullmatch(r"rocm(\d)(\d+)", suffix)
        self.assertIsNotNone(m, f"unexpected rocm version suffix {suffix!r}")
        expected_abi = f"{m.group(1)}.{m.group(2)}"  # rocm72 -> "7.2"

        rocm_reqs = _oss_optional_extras().get("rocm", [])
        rocm_abis = set(re.findall(r"rocm(\d+\.\d+)", " ".join(rocm_reqs)))
        self.assertIn(
            expected_abi,
            rocm_abis,
            f"rocm suffix {suffix!r} (ABI {expected_abi}) not found in rocm extras "
            f"wheel URLs (found ABIs: {sorted(rocm_abis)})",
        )
        stale = rocm_abis - {expected_abi}
        self.assertEqual(
            stale, set(), f"rocm extras reference ROCm ABIs {sorted(stale)} != suffix {expected_abi}"
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
