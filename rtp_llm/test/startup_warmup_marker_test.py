"""Unit tests for the fail-over warmup-skip marker in start_server.py.

Lever C from plan_jit.md: on container fail-over with a warm on-disk JIT cache,
skip the 82s real warmup that re-captures CUDA graphs, gated by a per-image
marker file inside the JIT cache directory.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main, mock

import rtp_llm.start_server as start_server


def _fake_configs(model_type="deepseek_v4", role="PREFILL"):
    role_config = SimpleNamespace(role_type=role)
    model_args = SimpleNamespace(model_type=model_type)
    return SimpleNamespace(role_config=role_config, model_args=model_args)


class WarmupMarkerTest(TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.jit_cache = os.path.join(self._tmpdir.name, "jit_cache")
        os.makedirs(self.jit_cache, exist_ok=True)

        self._env_patcher = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        for k in (
            "DSV4_STARTUP_REAL_WARMUP",
            "IMAGE_TAG",
            "HIPPO_APP_WORKDIR",
            "HIPPO_PROC_WORKDIR",
        ):
            os.environ.pop(k, None)
        os.environ["IMAGE_TAG"] = "img-sha-deadbeef"

        self._cache_patcher = mock.patch.object(
            start_server, "_local_jit_cache_dir", return_value=self.jit_cache
        )
        self._cache_patcher.start()
        # Make /etc/image-info lookup deterministic: force it to a path that
        # doesn't exist so only IMAGE_TAG env contributes unless a test
        # overrides this patch.
        self._image_info_patcher = mock.patch.object(
            start_server, "IMAGE_INFO_FILE", "/nonexistent/image-info"
        )
        self._image_info_patcher.start()

    def tearDown(self):
        self._image_info_patcher.stop()
        self._cache_patcher.stop()
        self._env_patcher.stop()
        self._tmpdir.cleanup()

    # ---- _resolve_image_tag ----

    def test_resolve_image_tag_from_env(self):
        os.environ["IMAGE_TAG"] = "abc123"
        self.assertEqual(start_server._resolve_image_tag(), "abc123")

    def test_resolve_image_tag_strips_whitespace(self):
        os.environ["IMAGE_TAG"] = "  abc123  \n"
        self.assertEqual(start_server._resolve_image_tag(), "abc123")

    def test_resolve_image_tag_fallback_to_file(self):
        os.environ.pop("IMAGE_TAG", None)
        info = os.path.join(self._tmpdir.name, "image-info")
        with open(info, "w") as f:
            f.write("file-tag-xyz\n")
        with mock.patch.object(start_server, "IMAGE_INFO_FILE", info):
            self.assertEqual(start_server._resolve_image_tag(), "file-tag-xyz")

    def test_resolve_image_tag_returns_none_when_unset(self):
        os.environ.pop("IMAGE_TAG", None)
        self.assertIsNone(start_server._resolve_image_tag())

    def test_resolve_image_tag_returns_none_when_empty_file(self):
        os.environ.pop("IMAGE_TAG", None)
        info = os.path.join(self._tmpdir.name, "image-info")
        with open(info, "w") as f:
            f.write("   \n")
        with mock.patch.object(start_server, "IMAGE_INFO_FILE", info):
            self.assertIsNone(start_server._resolve_image_tag())

    # ---- _real_warmup_baked_marker ----

    def test_marker_path_is_image_namespaced(self):
        marker = start_server._real_warmup_baked_marker()
        self.assertIsNotNone(marker)
        assert marker is not None
        self.assertTrue(marker.startswith(self.jit_cache + os.sep))
        self.assertIn("img-sha-deadbeef", os.path.basename(marker))

    def test_marker_none_when_no_jit_cache_dir(self):
        with mock.patch.object(start_server, "_local_jit_cache_dir", return_value=None):
            self.assertIsNone(start_server._real_warmup_baked_marker())

    def test_marker_none_when_image_tag_unresolvable(self):
        # Fail-closed: without an image tag the marker cannot be trusted, so we
        # never emit a path -- preventing both read and write.
        os.environ.pop("IMAGE_TAG", None)
        self.assertIsNone(start_server._real_warmup_baked_marker())

    def test_marker_sanitizes_unsafe_chars(self):
        os.environ["IMAGE_TAG"] = "registry/repo:tag with space"
        marker = start_server._real_warmup_baked_marker()
        assert marker is not None
        self.assertNotIn("/", os.path.basename(marker)[1:])
        self.assertNotIn(" ", os.path.basename(marker))

    # ---- _should_run_startup_real_warmup ----

    def test_auto_runs_warmup_when_no_marker(self):
        self.assertTrue(start_server._should_run_startup_real_warmup(_fake_configs()))

    def test_auto_skips_warmup_when_marker_exists(self):
        marker = start_server._real_warmup_baked_marker()
        assert marker is not None
        open(marker, "w").close()
        self.assertFalse(start_server._should_run_startup_real_warmup(_fake_configs()))

    def test_off_short_circuits_before_marker_check(self):
        os.environ["DSV4_STARTUP_REAL_WARMUP"] = "off"
        # No marker on disk -- still returns False because off wins.
        self.assertFalse(start_server._should_run_startup_real_warmup(_fake_configs()))

    def test_force_ignores_marker_for_prefill(self):
        marker = start_server._real_warmup_baked_marker()
        assert marker is not None
        open(marker, "w").close()
        os.environ["DSV4_STARTUP_REAL_WARMUP"] = "force"
        self.assertTrue(start_server._should_run_startup_real_warmup(_fake_configs()))

    def test_force_does_not_run_for_non_prefill(self):
        os.environ["DSV4_STARTUP_REAL_WARMUP"] = "force"
        self.assertFalse(
            start_server._should_run_startup_real_warmup(_fake_configs(role="FRONTEND"))
        )

    def test_auto_does_not_run_for_non_dsv4(self):
        self.assertFalse(
            start_server._should_run_startup_real_warmup(
                _fake_configs(model_type="qwen3")
            )
        )

    def test_auto_does_not_run_for_non_prefill(self):
        self.assertFalse(
            start_server._should_run_startup_real_warmup(_fake_configs(role="DECODE"))
        )

    def test_auto_runs_when_image_tag_unresolvable(self):
        # Fail-closed: no tag -> no marker -> warmup runs (safe default).
        os.environ.pop("IMAGE_TAG", None)
        self.assertTrue(start_server._should_run_startup_real_warmup(_fake_configs()))

    def test_marker_for_other_image_does_not_skip(self):
        other = os.path.join(
            self.jit_cache,
            f"{start_server.REAL_WARMUP_MARKER_PREFIX}some-old-image",
        )
        open(other, "w").close()
        self.assertTrue(start_server._should_run_startup_real_warmup(_fake_configs()))

    # ---- _write_real_warmup_marker ----

    def test_write_marker_creates_file_with_timestamp(self):
        start_server._write_real_warmup_marker()
        marker = start_server._real_warmup_baked_marker()
        assert marker is not None
        self.assertTrue(os.path.exists(marker))
        with open(marker) as f:
            content = f.read()
        self.assertIn("completed_at=", content)

    def test_write_marker_is_noop_when_no_tag(self):
        os.environ.pop("IMAGE_TAG", None)
        # Should not raise even though marker resolves to None.
        start_server._write_real_warmup_marker()
        # And the cache dir stays empty.
        self.assertEqual(os.listdir(self.jit_cache), [])

    def test_write_marker_creates_parent_dir_if_missing(self):
        nested = os.path.join(self._tmpdir.name, "nested", "jit_cache")
        with mock.patch.object(
            start_server, "_local_jit_cache_dir", return_value=nested
        ):
            start_server._write_real_warmup_marker()
            self.assertTrue(os.path.isdir(nested))
            entries = os.listdir(nested)
            self.assertEqual(len(entries), 1)
            self.assertTrue(
                entries[0].startswith(start_server.REAL_WARMUP_MARKER_PREFIX)
            )

    def test_round_trip_write_then_skip(self):
        # Simulates: first launch warmup completes -> marker written ->
        # fail-over relaunch -> _should_run_startup_real_warmup returns False.
        self.assertTrue(start_server._should_run_startup_real_warmup(_fake_configs()))
        start_server._write_real_warmup_marker()
        self.assertFalse(start_server._should_run_startup_real_warmup(_fake_configs()))


if __name__ == "__main__":
    main()
