import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.utils import jit_cache_env as cache_env


class CacheEnvTest(unittest.TestCase):
    def test_concurrent_directory_creation_preserves_peer_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "cache"
            mkdir = Path.mkdir

            def peer_creates(directory, *args, **kwargs):
                mkdir(directory, mode=0o775)
                directory.chmod(0o775)
                return mkdir(directory, *args, **kwargs)

            with mock.patch.object(Path, "mkdir", peer_creates):
                self.assertEqual(cache_env.ensure_writable_directory(path), path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o775)

    def test_new_nested_directory_is_private_under_umask(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "parent" / "cache"
            previous = os.umask(0o077)
            try:
                self.assertEqual(cache_env.ensure_writable_directory(path), path)
            finally:
                os.umask(previous)
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_directory_sources_and_capture_are_stable(self):
        env = {
            "TRITON_CACHE_DIR": "/automatic/triton",
            "DG_JIT_CACHE_DIR": " /explicit/deep-gemm ",
            "TORCH_EXTENSIONS_DIR": " ",
        }
        cache_env.configure_cache_env(
            "TRITON_CACHE_DIR", "/automatic/triton", automatic=True, environ=env
        )
        captured = cache_env.read_cache_env(
            ["TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR", "TORCH_EXTENSIONS_DIR"], env
        )
        self.assertEqual(captured.explicit_envs, {"DG_JIT_CACHE_DIR"})
        env["TRITON_CACHE_DIR"] = "/user/triton"
        self.assertEqual(captured.explicit_envs, {"DG_JIT_CACHE_DIR"})
        self.assertEqual(
            cache_env.read_cache_env(["TRITON_CACHE_DIR"], env).explicit_envs,
            {"TRITON_CACHE_DIR"},
        )

    def test_malformed_marker_keeps_presets_explicit(self):
        for payload in ("{", "[]", "null", '{"TRITON_CACHE_DIR": 7}'):
            with self.subTest(payload=payload):
                env = {
                    "TRITON_CACHE_DIR": "/configured/triton",
                    cache_env._AUTOMATIC_CACHE_ENVS: payload,
                }
                self.assertEqual(
                    cache_env.read_cache_env(["TRITON_CACHE_DIR"], env).explicit_envs,
                    {"TRITON_CACHE_DIR"},
                )
                cache_env.configure_cache_env(
                    "DG_JIT_CACHE_DIR", "/automatic/dg", automatic=True, environ=env
                )
                self.assertEqual(
                    json.loads(env[cache_env._AUTOMATIC_CACHE_ENVS]),
                    {"DG_JIT_CACHE_DIR": "/automatic/dg"},
                )

    def test_managed_paths_consume_only_their_automatic_entries(self):
        env = {}
        for name in ("TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR"):
            cache_env.configure_cache_env(
                name, "/fallback", automatic=True, environ=env
            )
        cache_env.configure_cache_env(
            "TRITON_CACHE_DIR", "/scope/triton", automatic=False, environ=env
        )
        self.assertEqual(env["TRITON_CACHE_DIR"], "/scope/triton")
        self.assertEqual(
            json.loads(env[cache_env._AUTOMATIC_CACHE_ENVS]),
            {"DG_JIT_CACHE_DIR": "/fallback"},
        )
        cache_env.configure_cache_env(
            "DG_JIT_CACHE_DIR", "/scope/deep-gemm", automatic=False, environ=env
        )
        self.assertNotIn(cache_env._AUTOMATIC_CACHE_ENVS, env)

    def test_existing_automatic_paths_retain_their_origin(self):
        env = {}
        cache_env.configure_cache_env(
            "TRITON_CACHE_DIR", "/fallback", automatic=True, environ=env
        )
        cache_env.configure_cache_env("TRITON_CACHE_DIR", "/fallback", environ=env)
        self.assertFalse(
            cache_env.read_cache_env(["TRITON_CACHE_DIR"], env).explicit_envs
        )

    def test_legacy_default_and_explicit_override_remain_opt_outs(self):
        env = {}
        cache_env.configure_cache_env(
            "DG_JIT_CACHE_DIR", "/legacy/default", environ=env
        )
        self.assertEqual(
            cache_env.read_cache_env(["DG_JIT_CACHE_DIR"], env).explicit_envs,
            {"DG_JIT_CACHE_DIR"},
        )
        cache_env.configure_cache_env(
            "DG_JIT_CACHE_DIR", "/automatic", automatic=True, environ=env
        )
        cache_env.configure_cache_env(
            "DG_JIT_CACHE_DIR", "/configured", automatic=False, environ=env
        )
        self.assertNotIn(cache_env._AUTOMATIC_CACHE_ENVS, env)
        self.assertEqual(env["DG_JIT_CACHE_DIR"], "/configured")


if __name__ == "__main__":
    unittest.main()
