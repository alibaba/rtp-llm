import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.utils import jit_sys_path_setup as wrapper_cache
from rtp_llm.utils import jit_cache_env as cache_env


class CacheEnvTest(unittest.TestCase):
    def test_default_staging_keeps_cubin_headers_after_runfiles_removal(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ):
            root = Path(temporary)
            source = (
                root
                / "action.runfiles/pip_cuda13_arm_torch_flashinfer_cubin/site-packages/flashinfer_cubin"
            )
            header = "cubins/fmha/include/flashInferMetaInfo.h"
            (source / header).parent.mkdir(parents=True)
            (source / header).write_text("bundled header\n")
            (source / "__init__.py").write_text("")
            for name, directory in (
                ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
                ("DG_JIT_CACHE_DIR", "deep_gemm"),
                ("TRITON_CACHE_DIR", "triton"),
            ):
                os.environ[name] = str(root / directory)
            for name in ("REMOTE_JIT_DIR", "RUNFILES_DIR", "_JIT_CACHE_PATHS"):
                os.environ.pop(name, None)

            def package_info(name):
                return (
                    ("0.6.9", str(source))
                    if name == "flashinfer_cubin"
                    else (None, None)
                )

            with mock.patch.object(
                wrapper_cache, "get_package_info", side_effect=package_info
            ):
                wrapper_cache.setup_jit_cache(cache_dir=root / "packages")
            paths = cache_env.backend_pythonpath_prefix()
            self.assertEqual(len(paths), 1)
            cached_header = Path(paths[0]) / "flashinfer_cubin" / header
            shutil.rmtree(root / "action.runfiles")
            self.assertEqual(cached_header.read_text(), "bundled header\n")
            self.assertNotIn(".runfiles", str(cached_header))

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


class BackendPythonpathPrefixTest(unittest.TestCase):
    def test_returns_stable_cache_entries_in_order(self):
        env = {"_JIT_CACHE_PATHS": os.pathsep.join(("/cache/a", "/cache/b"))}
        self.assertEqual(
            cache_env.backend_pythonpath_prefix(env), ["/cache/a", "/cache/b"]
        )

    def test_skips_empty_entries(self):
        env = {"_JIT_CACHE_PATHS": f"{os.pathsep}/cache/a{os.pathsep}"}
        self.assertEqual(cache_env.backend_pythonpath_prefix(env), ["/cache/a"])

    def test_absent_variable_yields_no_prefix(self):
        self.assertEqual(cache_env.backend_pythonpath_prefix({}), [])


if __name__ == "__main__":
    unittest.main()
