"""Mock remote storage, but use real archives, processes and Triton host .so loads."""

import contextlib
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SOURCE = r"""
#include <Python.h>
int rtp_cache_probe_value(void) { return 42; }
static PyObject* answer(PyObject* self, PyObject* args) {
    return PyLong_FromLong(rtp_cache_probe_value());
}
static PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, "probe"}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "rtp_cache_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit_rtp_cache_probe(void) { return PyModule_Create(&module); }
"""

GPU_FIXTURES = {
    "deep_gemm": "hash/kernel.cubin",
    "trtllm_deep_gemm": "hash/nvcc_kernel.cubin",
}


def _manager(local: Path, remote: Path):
    from rtp_llm.utils import jit_cache_manager as jit

    for item in jit.COMPONENTS:
        os.environ.pop(item.env_name, None)
    os.environ.pop("TEST_JIT_LOCAL_DIR", None)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("torch.version.hip", None))
        stack.enter_context(mock.patch("torch.version.cuda", "13.0"))
        for name, value in (
            ("_accelerator_scope", "cuda-test"),
            ("_torch_scope", "torch-test"),
            ("_cpp_runtime_scope", "cxx-test"),
        ):
            stack.enter_context(mock.patch.object(jit, name, return_value=value))
        stack.enter_context(mock.patch.object(jit, "_pkg_version", return_value="1_0"))
        root = jit.resolve_local_root(str(local))
        components, compatible = jit.setup_jit_cache_env(root)
        assert compatible
        manager = jit.JitCacheManager(jit.resolve_remote_root(remote), components, root)
    manager.start_background_sync()
    return manager


def _stage(local: Path, remote: Path, mode: str) -> None:
    # Hardware probing is mocked; no CUDA context/model/service is created.
    with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
        "torch.cuda.device_count", return_value=0
    ):
        manager = _manager(local, remote)
    try:
        from triton.runtime import build
        from triton.runtime.cache import get_cache_manager

        cold = mode == "seed"
        guard = (
            contextlib.nullcontext()
            if cold
            else mock.patch.object(
                build,
                "_build",
                side_effect=AssertionError("cache hit must not compile"),
            )
        )
        with guard:
            module = build.compile_module_from_src(SOURCE, "rtp_cache_probe")
        loaded = Path(module.__file__)
        assert module.answer() == 42
        assert Path(os.environ["TRITON_CACHE_DIR"]) in loaded.parents
        key = hashlib.sha256((SOURCE + build.platform_key()).encode()).hexdigest()
        cache = get_cache_manager(key)
        if cold:
            cache.put_group("probe.json", {loaded.name: str(loaded)})
        else:
            assert cache.get_group("probe.json") == {loaded.name: str(loaded)}

        from rtp_llm.utils import jit_cache_manager as jit

        directories = {
            c.name: Path(os.environ[c.env_name])
            for c in jit.COMPONENTS
            if c.env_name in os.environ
        }
        for name, rel in GPU_FIXTURES.items():
            artifact = directories[name] / rel
            expected = f"mock GPU artifact: {name}".encode()
            if cold:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(expected)
            assert artifact.read_bytes() == expected
        tilelang_so = directories["tilelang"] / "hash/probe.so"
        if cold:
            tilelang_so.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(loaded, tilelang_so)
        assert ctypes.CDLL(str(tilelang_so)).rtp_cache_probe_value() == 42
        if cold:
            manager._dirty.set()
            manager.publish_pending_snapshot()
        print(
            f"{mode}: real .so returned 42; GPU fixtures restored; path={loaded}",
            flush=True,
        )
    finally:
        manager.stop()


class RemoteJitCacheE2ETest(unittest.TestCase):
    def round_trip(self, directory=None, relocate=False):
        with tempfile.TemporaryDirectory(
            prefix="jit-e2e-", dir=directory
        ) as tmp, tempfile.TemporaryDirectory(
            prefix="jit-remote-"
        ) as remote, tempfile.TemporaryDirectory(
            prefix="jit-seed-"
        ) as seed:
            local = Path(tmp) / "local"
            seed_local = Path(seed) / "local" if relocate else local

            def run(mode):
                return subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        mode,
                        str(seed_local if mode == "seed" else local),
                        remote,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=dict(
                        os.environ,
                        PYTHONPATH=os.pathsep.join(
                            str(Path(p or os.getcwd()).absolute()) for p in sys.path
                        ),
                    ),
                )

            def finish(process):
                try:
                    output, _ = process.communicate(timeout=60)
                except BaseException:
                    process.kill()
                    process.communicate()
                    raise
                self.assertEqual(process.returncode, 0, output)
                self.assertIn("real .so returned 42", output)
                print(output, end="")

            finish(run("seed"))
            self.assertTrue(list(Path(remote).rglob("*.jit_snapshot.tar.zst")))
            # Only this test's stopped producer tree is removed. Consumers either
            # use the same path or relocate its snapshot from disk to tmpfs.
            shutil.rmtree(seed_local)
            consumers = [run("hit"), run("hit")]
            try:
                for process in consumers:
                    finish(process)
            finally:
                for process in consumers:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            before = {p: p.stat().st_ino for p in local.rglob("*.so")}
            self.assertEqual(len(before), 2)
            finish(run("warm"))
            self.assertEqual(before, {p: p.stat().st_ino for p in before})

    def test_disk_round_trip(self):
        self.round_trip()

    @unittest.skipUnless(Path("/dev/shm").is_dir(), "requires Linux tmpfs")
    def test_tmpfs_round_trip(self):
        self.round_trip("/dev/shm")

    @unittest.skipUnless(Path("/dev/shm").is_dir(), "requires Linux tmpfs")
    def test_disk_snapshot_relocates_to_tmpfs(self):
        self.round_trip("/dev/shm", relocate=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] in ("seed", "hit", "warm"):
        _stage(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[1])
    else:
        unittest.main()
