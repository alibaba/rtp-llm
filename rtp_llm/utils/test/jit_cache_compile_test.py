"""A real Triton compile, local reuse, and remote snapshot restore."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store

RESULT_PREFIX = "JIT_CACHE_COMPILE_RESULT="


def run_kernel(bias: int) -> None:
    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("The real JIT cache test requires a CUDA GPU")

    @triton.jit
    def add_bias(x, out, n: tl.constexpr, bias: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        values = tl.load(x + offsets, offsets < n, other=0)
        tl.store(out + offsets, values + bias, offsets < n)

    cache_hits = []

    def observe_compile(*, cache_hit: bool, **_):
        cache_hits.append(cache_hit)

    x = torch.arange(1025, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.listener = observe_compile
        add_bias[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), bias, 256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + bias, rtol=0, atol=0)
    if not cache_hits:
        raise AssertionError(
            "Triton compilation listener did not report a cache result"
        )
    print(RESULT_PREFIX + json.dumps({"compilations": cache_hits.count(False)}))


class JitCacheCompileTest(unittest.TestCase):
    def test_compile_result_ignores_unrelated_stdout(self):
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="warning before\n"
            + RESULT_PREFIX
            + '{"compilations": 1}\nwarning after\n',
            stderr="",
        )
        with mock.patch.object(subprocess, "run", return_value=result):
            self.assertEqual(self._compile(1), 1)

    def test_compile_reuse_restore_and_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ):
            root = Path(temporary).resolve()
            os.environ["TEST_JIT_LOCAL_DIR"] = str(root / "local")
            # Only the Triton component participates in this private test scope.
            for item in jit.COMPONENTS:
                os.environ[item.env_name] = str(root / "excluded" / item.name)
            os.environ.pop("TRITON_CACHE_DIR")
            jit.setup_jit_cache_env.cache_clear()
            self.addCleanup(jit.setup_jit_cache_env.cache_clear)
            (root / "remote").mkdir()
            config = SimpleNamespace(
                manage_jit_cache=True,
                remote_jit_dir=str(root / "remote"),
                jit_cache_setup_timeout_s=30,
            )
            producer = jit.start_from_config(config)
            self.assertIsNotNone(producer, "JIT cache bootstrap must succeed")
            try:
                self.assertEqual(
                    [c.name for c in producer.scope.components], ["triton"]
                )
                self.assertGreater(self._compile(1), 0, "cold run must really compile")
                self.assertEqual(
                    self._compile(1), 0, "a fresh process must reuse the local cache"
                )
            finally:
                producer.stop()
            snapshots = list(
                producer.store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}")
            )
            self.assertTrue(
                snapshots, "normal shutdown must publish the compiled cache"
            )
            compiled = {
                name: path.read_bytes()
                for name, path in producer._snapshot_files().items()
            }
            self.assertTrue(any(name.endswith(".cubin") for name in compiled))
            shutil.rmtree(producer.scope.root)  # this test owns the entire scope
            consumer = jit.start_from_config(config)
            self.assertIsNotNone(
                consumer, "restoring the same absolute scope must succeed"
            )
            try:
                self.assertIsNotNone(consumer._restored)
                self.assertEqual(
                    {
                        name: path.read_bytes()
                        for name, path in consumer._snapshot_files().items()
                    },
                    compiled,
                )
                self.assertEqual(
                    self._compile(1), 0, "restored artifacts must be reusable"
                )
                self.assertGreater(
                    self._compile(2), 0, "new specialization must compile"
                )
                self.assertEqual(
                    self._compile(2), 0, "new specialization must be reusable"
                )
            finally:
                consumer.stop()

    def _compile(self, bias: int) -> int:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(bias)],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reports = [
            line[len(RESULT_PREFIX) :]
            for line in result.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        self.assertEqual(len(reports), 1, result.stdout + result.stderr)
        count = json.loads(reports[0])["compilations"]
        print(f"Triton specialization {bias}: {count} compilations")
        return count


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_kernel(int(sys.argv[2]))
    else:
        unittest.main()
