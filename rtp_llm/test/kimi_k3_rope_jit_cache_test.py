"""Exercise remote JIT restore with the real strided K3 vision RoPE kernel."""

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

RESULT_PREFIX = "K3_ROPE_JIT_RESULT="


def run_rope(head_dim: int) -> None:
    import torch
    import triton

    from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
        maybe_fused_apply_rope,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("K3 vision RoPE JIT probe requires CUDA")
    torch.manual_seed(29)
    qkv = torch.randn(7, 3, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.unbind(1)
    assert q.stride(0) != q.numel() // q.shape[0]
    angles = torch.randn(7, head_dim // 2, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    hits = []
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.listener = lambda *, cache_hit, **_: hits.append(
            cache_hit
        )
        result = maybe_fused_apply_rope(q, k, freqs)
    if result is None or not hits:
        raise AssertionError("the fused K3 RoPE kernel did not launch")
    for actual, source in zip(result, (q, k)):
        complex_source = torch.view_as_complex(source.float().reshape(7, 5, -1, 2))
        reference = torch.view_as_real(complex_source * freqs[:, None, :])
        torch.testing.assert_close(
            actual, reference.flatten(-2).to(actual.dtype), rtol=0.02, atol=0.02
        )
    torch.cuda.synchronize()
    print(RESULT_PREFIX + json.dumps({"compilations": hits.count(False)}))


class KimiK3RopeJitCacheTest(unittest.TestCase):
    def test_cold_compile_restore_and_new_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ):
            root = Path(temporary).resolve()
            os.environ["MODEL_TYPE"] = "kimi_k3"
            os.environ["TEST_JIT_LOCAL_DIR"] = str(root / "local")
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
            self.assertIsNotNone(producer)
            try:
                self.assertEqual([item.name for item in producer.scope.components], ["triton"])
                self.assertGreater(self._compile(64), 0)
                self.assertEqual(self._compile(64), 0)
            finally:
                producer.stop()
            snapshots = list(
                producer.store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}")
            )
            self.assertTrue(snapshots)
            files = producer._snapshot_files()
            self.assertTrue(any(name.endswith(".cubin") for name in files))
            shutil.rmtree(producer.scope.root)
            consumer = jit.start_from_config(config)
            self.assertIsNotNone(consumer)
            try:
                self.assertIsNotNone(consumer._restored)
                self.assertEqual(self._compile(64), 0)
                self.assertGreater(self._compile(128), 0)
            finally:
                consumer.stop()

    def _compile(self, head_dim: int) -> int:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(head_dim)],
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
        return json.loads(reports[0])["compilations"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_rope(int(sys.argv[2]))
    else:
        unittest.main()
